import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler
from pprint import pformat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None
from tqdm import tqdm
import time

from train_config import TrainingConfig
from data_loader import (
    create_data_loaders,
    create_random_sr_loaders,
    OnTheFlyOptions,
)
from KawaiiSR.KawaiiSR import KawaiiSR
from loss.KawaiiLoss import KawaiiLoss, DiscriminatorLoss

try:
    from Discriminator.UNetDiscriminatorSN import UNetDiscriminatorSN
except Exception:
    UNetDiscriminatorSN = None


class _CUDAPrefetcher:
    """CUDA 异步预取，减少数据拷贝对训练主流的阻塞。"""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = torch.device(device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.device.type != 'cuda':
            yield from self.loader
            return

        stream = torch.cuda.Stream(device=self.device)
        loader_iter = iter(self.loader)
        next_batch = None
        finished = False

        def _move_to_device(batch):
            if torch.is_tensor(batch):
                return batch.to(self.device, non_blocking=True)
            if isinstance(batch, (list, tuple)):
                return type(batch)(_move_to_device(item) for item in batch)
            if isinstance(batch, dict):
                return {k: _move_to_device(v) for k, v in batch.items()}
            return batch

        def _prefetch():
            nonlocal next_batch, finished
            try:
                batch = next(loader_iter)
            except StopIteration:
                finished = True
                next_batch = None
                return
            with torch.cuda.stream(stream):
                next_batch = _move_to_device(batch)

        _prefetch()
        while not finished:
            torch.cuda.current_stream(device=self.device).wait_stream(stream)
            batch = next_batch
            _prefetch()
            if batch is not None:
                yield batch


class KawaiiTrainer:
    """单阶段训练器（已精简去除三阶段相关逻辑）"""

    def __init__(self, config: TrainingConfig):
        self.cfg = config

        # 目录
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == 'cpu' else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            if getattr(self.cfg, 'allow_tf32', True):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except AttributeError:
                    pass
        self._non_blocking = self.device.type == 'cuda' and bool(self.cfg.pin_memory)

        # 模型与损失
        self.model = KawaiiSR(**self.cfg.model_config).to(self.device)
        self._maybe_compile_model()
        # 支持全局损失缩放：在不改变各损失相对比例的情况下统一放大/缩小总梯度规模
        self.loss_global_scale = float(getattr(self.cfg, 'loss_global_scale', 1.0))
        if self.loss_global_scale <= 0:
            raise ValueError('loss_global_scale 必须大于 0。')
        self.loss_fn = KawaiiLoss(
            lambda_char=self.cfg.loss_weights.get('pixel', 1.0) * self.loss_global_scale,
            lambda_lap=self.cfg.loss_weights.get('frequency', 0.0) * self.loss_global_scale,
            lambda_perc=self.cfg.loss_weights.get('perceptual', 0.0) * self.loss_global_scale,
            lambda_adv=self.cfg.loss_weights.get('adversarial', 0.0) * self.loss_global_scale,
            lambda_vgg=self.cfg.loss_weights.get('vgg', 0.0) * self.loss_global_scale,
            enable_anime_loss=bool(self.cfg.loss_options.get('enable_anime_loss', False)),
            device=str(self.device)
        )

        # 可选判别器
        self.gan_enabled = bool(self.cfg.gan.get('enabled', False) and self.cfg.loss_weights.get('adversarial', 0.0) > 0)
        if self.gan_enabled and UNetDiscriminatorSN is not None:
            self.disc = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(self.device)
            self.disc_opt = optim.AdamW(self.disc.parameters(), lr=float(self.cfg.gan.get('discriminator_lr', 1e-4)), betas=(0.5, 0.999))
            # 使用统一的 Hinge 判别器损失实现，避免与 KawaiiLoss 中的生成器对抗项不一致
            self.disc_loss_fn = DiscriminatorLoss(device=str(self.device))
        else:
            self.disc, self.disc_opt = None, None
            self.disc_loss_fn = None

        # 优化器与调度器（调度器在 train() 中根据 batch 数动态初始化）
        self.opt = optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        self.sched: Optional[optim.lr_scheduler._LRScheduler] = None

        # AMP、日志、指标
        self.scaler = GradScaler(enabled=self.cfg.mixed_precision)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = None
        if getattr(self.cfg, 'save_on_lpips', False):
            if LearnedPerceptualImagePatchSimilarity is None:
                raise RuntimeError('LPIPS 指标已启用，但当前环境缺少 torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity。请安装/升级 torchmetrics 后再试。')
            try:
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)
                self._console('[Console] LPIPS 指标已启用 (用于保存判断)。')
            except Exception as exc:
                raise RuntimeError(f'LPIPS 初始化失败: {exc}') from exc

        # 日志系统（文件旋转 + 控制台）
        self.logger = logging.getLogger(f'KawaiiTrainer[{id(self)}]')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        log_path = Path(self.cfg.checkpoint_dir) / 'training.log'
        if not self.logger.handlers:
            fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
            ch = logging.StreamHandler()
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # 训练状态
        self.global_step = 0
        self.best_metrics: Dict[str, float] = {}
        # 早停计数（以“验证轮次”为单位统计无提升次数）
        self._no_improve_validations = 0

        gan_status_bits = []
        if not self.cfg.gan.get('enabled', False):
            gan_status_bits.append('gan.enabled=false')
        if self.cfg.loss_weights.get('adversarial', 0.0) <= 0:
            gan_status_bits.append('adversarial weight<=0')
        if UNetDiscriminatorSN is None:
            gan_status_bits.append('UNetDiscriminatorSN import failed')
        if self.gan_enabled:
            self._console('[Console] GAN path active: discriminator + adversarial loss engaged.')
        else:
            reason = ', '.join(gan_status_bits) if gan_status_bits else 'unknown reason'
            self._console(f'[Console] GAN path disabled ({reason}).')

    def _console(self, message: str):
        try:
            tqdm.write(message)
        except Exception:
            print(message)

    def _maybe_compile_model(self):
        compile_enabled = bool(getattr(self.cfg, 'torch_compile', False))
        if not compile_enabled:
            self._console('[Console] torch.compile 未启用（配置为 false）。')
            return
        if not hasattr(torch, 'compile'):
            self._console('[Console] 当前 PyTorch 版本不支持 torch.compile，保持 eager 模式。')
            return

        try:
            self.model = torch.compile(self.model)
            self._console('[Console] torch.compile 已启用，生成器将以编译模式训练。')
        except Exception as exc:
            self._console(f'[Console] torch.compile 失败 ({exc})，已回退至 eager 模式。')

    def _init_scheduler(self, total_steps: int) -> None:
        """按 batch 建立余弦退火调度器。"""
        if total_steps <= 0:
            self.sched = None
            return
        self.sched = optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=int(total_steps),
            eta_min=self.cfg.learning_rate * 0.1,
        )

    def _wrap_loader(self, loader):
        if getattr(self.cfg, 'enable_cuda_prefetch', False) and self.device.type == 'cuda':
            return _CUDAPrefetcher(loader, device=self.device)
        return loader

    def load_weights(self, path: str, *, strict: bool = True) -> Dict[str, Any]:
        """加载仅包含模型权重的文件，可用于全新训练的热身。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            state_dict = payload['model_state_dict']
        else:
            state_dict = payload
        self.model.load_state_dict(state_dict, strict=strict)
        return payload if isinstance(payload, dict) else {'model_state_dict': state_dict}

    def _load_resume_state(self, path: str) -> Dict[str, Any]:
        """加载恢复训练所需的优化器/调度器等状态。"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        required_keys = ['optimizer_state_dict', 'global_step', 'epoch']
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise KeyError(f'恢复训练状态缺少必要字段: {missing}')

        self.opt.load_state_dict(state['optimizer_state_dict'])

        if self.scaler and state.get('scaler_state_dict'):
            self.scaler.load_state_dict(state['scaler_state_dict'])

        self.best_metrics = state.get('best_metrics', {})
        self.global_step = int(state.get('global_step', 0))
        self._no_improve_validations = int(state.get('no_improve_validations', 0))

        return state

    def _compute_metrics(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        # 输入/输出均已在 [0,1]
        sr01 = sr.clamp(0.0, 1.0)
        hr01 = hr.clamp(0.0, 1.0)
        mse = F.mse_loss(sr01, hr01)
        metrics = {
            'psnr': self.psnr(sr01, hr01).item(),
            'ssim': self.ssim(sr01, hr01).item(),
            'mse': mse.item(),
        }
        if self.lpips is not None:
            try:
                lp = self.lpips(sr01, hr01).item()
            except Exception:
                lp = float('nan')
            metrics['lpips'] = lp
        return metrics

    def _check_improvement(self, current: Dict[str, float], best: Dict[str, float]) -> bool:
        """多指标保存策略：任一被启用的指标取得提升即可。
        提升判定：
          - PSNR / SSIM: 变大
          - LPIPS / val_loss: 变小
        """
        improved_flags = []
        if getattr(self.cfg, 'save_on_psnr', True):
            if current['psnr'] > best.get('psnr', float('-inf')):
                improved_flags.append('psnr')
        if getattr(self.cfg, 'save_on_ssim', False):
            if current['ssim'] > best.get('ssim', float('-inf')):
                improved_flags.append('ssim')
        if getattr(self.cfg, 'save_on_val_loss', False):
            if current['loss'] < best.get('loss', float('inf')):
                improved_flags.append('val_loss')
        if getattr(self.cfg, 'save_on_lpips', False) and 'lpips' in current and current['lpips'] is not None:
            prev = best.get('lpips', float('inf'))
            if current['lpips'] < prev:
                improved_flags.append('lpips')
        if improved_flags:
            self._console(f"[Console] Improvement triggers: {', '.join(improved_flags)}")
            return True
        return False

    def _train_disc_step(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        if not self.gan_enabled or self.disc is None or self.disc_opt is None:
            return 0.0
        self.disc.train()
        self.disc_opt.zero_grad(set_to_none=True)
        # 统一使用 Hinge 判别器损失
        real_logits = self.disc(real)
        fake_logits = self.disc(fake.detach())
        if self.disc_loss_fn is not None:
            loss_d = self.disc_loss_fn(real_logits, fake_logits)
        else:
            panic("Discriminator loss function not defined.")
        loss_d.backward()
        self.disc_opt.step()
        return float(loss_d.item())

    def _compute_total_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # fake logits: 若未启用GAN，提供全零占位以兼容KawaiiLoss需要的输入
        if self.gan_enabled and self.disc is not None:
            fake_logits = self.disc(sr)
        else:
            fake_logits = torch.zeros(sr.shape[0], 1, device=sr.device)

        total, parts = self.loss_fn(sr, hr, fake_logits)

        # 可选生成器对抗项（若 KawaiiLoss 已包含 adversarial 则 parts 中自带；这里不再重复叠加）
        return total, {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in parts.items()}

    def _save_checkpoint(self, epoch: int, is_best: bool):
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        weights_payload = {
            'model_state_dict': self.model.state_dict(),
            'config': dict(self.cfg.__dict__),
        }
        state_payload = {
            'epoch': int(epoch),
            'global_step': int(self.global_step),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.sched.state_dict() if self.sched else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metrics': dict(self.best_metrics),
            'no_improve_validations': int(self._no_improve_validations),
            'scheduler_total_steps': getattr(self.sched, 'T_max', None) if self.sched else None,
        }

        torch.save(weights_payload, checkpoint_dir / 'last_weights.pth')
        torch.save(state_payload, checkpoint_dir / 'last_state.pth')

        if is_best:
            torch.save(weights_payload, checkpoint_dir / 'best_weights.pth')
            torch.save(state_payload, checkpoint_dir / 'best_state.pth')
        dt = time.perf_counter() - t0
        try:
            self.logger.info(f"Checkpoint saved (epoch={epoch+1}, best={is_best}) in {dt:.2f}s")
        except Exception:
            self._console(f"[Console] Checkpoint saved in {dt:.2f}s")
       
        if self.gan_enabled and self.disc is not None and self.disc_opt is not None:
            disc_weights_payload = {'model_state_dict': self.disc.state_dict()}
            disc_state_payload = {'optimizer_state_dict': self.disc_opt.state_dict()}
            
            torch.save(disc_weights_payload, checkpoint_dir / 'disc_weight.pth')
            torch.save(disc_state_payload, checkpoint_dir / 'disc_state.pth')

        dt = time.perf_counter() - t0
        log_msg = (
            f"Checkpoint saved (epoch={epoch+1}, best={is_best}"
            f"{' +Disc' if self.gan_enabled else ''}) in {dt:.2f}s"
        )
        try:
            self.logger.info(log_msg)
        except Exception:
            self._console(f"[Console] {log_msg}")

    def train(
        self,
        *,
        load_weights: Optional[str] = None,
        resume_weights: Optional[str] = None,
        resume_state: Optional[str] = None,
    ):
        if (resume_weights is None) != (resume_state is None):
            raise ValueError('恢复训练需要同时提供 resume_weights 与 resume_state。')
        if load_weights and resume_weights:
            raise ValueError('load_weights 不可与 resume_* 参数同时使用。')

        resume_state_payload: Optional[Dict[str, Any]] = None
        pending_scheduler_state: Optional[Dict[str, Any]] = None
        start_epoch = 0

        if load_weights:
            if not os.path.exists(load_weights):
                raise FileNotFoundError(f'权重文件不存在: {load_weights}')
            self._console(f"[Console] 加载预训练权重: {load_weights}")
            self.load_weights(load_weights)
            # 新一轮训练，重置历史指标
            self.best_metrics = {}
            self.global_step = 0
            self._no_improve_validations = 0

        if resume_weights:
            if not os.path.exists(resume_weights):
                raise FileNotFoundError(f'恢复训练的权重文件不存在: {resume_weights}')
            if not os.path.exists(resume_state):
                raise FileNotFoundError(f'恢复训练的状态文件不存在: {resume_state}')

            self._console(f"[Console] 恢复训练 - 加载权重: {resume_weights}")
            self.load_weights(resume_weights)

            self._console(f"[Console] 恢复训练 - 加载状态: {resume_state}")
            resume_state_payload = self._load_resume_state(resume_state)
            pending_scheduler_state = resume_state_payload.get('scheduler_state_dict')
            start_epoch = int(resume_state_payload.get('epoch', -1)) + 1
            if start_epoch < 0:
                start_epoch = 0
            
            if self.gan_enabled:
                self._console("[Console] GAN已启用, 尝试从固定路径恢复判别器...")
                checkpoint_dir = Path(self.cfg.checkpoint_dir)
                disc_weights_path = checkpoint_dir / 'disc_weight.pth'
                disc_state_path = checkpoint_dir / 'disc_state.pth'

                if disc_weights_path.exists() and disc_state_path.exists():
                    try:
                        # 加载判别器权重
                        disc_weights_payload = torch.load(disc_weights_path, map_location=self.device)
                        self.disc.load_state_dict(disc_weights_payload['model_state_dict'])

                        # 加载判别器优化器状态
                        disc_state_payload = torch.load(disc_state_path, map_location=self.device)
                        self.disc_opt.load_state_dict(disc_state_payload['optimizer_state_dict'])
                        
                        self._console("[Console] 成功从 disc_weight.pth 和 disc_state.pth 恢复判别器。")
                    except Exception as e:
                        self._console(f"[Console] 加载判别器检查点失败 ({e})，将使用新的判别器。")
                else:
                    self._console("[Console] 未找到判别器检查点，将使用新的判别器进行训练。")

        config_text = pformat(vars(self.cfg), indent=2, width=120)
        self._console('[Console] Active training config:\n' + config_text)
        self.logger.info('Active training config:\n%s', config_text)

        # 数据
        if self.cfg.use_online_data:
            options_kwargs = dict(self.cfg.online_data_options or {})
            try:
                options = OnTheFlyOptions(**options_kwargs)
            except TypeError as exc:
                self._console(f"[Console] online_data_options 参数错误 {exc}，使用默认设置。")
                options = OnTheFlyOptions()

            train_loader, val_loader = create_random_sr_loaders(
                train_image_dir=self.cfg.train_data_path,
                val_image_dir=self.cfg.val_data_path,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=self.cfg.dataloader_prefetch_factor,
                persistent_workers=self.cfg.dataloader_persistent_workers,
                options=options,
            )
            pipeline_mode = 'online'
        else:
            train_loader, val_loader = create_data_loaders(
                train_data_path=self.cfg.train_data_path,
                val_data_path=self.cfg.val_data_path,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=self.cfg.dataloader_prefetch_factor,
                persistent_workers=self.cfg.dataloader_persistent_workers,
                train_csv='dataset_index.csv',
                val_csv='dataset_index.csv',
            )
            pipeline_mode = 'offline'

        has_validation = val_loader is not None
        val_size = len(val_loader.dataset) if has_validation else 0
        self._console(
            f"[Console] Dataset summary | mode: {pipeline_mode} | train: {len(train_loader.dataset)} | val: {val_size} | batch: {self.cfg.batch_size}"
        )

        try:
            steps_per_epoch = len(train_loader)
        except TypeError as exc:
            raise RuntimeError('训练 DataLoader 未实现 __len__，无法按 batch 调整学习率。') from exc
        if steps_per_epoch <= 0:
            raise RuntimeError('训练 DataLoader 未包含任何 batch。')

        scheduler_total_steps = steps_per_epoch * self.cfg.epochs
        if resume_state_payload and resume_state_payload.get('scheduler_total_steps'):
            scheduler_total_steps = int(resume_state_payload['scheduler_total_steps'])

        self._init_scheduler(scheduler_total_steps)
        if pending_scheduler_state:
            if not self.sched:
                raise RuntimeError('恢复训练需要有效的调度器，但当前未初始化。')
            self.sched.load_state_dict(pending_scheduler_state)

        if not self.best_metrics:
            self.best_metrics = {}
        # 读取早停耐心值（<=0 视为关闭早停）
        patience = int(getattr(self.cfg, 'early_stopping_patience', 0) or 0)

        if start_epoch >= self.cfg.epochs:
            self._console('[Console] 起始 epoch 已达到或超过配置上限，训练直接结束。')
            return

        did_any_validation = False
        final_epoch_idx = start_epoch - 1

        for epoch in range(start_epoch, self.cfg.epochs):
            final_epoch_idx = epoch
            self.model.train()
            epoch_start_wall = time.perf_counter()
            # 训练期内统计
            train_loss_sum = 0.0
            train_psnr_sum = 0.0
            train_ssim_sum = 0.0
            train_disc_loss_sum = 0.0
            batch_count = 0
            # 使用 dynamic_ncols=True 让 tqdm 自适应当前终端宽度，避免固定 100 导致信息被截断
            train_iterable = self._wrap_loader(train_loader)
            pbar = tqdm(
                train_iterable,
                desc=f"Train {epoch+1}/{self.cfg.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            data_time_sum = 0.0
            step_time_sum = 0.0
            batch_fetch_start = time.perf_counter()
            for i, (lr_img, hr_img) in enumerate(pbar):
                data_wait = time.perf_counter() - batch_fetch_start
                data_time_sum += data_wait

                if lr_img.device != self.device:
                    lr_img = lr_img.to(self.device, non_blocking=self._non_blocking)
                if hr_img.device != self.device:
                    hr_img = hr_img.to(self.device, non_blocking=self._non_blocking)

                compute_start = time.perf_counter()

                # 判别器更新（若启用）
                disc_loss = 0.0
                if self.gan_enabled and self.disc is not None:
                    # 先用上一轮的生成器输出（不可用），这里简单采用先前生成的思路不现实；直接用当前生成器前向一次用于 D
                    with torch.no_grad():
                        fake_img_for_d = self.model(lr_img)
                    disc_loss = self._train_disc_step(real=hr_img, fake=fake_img_for_d)

                # 生成器更新
                self.opt.zero_grad(set_to_none=True)
                # 冻结判别器参数，避免在G步对D累积梯度
                disc_prev_requires_grad = []
                if self.disc is not None:
                    for p in self.disc.parameters():
                        disc_prev_requires_grad.append(p.requires_grad)
                        p.requires_grad = False

                # 在生成器步中将判别器切到 eval，避免 BN/状态在 G 反传时被更新
                disc_prev_mode = None
                if self.disc is not None:
                    disc_prev_mode = self.disc.training
                    self.disc.eval()

                if self.cfg.mixed_precision:
                    with autocast():
                        sr_img = self.model(lr_img)
                        total_loss, parts = self._compute_total_loss(sr_img, hr_img)
                    self.scaler.scale(total_loss).backward()
                    if self.cfg.gradient_clip_norm and self.cfg.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    sr_img = self.model(lr_img)
                    total_loss, parts = self._compute_total_loss(sr_img, hr_img)
                    total_loss.backward()
                    if self.cfg.gradient_clip_norm and self.cfg.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)
                    self.opt.step()

                # 恢复判别器训练/评估模式
                if self.disc is not None and disc_prev_mode is not None:
                    self.disc.train(disc_prev_mode)

                if self.sched:
                    self.sched.step()

                # 恢复判别器参数的 requires_grad 标志
                if self.disc is not None and disc_prev_requires_grad:
                    for p, flag in zip(self.disc.parameters(), disc_prev_requires_grad):
                        p.requires_grad = flag

                # 指标
                metrics = self._compute_metrics(sr_img.detach(), hr_img)
                # 进度条展示：总损失 + 各启用子损失(已乘以权重)
                weighted_parts_display = {}
                # parts 中保存的是未乘权重的原始值，这里按配置再次乘权重输出
                for name, raw_val in parts.items():
                    if name == 'total_g_loss':
                        continue  # 跳过总和字段
                    # 与实际训练一致：显示值也乘以全局缩放系数
                    weight = float(self.cfg.loss_weights.get(name, 0.0)) * self.loss_global_scale
                    if weight <= 0:
                        continue  # 未启用
                    try:
                        weighted_val = raw_val * weight
                    except Exception:
                        continue
                    # 控制长度，避免 tqdm 后缀过长；保留 3 位小数
                    weighted_parts_display[name] = f"{weighted_val:.3f}"

                postfix_dict = {
                    'loss': f"{float(total_loss.item()):.4f}",
                    'psnr': f"{metrics['psnr']:.2f}",
                    'ssim': f"{metrics['ssim']:.4f}",
                }
                # 将子损失加入（顺序：像素、频域、感知、对抗、vgg 若存在）
                preferred_order = ['pixel', 'frequency', 'perceptual', 'adversarial', 'vgg']
                for key in preferred_order:
                    if key in weighted_parts_display:
                        postfix_dict[key] = weighted_parts_display[key]
                # 其余（如果有自定义扩展）
                for k, v in weighted_parts_display.items():
                    if k not in postfix_dict:
                        postfix_dict[k] = v
                pbar.set_postfix(postfix_dict)

                # 累计
                train_loss_sum += float(total_loss.item())
                train_psnr_sum += metrics['psnr']
                train_ssim_sum += metrics['ssim']
                train_disc_loss_sum += float(disc_loss)
                batch_count += 1

                # 仅日志输出

                self.global_step += 1
                step_time_sum += time.perf_counter() - compute_start
                batch_fetch_start = time.perf_counter()

            # 计算训练平均
            if batch_count > 0:
                avg_train_loss = train_loss_sum / batch_count
                avg_train_psnr = train_psnr_sum / batch_count
                avg_train_ssim = train_ssim_sum / batch_count
                avg_train_disc = train_disc_loss_sum / batch_count if self.gan_enabled else 0.0
                avg_data_time = data_time_sum / batch_count
                avg_step_time = step_time_sum / batch_count
            else:
                avg_train_loss = avg_train_psnr = avg_train_ssim = avg_train_disc = 0.0
                avg_data_time = avg_step_time = 0.0

            # 仅日志输出

            # 验证
            if has_validation and (epoch % self.cfg.val_every) == 0:
                self.model.eval()
                val_start_wall = time.perf_counter()
                val_losses = []
                val_psnr = []
                val_ssim = []
                val_lpips = [] if self.lpips is not None else None
                with torch.no_grad():
                    for lr_img, hr_img in tqdm(
                        val_loader,
                        desc='Validate',
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        if lr_img.device != self.device:
                            lr_img = lr_img.to(self.device, non_blocking=self._non_blocking)
                        if hr_img.device != self.device:
                            hr_img = hr_img.to(self.device, non_blocking=self._non_blocking)
                        sr_img = self.model(lr_img)
                        total_loss, _ = self._compute_total_loss(sr_img, hr_img)
                        metrics = self._compute_metrics(sr_img, hr_img)
                        val_losses.append(float(total_loss.item()))
                        val_psnr.append(metrics['psnr'])
                        val_ssim.append(metrics['ssim'])
                        if val_lpips is not None and 'lpips' in metrics:
                            val_lpips.append(metrics['lpips'])

                avg_loss = float(sum(val_losses) / max(1, len(val_losses)))
                avg_psnr = float(sum(val_psnr) / max(1, len(val_psnr)))
                avg_ssim = float(sum(val_ssim) / max(1, len(val_ssim)))
                avg_lpips = None
                if val_lpips is not None and len(val_lpips) > 0:
                    avg_lpips = float(sum(val_lpips) / len(val_lpips))
                val_time_wall = time.perf_counter() - val_start_wall
                epoch_time_wall = time.perf_counter() - epoch_start_wall
                imgs_per_sec = (batch_count * self.cfg.batch_size) / max(1e-6, (data_time_sum + step_time_sum))

                # 仅日志输出

                # 日志输出（含训练与验证平均）
                self.logger.info(
                    f"Epoch {epoch+1:03d}/{self.cfg.epochs} | "
                    f"Train: loss={avg_train_loss:.4f} psnr={avg_train_psnr:.2f} ssim={avg_train_ssim:.4f}"
                    + (f" d_loss={avg_train_disc:.4f}" if self.gan_enabled else "") +
                    f" || Val: loss={avg_loss:.4f} psnr={avg_psnr:.2f} ssim={avg_ssim:.4f} | lr={self.opt.param_groups[0]['lr']:.2e}"
                    + f" | data_t={avg_data_time*1000:.1f}ms step_t={avg_step_time*1000:.1f}ms val_t={val_time_wall:.1f}s epoch_t={epoch_time_wall:.1f}s imgs/s~{imgs_per_sec:.1f}"
                )

                # 保存
                current_metrics = {
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'loss': avg_loss,
                    'lpips': avg_lpips if avg_lpips is not None else None,
                }
                improved = self._check_improvement(current_metrics, self.best_metrics)
                if improved:
                    self.best_metrics.update(current_metrics)
                    self._no_improve_validations = 0
                else:
                    self._no_improve_validations += 1

                # 始终刷新 last.pth；若提升则同时刷新 best.pth
                self._save_checkpoint(epoch=epoch, is_best=improved)
                did_any_validation = True

                # 早停判断：按“验证轮次”计数，而非所有 epoch
                if patience > 0 and self._no_improve_validations >= patience:
                    self._console(
                        f"[Console] Early stopping triggered after {self._no_improve_validations} validations without improvement (patience={patience})."
                    )
                    # 记录并结束训练循环
                    self.logger.info(
                        f"Early stopping at epoch {epoch+1}: no improvement in {self._no_improve_validations} validation(s), patience={patience}"
                    )
                    break
            else:
                # 没做验证时也记录训练平均
                self.logger.info(
                    f"Epoch {epoch+1:03d}/{self.cfg.epochs} | "
                    f"Train: loss={avg_train_loss:.4f} psnr={avg_train_psnr:.2f} ssim={avg_train_ssim:.4f}"
                    + (f" d_loss={avg_train_disc:.4f}" if self.gan_enabled else "") +
                    f" | lr={self.opt.param_groups[0]['lr']:.2e} data_t={avg_data_time*1000:.1f}ms step_t={avg_step_time*1000:.1f}ms"
                )

        if not did_any_validation and final_epoch_idx >= 0:
            self._console('[Console] 未执行验证，仍会保存最后一次权重。')
            self._save_checkpoint(epoch=final_epoch_idx, is_best=False)

    # 结束

        self._console(
            f"[Console] Training finished. Best PSNR: {self.best_metrics.get('psnr', 0.0):.2f}dB | checkpoints saved to {self.cfg.checkpoint_dir}"
        )
