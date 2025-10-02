import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from train_config import TrainingConfig
from data_loader import create_data_loaders
from KawaiiSR.KawaiiSR import KawaiiSR
from loss.KawaiiLoss import KawaiiLoss

try:
    from Discriminator.UNetDiscriminatorSN import UNetDiscriminatorSN
except Exception:
    UNetDiscriminatorSN = None


class KawaiiTrainer:
    """单阶段训练器（已精简去除三阶段相关逻辑）"""

    def __init__(self, config: TrainingConfig):
        self.cfg = config

        # 目录
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == 'cpu' else 'cpu')

        # 模型与损失
        self.model = KawaiiSR(**self.cfg.model_config).to(self.device)
        self._maybe_compile_model()
        self.loss_fn = KawaiiLoss(
            lambda_char=self.cfg.loss_weights.get('pixel', 1.0),
            lambda_lap=self.cfg.loss_weights.get('frequency', 0.0),
            lambda_perc=self.cfg.loss_weights.get('perceptual', 0.0),
            lambda_adv=self.cfg.loss_weights.get('adversarial', 0.0),
            lambda_vgg=self.cfg.loss_weights.get('vgg', 0.0),
            enable_anime_loss=bool(self.cfg.loss_options.get('enable_anime_loss', False)),
            device=str(self.device)
        )

        # 可选判别器
        self.gan_enabled = bool(self.cfg.gan.get('enabled', False) and self.cfg.loss_weights.get('adversarial', 0.0) > 0)
        if self.gan_enabled and UNetDiscriminatorSN is not None:
            self.disc = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(self.device)
            self.disc_opt = optim.Adam(self.disc.parameters(), lr=float(self.cfg.gan.get('discriminator_lr', 1e-4)), betas=(0.5, 0.999))
        else:
            self.disc, self.disc_opt = None, None

        # 优化器与调度器
        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.cfg.epochs, eta_min=self.cfg.learning_rate * 0.1)

        # AMP、日志、指标
        self.scaler = GradScaler(enabled=self.cfg.mixed_precision)
        self.writer = SummaryWriter(log_dir=str(Path(self.cfg.tensorboard_dir)))
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

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
        compile_cfg = getattr(self.cfg, 'torch_compile', True)
        if not compile_cfg:
            return
        if not hasattr(torch, 'compile'):
            self._console('[Console] torch.compile 不可用，保持 eager 模式。')
            return

        compile_kwargs = compile_cfg if isinstance(compile_cfg, dict) else {}
        try:
            self.model = torch.compile(self.model, **compile_kwargs)
            self._console('[Console] torch.compile 已启用，生成器将以编译模式训练。')
        except Exception as exc:
            self._console(f'[Console] torch.compile 失败 ({exc}），已回退至 eager 模式。')

    def _compute_metrics(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        # 输入/输出均已在 [0,1]
        sr01 = sr.clamp(0.0, 1.0)
        hr01 = hr.clamp(0.0, 1.0)
        mse = F.mse_loss(sr01, hr01)
        return {
            'psnr': self.psnr(sr01, hr01).item(),
            'ssim': self.ssim(sr01, hr01).item(),
            'mse': mse.item(),
        }

    def _train_disc_step(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        if not self.gan_enabled or self.disc is None or self.disc_opt is None:
            return 0.0
        self.disc.train()
        self.disc_opt.zero_grad(set_to_none=True)
        # HingeGAN-like losses if present in KawaiiLoss; otherwise simple BCE-like signals
        real_pred = self.disc(real)
        fake_pred = self.disc(fake.detach())
        # 使用最简单的hinge实现（不依赖外部loss文件）
        loss_real = torch.relu(1.0 - real_pred).mean()
        loss_fake = torch.relu(1.0 + fake_pred).mean()
        loss_d = loss_real + loss_fake
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
        payload = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.sched.state_dict() if self.sched else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metrics': self.best_metrics,
            'config': self.cfg.__dict__,
        }
        last_path = Path(self.cfg.checkpoint_dir) / 'last.pth'
        torch.save(payload, last_path)
        if is_best:
            best_path = Path(self.cfg.checkpoint_dir) / 'best.pth'
            torch.save(payload, best_path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt and self.opt:
                try:
                    self.opt.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    pass
            if 'scheduler_state_dict' in ckpt and self.sched and ckpt['scheduler_state_dict']:
                try:
                    self.sched.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    pass
            if 'scaler_state_dict' in ckpt and self.scaler and ckpt['scaler_state_dict']:
                try:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                except Exception:
                    pass
            self.best_metrics = ckpt.get('best_metrics', {})
            self.global_step = ckpt.get('global_step', 0)
        else:
            # 兼容仅 model state_dict 的情况
            self.model.load_state_dict(ckpt)

    def train(self, resume: Optional[str] = None):
        # 数据
        train_loader, val_loader = create_data_loaders(
            train_data_path=self.cfg.train_data_path,
            val_data_path=self.cfg.val_data_path,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            train_csv='dataset_index.csv',
            val_csv='dataset_index.csv',
        )

        self._console(
            f"[Console] Dataset summary | train: {len(train_loader.dataset)} pairs | val: {len(val_loader.dataset)} pairs | batch: {self.cfg.batch_size}"
        )


        # 恢复
        if resume and os.path.exists(resume):
            self.load_checkpoint(resume)

        best_psnr = self.best_metrics.get('psnr', 0.0)
        # 读取早停耐心值（<=0 视为关闭早停）
        patience = int(getattr(self.cfg, 'early_stopping_patience', 0) or 0)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            # 训练期内统计
            train_loss_sum = 0.0
            train_psnr_sum = 0.0
            train_ssim_sum = 0.0
            train_disc_loss_sum = 0.0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{self.cfg.epochs}", leave=False, ncols=100)
            for i, (lr_img, hr_img) in enumerate(pbar):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

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

                # 恢复判别器参数的 requires_grad 标志
                if self.disc is not None and disc_prev_requires_grad:
                    for p, flag in zip(self.disc.parameters(), disc_prev_requires_grad):
                        p.requires_grad = flag

                # 指标
                metrics = self._compute_metrics(sr_img.detach(), hr_img)
                pbar.set_postfix({
                    'loss': f"{float(total_loss.item()):.4f}",
                    'psnr': f"{metrics['psnr']:.2f}",
                    'ssim': f"{metrics['ssim']:.4f}",
                })

                # 累计
                train_loss_sum += float(total_loss.item())
                train_psnr_sum += metrics['psnr']
                train_ssim_sum += metrics['ssim']
                train_disc_loss_sum += float(disc_loss)
                batch_count += 1

                # 批日志（TensorBoard）
                if self.writer and (self.global_step % self.cfg.log_every == 0):
                    self.writer.add_scalar('train/total_loss', float(total_loss.item()), self.global_step)
                    for k, v in parts.items():
                        self.writer.add_scalar(f'train/loss_{k}', float(v), self.global_step)
                    self.writer.add_scalar('train/psnr', metrics['psnr'], self.global_step)
                    self.writer.add_scalar('train/ssim', metrics['ssim'], self.global_step)
                    if self.disc is not None:
                        self.writer.add_scalar('train/disc_loss', float(disc_loss), self.global_step)

                self.global_step += 1

            # 调度器步进
            if self.sched:
                self.sched.step()

            # 计算训练平均
            if batch_count > 0:
                avg_train_loss = train_loss_sum / batch_count
                avg_train_psnr = train_psnr_sum / batch_count
                avg_train_ssim = train_ssim_sum / batch_count
                avg_train_disc = train_disc_loss_sum / batch_count if self.gan_enabled else 0.0
            else:
                avg_train_loss = avg_train_psnr = avg_train_ssim = avg_train_disc = 0.0

            # 写入 epoch 级训练平均到 TensorBoard
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
                self.writer.add_scalar('epoch/train_psnr', avg_train_psnr, epoch)
                self.writer.add_scalar('epoch/train_ssim', avg_train_ssim, epoch)
                if self.gan_enabled:
                    self.writer.add_scalar('epoch/train_disc_loss', avg_train_disc, epoch)

            # 验证
            if (epoch % self.cfg.val_every) == 0:
                self.model.eval()
                val_losses = []
                val_psnr = []
                val_ssim = []
                with torch.no_grad():
                    for lr_img, hr_img in tqdm(val_loader, desc='Validate', leave=False, ncols=100):
                        lr_img = lr_img.to(self.device)
                        hr_img = hr_img.to(self.device)
                        sr_img = self.model(lr_img)
                        total_loss, _ = self._compute_total_loss(sr_img, hr_img)
                        metrics = self._compute_metrics(sr_img, hr_img)
                        val_losses.append(float(total_loss.item()))
                        val_psnr.append(metrics['psnr'])
                        val_ssim.append(metrics['ssim'])

                avg_loss = float(sum(val_losses) / max(1, len(val_losses)))
                avg_psnr = float(sum(val_psnr) / max(1, len(val_psnr)))
                avg_ssim = float(sum(val_ssim) / max(1, len(val_ssim)))

                # 写入日志
                if self.writer:
                    self.writer.add_scalar('val/loss', avg_loss, epoch)
                    self.writer.add_scalar('val/psnr', avg_psnr, epoch)
                    self.writer.add_scalar('val/ssim', avg_ssim, epoch)
                    self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], epoch)

                # 日志输出（含训练与验证平均）
                self.logger.info(
                    f"Epoch {epoch+1:03d}/{self.cfg.epochs} | "
                    f"Train: loss={avg_train_loss:.4f} psnr={avg_train_psnr:.2f} ssim={avg_train_ssim:.4f}"
                    + (f" d_loss={avg_train_disc:.4f}" if self.gan_enabled else "") +
                    f" || Val: loss={avg_loss:.4f} psnr={avg_psnr:.2f} ssim={avg_ssim:.4f} | lr={self.opt.param_groups[0]['lr']:.2e}"
                )

                # 保存
                improved = avg_psnr > best_psnr
                if improved:
                    best_psnr = avg_psnr
                    self.best_metrics = {'psnr': best_psnr, 'ssim': avg_ssim, 'loss': avg_loss}
                    self._console(
                        f"[Console] New best PSNR {best_psnr:.2f}dB (SSIM {avg_ssim:.4f}, loss {avg_loss:.4f}) at epoch {epoch+1}"
                    )
                    # 有提升则重置计数
                    self._no_improve_validations = 0
                else:
                    # 无提升则计数+1
                    self._no_improve_validations += 1
                    self._save_checkpoint(epoch=epoch, is_best=improved)

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
                    f" | lr={self.opt.param_groups[0]['lr']:.2e}"
                )

        # 结束
        if self.writer:
            self.writer.close()

        self._console(
            f"[Console] Training finished. Best PSNR: {self.best_metrics.get('psnr', 0.0):.2f}dB | checkpoints saved to {self.cfg.checkpoint_dir}"
        )
