import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

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
        self.loss_fn = KawaiiLoss(
            lambda_char=self.cfg.loss_weights.get('pixel', 1.0),
            lambda_lap=self.cfg.loss_weights.get('frequency', 0.0),
            lambda_perc=self.cfg.loss_weights.get('perceptual', 0.0),
            lambda_vgg=self.cfg.loss_weights.get('vgg', 0.0),
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

        # 日志系统（简单按 epoch 输出到文件 + 控制台）
        self.logger = logging.getLogger('KawaiiTrainer')
        self.logger.setLevel(logging.INFO)
        log_path = Path(self.cfg.checkpoint_dir) / 'training.log'
        if not self.logger.handlers:  # 防止重复添加 handler
            fh = logging.FileHandler(log_path, encoding='utf-8')
            ch = logging.StreamHandler()
            fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # 训练状态
        self.global_step = 0
        self.best_metrics: Dict[str, float] = {}

    def _compute_metrics(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        sr01 = ((sr + 1) / 2).clamp_(0.0, 1.0)
        hr01 = ((hr + 1) / 2).clamp_(0.0, 1.0)
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

        # 恢复
        if resume and os.path.exists(resume):
            self.load_checkpoint(resume)

        best_psnr = self.best_metrics.get('psnr', 0.0)

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
                self._save_checkpoint(epoch=epoch, is_best=improved)
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
