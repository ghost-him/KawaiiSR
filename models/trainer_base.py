import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from train_config import TrainingConfig, StageConfig
from data_loader import create_data_loaders

class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # 训练状态
        self.current_stage = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        self.early_stopping_counter = 0
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
        # 判别器相关（子类可选实现）
        self.discriminator = None
        self.discriminator_optimizer = None
        self.discriminator_scheduler = None
        
        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 日志记录
        self.writer = None
        self.metrics_history = defaultdict(list)
        
        # 创建必要的目录
        self._create_directories()
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 初始化评估指标
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        
    def _create_directories(self):
        """创建必要的目录"""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_tensorboard(self, stage_name: str):
        """设置TensorBoard"""
        log_dir = Path(self.config.tensorboard_dir) / stage_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
    def _setup_optimizer(self, stage_config: StageConfig):
        """设置优化器和学习率调度器"""
        # 根据是否冻结HAT来设置参数
        if stage_config.freeze_hat:
            # 只训练残差块和输出层
            trainable_params = []
            for name, param in self.model.named_parameters():
                if 'hat_model' not in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        else:
            # 训练所有参数
            trainable_params = []
            for param in self.model.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            trainable_params,
            lr=stage_config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=stage_config.epochs,
            eta_min=stage_config.learning_rate * 0.1
        )
        
    def _freeze_hat_model(self):
        """冻结HAT模型参数"""
        for name, param in self.model.named_parameters():
            if 'hat_model' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def _unfreeze_all_parameters(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def _compute_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        stage_config: StageConfig,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数（子类实现）"""
        pass
    
    def _train_discriminator_step(
        self,
        hr_images: torch.Tensor,
        sr_images: torch.Tensor
    ) -> Dict[str, float]:
        """训练判别器步骤（子类可选实现）"""
        return {}
    
    def _compute_metrics(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        使用 torchmetrics 高效计算评估指标

        Args:
            sr_images (torch.Tensor): 超分辨率图像张量，形状为 (B, C, H, W)，数值范围 [-1, 1]
            hr_images (torch.Tensor): 高分辨率（真实）图像张量，形状为 (B, C, H, W)，数值范围 [-1, 1]

        Returns:
            Dict[str, float]: 包含 'psnr', 'ssim', 'mse' 的字典
        """
        # 确保输入张量在正确的设备上
        sr_images = sr_images.to(self.device)
        hr_images = hr_images.to(self.device)

        # 反归一化到 [0, 1]，这是指标计算需要的范围
        # 使用 .clamp_() 来确保数值不会因浮点误差超出范围
        sr_images = ((sr_images + 1) / 2).clamp_(0.0, 1.0)
        hr_images = ((hr_images + 1) / 2).clamp_(0.0, 1.0)

        # 使用 torch.nn.functional.mse_loss 计算 MSE
        mse = F.mse_loss(sr_images, hr_images)

        # 使用 torchmetrics 计算 PSNR 和 SSIM
        # torchmetrics 会自动处理批次数据
        psnr = self.psnr(sr_images, hr_images)
        ssim = self.ssim(sr_images, hr_images)

        return {
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'mse': mse.item()
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        stage_config: StageConfig,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        if hasattr(self, 'discriminator'):
            self.discriminator.train()
            
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        # 创建进度条
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1} Training",
            leave=False,
            ncols=100
        )
        
        try:
            for batch_idx, (lr_images, hr_images) in pbar:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                # 训练生成器
                self.optimizer.zero_grad()

                # 前向传播
                if self.config.mixed_precision:
                    with autocast():
                        sr_images = self.model(lr_images)
                        loss, loss_components = self._compute_loss(
                            sr_images, hr_images, stage_config, epoch
                        )

                    # 反向传播
                    self.scaler.scale(loss).backward()

                    # 梯度裁剪
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    sr_images = self.model(lr_images)
                    loss, loss_components = self._compute_loss(
                        sr_images, hr_images, stage_config, epoch
                    )
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.optimizer.step()
                
                # 训练判别器（如果存在且当前阶段需要）
                disc_loss_value = 0.0
                if hasattr(self, '_train_discriminator_step'):
                    disc_metrics = self._train_discriminator_step(hr_images, sr_images.detach())
                    if 'discriminator_loss' in disc_metrics:
                        disc_loss_value = disc_metrics['discriminator_loss']
                        epoch_losses['discriminator'].append(disc_loss_value)
                
                # 记录损失
                epoch_losses['total'].append(loss.item())
                for key, value in loss_components.items():
                    epoch_losses[key].append(value)
                
                # 计算指标
                metrics = self._compute_metrics(sr_images, hr_images)
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                
                # 更新进度条显示
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{metrics.get("psnr", 0):.2f}',
                    'SSIM': f'{metrics.get("ssim", 0):.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # 记录到TensorBoard (移除了每隔一定轮数的控制台输出)
                if batch_idx % self.config.log_frequency == 0:
                    self._log_batch_metrics_tensorboard_only(
                        epoch, batch_idx, len(train_loader),
                        loss.item(), loss_components, metrics,
                        disc_loss_value if disc_loss_value > 0 else None
                    )
                
                self.global_step += 1
                
        except KeyboardInterrupt:
            print(f"\n训练在 epoch {epoch}, batch {batch_idx} 被中断")
            # 重新抛出异常让上层处理
            raise
        
        # 计算epoch平均值
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        stage_config: StageConfig,
        epoch: int
    ) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        # 创建验证进度条
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1} [{stage_config.name}] Validation",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for lr_images, hr_images in pbar:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                # 前向传播
                if self.config.mixed_precision:
                    with autocast():
                        sr_images = self.model(lr_images)
                        loss, loss_components = self._compute_loss(
                            sr_images, hr_images, stage_config, epoch
                        )
                else:
                    sr_images = self.model(lr_images)
                    loss, loss_components = self._compute_loss(
                        sr_images, hr_images, stage_config, epoch
                    )
                
                # 记录损失
                epoch_losses['total'].append(loss.item())
                for key, value in loss_components.items():
                    epoch_losses[key].append(value)
                
                # 计算指标
                metrics = self._compute_metrics(sr_images, hr_images)
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                
                # 更新验证进度条显示
                pbar.set_postfix({
                    'Val_Loss': f'{loss.item():.4f}',
                    'Val_PSNR': f'{metrics.get("psnr", 0):.2f}',
                    'Val_SSIM': f'{metrics.get("ssim", 0):.4f}'
                })
        
        # 计算epoch平均值
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _log_batch_metrics_tensorboard_only(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        loss_components: Dict[str, float],
        metrics: Dict[str, float],
        disc_loss: Optional[float] = None
    ):
        """只记录TensorBoard指标，不输出到控制台"""
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Train/GeneratorLoss', loss, self.global_step)
            if disc_loss is not None:
                self.writer.add_scalar('Train/DiscriminatorLoss', disc_loss, self.global_step)
            for key, value in loss_components.items():
                self.writer.add_scalar(f'Train/Loss_{key}', value, self.global_step)
            for key, value in metrics.items():
                self.writer.add_scalar(f'Train/{key.upper()}', value, self.global_step)
    
    def _log_batch_metrics(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        loss_components: Dict[str, float],
        metrics: Dict[str, float],
        disc_loss: Optional[float] = None
    ):
        """记录批次指标（保留原方法以防其他地方调用）"""
        # 控制台输出已被移除，只保留TensorBoard记录
        self._log_batch_metrics_tensorboard_only(
            epoch, batch_idx, total_batches, loss, loss_components, metrics, disc_loss
        )
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """记录epoch指标"""
        # 控制台输出
        print(f"\nEpoch {epoch} Summary:")
        disc_train_info = f", Disc: {train_metrics.get('discriminator', 0):.4f}" if 'discriminator' in train_metrics else ""
        print(f"Train - Gen Loss: {train_metrics['total']:.4f}{disc_train_info}, "
              f"PSNR: {train_metrics.get('psnr', 0):.2f}, "
              f"SSIM: {train_metrics.get('ssim', 0):.4f}")
        print(f"Val   - Loss: {val_metrics['total']:.4f}, "
              f"PSNR: {val_metrics.get('psnr', 0):.2f}, "
              f"SSIM: {val_metrics.get('ssim', 0):.4f}")
        
        # TensorBoard记录
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Epoch/Train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Epoch/Val_{key}', value, epoch)
            
            # 记录学习率
            if self.optimizer:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Epoch/Learning_Rate', lr, epoch)
            
            # 记录判别器学习率（如果存在）
            if hasattr(self, 'discriminator_optimizer') and self.discriminator_optimizer:
                disc_lr = self.discriminator_optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Epoch/Discriminator_Learning_Rate', disc_lr, epoch)
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """判断是否应该早停"""
        # 使用PSNR作为主要指标
        current_psnr = val_metrics.get('psnr', 0)
        
        if 'psnr' not in self.best_metrics or current_psnr > self.best_metrics['psnr']:
            self.best_metrics.update(val_metrics)
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(
        self,
        epoch: int,
        stage_name: str,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage_name,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metrics': self.best_metrics,
            'config': self.config.__dict__,
            'metrics_history': dict(self.metrics_history)
        }
        
        # 保存判别器状态（如果存在）
        if hasattr(self, 'discriminator') and self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
        
        if hasattr(self, 'discriminator_optimizer') and self.discriminator_optimizer:
            checkpoint['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # 保存检查点
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if is_best:
            checkpoint_path = checkpoint_dir / f'best_{stage_name}.pth'
        else:
            checkpoint_path = checkpoint_dir / f'{stage_name}_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 清理旧的检查点
        self._cleanup_old_checkpoints(stage_name)
    
    def _cleanup_old_checkpoints(self, stage_name: str):
        """清理旧的检查点文件"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        pattern = f'{stage_name}_epoch_*.pth'
        
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        
        # 保留最新的N个检查点
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """加载检查点"""
        # 使用weights_only=False以兼容PyTorch 2.6+的安全限制
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if load_optimizer and checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if load_scheduler and checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载混合精度状态
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 加载判别器状态（如果存在）
        if hasattr(self, 'discriminator') and self.discriminator and checkpoint.get('discriminator_state_dict'):
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if hasattr(self, 'discriminator_optimizer') and self.discriminator_optimizer and checkpoint.get('discriminator_optimizer_state_dict'):
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metrics = checkpoint.get('best_metrics', {})
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
        return checkpoint
    
    def train_stage(
        self,
        stage_name: str,
        stage_config: StageConfig,
        resume_from_checkpoint: Optional[str] = None
    ):
        """训练单个阶段"""
        print(f"\n=== 开始训练阶段: {stage_config.name} ===")
        
        self.current_stage = stage_name
        
        # 设置TensorBoard
        self._setup_tensorboard(stage_name)
        
        # 设置优化器
        self._setup_optimizer(stage_config)
        
        # 恢复检查点
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            checkpoint = self.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint.get('epoch', 0) + 1
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            train_data_path=self.config.train_data_path,
            val_data_path=self.config.val_data_path,
            batch_size=stage_config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            stage=stage_name,
            train_csv='dataset_index.csv',
            val_csv='dataset_index.csv'
        )
        
        # 训练循环
        epoch_pbar = tqdm(
            range(start_epoch, stage_config.epochs),
            desc=f"Stage {stage_name} Progress",
            position=0,
            leave=True
        )
        
        try:
            for epoch in epoch_pbar:
                self.current_epoch = epoch
                
                # 训练
                train_metrics = self._train_epoch(train_loader, stage_config, epoch)
                
                # 验证
                if epoch % self.config.val_frequency == 0:
                    val_metrics = self._validate_epoch(val_loader, stage_config, epoch)
                    
                    # 记录指标
                    self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                    
                    # 优化的保存策略：每个epoch中间保存一次
                    # 只在epoch的一半时保存常规检查点，最后保存最佳检查点
                    should_save_checkpoint = False
                    is_best = False
                    
                    # 检查是否是最佳模型
                    current_psnr = val_metrics.get('psnr', 0)
                    if current_psnr > self.best_metrics.get('psnr', 0):
                        self.best_metrics['psnr'] = current_psnr
                        is_best = True
                        should_save_checkpoint = True
                    
                    # 每个epoch的中间点保存一次（不是最佳模型时）
                    if epoch > 0 and epoch % max(1, stage_config.epochs // 2) == 0 and not is_best:
                        should_save_checkpoint = True
                    
                    # 每10个epoch强制保存一次（防止丢失）
                    if epoch % 10 == 0:
                        should_save_checkpoint = True
                    
                    if should_save_checkpoint:
                        self.save_checkpoint(epoch, stage_name, is_best)
                    
                    # 更新epoch进度条显示
                    epoch_pbar.set_postfix({
                        'Train_Loss': f'{train_metrics.get("total", 0):.4f}',
                        'Val_PSNR': f'{val_metrics.get("psnr", 0):.2f}',
                        'Best_PSNR': f'{self.best_metrics.get("psnr", 0):.2f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # 早停检查
                    if self._should_early_stop(val_metrics):
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                # 更新学习率
                if self.scheduler:
                    self.scheduler.step()
            
            # 保存最终检查点
            self.save_checkpoint(self.current_epoch, stage_name, is_best=True)
            
        except KeyboardInterrupt:
            print(f"\n训练被用户中断 (Ctrl+C)")
            print(f"正在安全保存当前进度...")
            
            # 安全保存当前状态
            try:
                # 确保CUDA同步
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 保存中断时的检查点
                self.save_checkpoint(
                    self.current_epoch, 
                    stage_name, 
                    is_best=False,
                    additional_info={'interrupted': True, 'interrupt_time': time.time()}
                )
                print(f"检查点已安全保存到 epoch {self.current_epoch}")
                
            except Exception as save_error:
                print(f"保存检查点时出错: {save_error}")
                print("尝试紧急保存...")
                
                # 紧急保存策略：只保存模型权重
                try:
                    emergency_path = self.checkpoint_dir / f"emergency_save_{stage_name}_epoch_{self.current_epoch}.pth"
                    torch.save({
                        'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                        'epoch': self.current_epoch,
                        'stage': stage_name,
                        'emergency_save': True
                    }, emergency_path)
                    print(f"紧急保存完成: {emergency_path}")
                except Exception as emergency_error:
                    print(f"紧急保存也失败了: {emergency_error}")
            
            # 重新抛出异常以正确退出
            raise
        
        if self.writer:
            self.writer.close()
        
        print(f"=== 阶段 {stage_config.name} 训练完成 ===")