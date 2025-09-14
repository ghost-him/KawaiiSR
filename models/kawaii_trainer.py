import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from .trainer_base import BaseTrainer
from .train_config import TrainingConfig, StageConfig
from .loss.KawaiiLoss import KawaiiLoss

from .loss.DiscriminatorLoss import DiscriminatorLoss
from .Discriminator.UNetDiscriminatorSN import UNetDiscriminatorSN

class KawaiiSRTrainer(BaseTrainer):
    """KawaiiSR三阶段训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        super().__init__(model, config, device)
        
        # 初始化判别器
        self.discriminator = UNetDiscriminatorSN(
            num_in_ch=3,
            num_feat=64,
            skip_connection=True
        ).to(device)
        
        # 初始化损失函数
        loss_weights = getattr(config, 'loss_weights', {})
        self.kawaii_loss = KawaiiLoss(
            lambda_char=loss_weights.get('pixel', 1.0),
            lambda_lap=loss_weights.get('frequency', 0.1), 
            lambda_perc=loss_weights.get('perceptual', 0.1),

            lambda_vgg=loss_weights.get('vgg', 0.1),
            device=device
        )
        
        self.discriminator_loss = DiscriminatorLoss(
            gan_weight=getattr(config, 'gan_weight', 0.1),
            device=device
        )
        

        
        # 损失权重历史（用于动态调整）
        self.loss_weight_history = {
            'pixel': [],
            'perceptual': [],
            'adversarial': [],
            'frequency': [],

        }
        
        # 判别器优化器
        self.discriminator_optimizer = None
        self._training_discriminator = False
        

    
    def _compute_dynamic_weights(
        self,
        stage_config: StageConfig,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """计算动态损失权重"""
        progress = epoch / total_epochs
        
        if stage_config.name == "stage1":
            # 第一阶段：主要关注像素损失和感知损失
            weights = {
                'pixel': stage_config.loss_weights['pixel'],
                'perceptual': stage_config.loss_weights['perceptual'] * (1 + 0.5 * progress),
                'adversarial': 0.0,  # 第一阶段不使用对抗损失
                'frequency': stage_config.loss_weights['frequency'] * (0.5 + 0.5 * progress),

            }
            
        elif stage_config.name == "stage2":
            # 第二阶段：逐渐引入对抗损失
            warmup_progress = min(progress * 2, 1.0)  # 前50%用于warmup
            
            weights = {
                'pixel': stage_config.loss_weights['pixel'] * (1 - 0.3 * progress),
                'perceptual': stage_config.loss_weights['perceptual'],
                'adversarial': stage_config.loss_weights['adversarial'] * warmup_progress,
                'frequency': stage_config.loss_weights['frequency'],

            }
            
        else:  # stage3
            # 第三阶段：精细化调优
            weights = {
                'pixel': stage_config.loss_weights['pixel'] * 0.7,  # 降低像素损失权重
                'perceptual': stage_config.loss_weights['perceptual'] * 0.8,
                'adversarial': stage_config.loss_weights['adversarial'] * 0.9,
                'frequency': stage_config.loss_weights['frequency'],

            }
        
        # 记录权重历史
        for key, value in weights.items():
            self.loss_weight_history[key].append(value)
        
        return weights
    
    def _compute_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        stage_config: StageConfig,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数"""
        # 获取动态权重
        weights = self._compute_dynamic_weights(stage_config, epoch, stage_config.epochs)
        
        loss_components = {}
        total_loss = 0.0
        
        # 1. KawaiiLoss (包含像素、感知、频率损失)
        kawaii_loss_dict = self.kawaii_loss(sr_images, hr_images)
        kawaii_total = kawaii_loss_dict['total_loss']
        
        # 分别记录各个子损失
        if weights['pixel'] > 0:
            loss_components['pixel'] = kawaii_loss_dict['pixel_loss'].item()
            
        if weights['perceptual'] > 0:
            loss_components['perceptual'] = kawaii_loss_dict['perceptual_loss'].item()
            
        if weights['frequency'] > 0:
            loss_components['frequency'] = kawaii_loss_dict['frequency_loss'].item()
        
        # 根据阶段调整KawaiiLoss权重
        kawaii_weight = weights['pixel'] + weights['perceptual'] + weights['frequency']
        
        total_loss += kawaii_weight * kawaii_total
        
        # 2. 对抗损失 (从第二阶段开始)
        if weights['adversarial'] > 0 and stage_config.name != "stage1":
            # 生成器损失
            fake_pred = self.discriminator(sr_images)
            gen_loss = self.discriminator_loss.generator_loss(fake_pred)
            loss_components['adversarial'] = gen_loss.item()
            total_loss += weights['adversarial'] * gen_loss
        
        # 记录权重信息
        loss_components.update({f'weight_{k}': v for k, v in weights.items()})
        
        return total_loss, loss_components
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                    stage: str = None) -> Dict[str, torch.Tensor]:
        """计算三阶段训练的损失函数"""
        if stage is None:
            stage = self.current_stage
            
        losses = {}
        total_loss = 0
        
        # 获取当前阶段的权重
        weights = self.loss_weights
        
        # 1. KawaiiLoss (包含像素、感知、频域损失)
        if weights['kawaii'] > 0:
            kawaii_loss_dict = self.kawaii_loss(outputs, targets)
            kawaii_total = kawaii_loss_dict['total_loss']
            losses.update(kawaii_loss_dict)  # 包含所有子损失
            total_loss += weights['kawaii'] * kawaii_total
        
        # 2. 对抗损失 (从stage2开始)
        if weights['adversarial'] > 0 and stage in ['stage2', 'stage3']:
            # 生成器损失
            fake_pred = self.discriminator(outputs)
            gen_loss = self.discriminator_loss.generator_loss(fake_pred)
            losses['generator'] = gen_loss
            total_loss += weights['adversarial'] * gen_loss
        
        losses['total'] = total_loss
        return losses
    
    def setup_discriminator_optimizer(self):
        """设置判别器优化器"""
        if self.current_stage in ['stage2', 'stage3']:
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=getattr(self.config, 'discriminator_lr', 0.0001),
                betas=(0.5, 0.999)
            )
    
    def train_discriminator_step(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """训练判别器的一步"""
        if self.discriminator_optimizer is None or self.current_stage == 'stage1':
            return {}
        
        self._training_discriminator = True
        self.discriminator_optimizer.zero_grad()
        
        # 计算判别器损失
        real_pred = self.discriminator(real_images)
        fake_pred = self.discriminator(fake_images.detach())
        disc_loss = self.discriminator_loss.discriminator_loss(real_pred, fake_pred)
        
        # 反向传播
        disc_loss.backward()
        self.discriminator_optimizer.step()
        
        self._training_discriminator = False
        
        return {'discriminator_loss': disc_loss.item()}
    

    
    def _adjust_model_for_stage(self, stage: str):
        """根据训练阶段调整模型参数"""
        if stage == "stage1":
            # 第一阶段：全模型训练
            self._unfreeze_all_parameters()
            print("Stage 1: Full model training")
            
        elif stage == "stage2":
            # 第二阶段：启用分块训练和判别器
            self._unfreeze_all_parameters()
            self.setup_discriminator_optimizer()
            print("Stage 2: Enabled patch-based training and discriminator")
            
        elif stage == "stage3":
            # 第三阶段：冻结HAT模型，只训练残差块
            self._freeze_hat_model()

            self.setup_discriminator_optimizer()
            print("Stage 3: Frozen HAT model, training residual blocks only")
    
    def train_stage(
        self,
        stage_name: str,
        stage_config: StageConfig,
        resume_from_checkpoint: Optional[str] = None
    ):
        """训练单个阶段（重写父类方法以添加阶段特定设置）"""
        print(f"\n=== 配置阶段: {stage_config.name} ===")
        
        # 应用阶段特定设置
        self._adjust_model_for_stage(stage_config.name)
        
        # 调用父类的训练方法
        super().train_stage(stage_name, stage_config, resume_from_checkpoint)
    
    def train_all_stages(
        self,
        resume_from_stage: Optional[str] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """训练所有三个阶段"""
        stages = [
            ("stage1", self.config.stage1),
            ("stage2", self.config.stage2),
            ("stage3", self.config.stage3)
        ]
        
        start_stage_idx = 0
        if resume_from_stage:
            stage_names = [s[0] for s in stages]
            if resume_from_stage in stage_names:
                start_stage_idx = stage_names.index(resume_from_stage)
                print(f"Resuming from stage: {resume_from_stage}")
        
        for i, (stage_name, stage_config) in enumerate(stages[start_stage_idx:], start_stage_idx):
            # 确定是否需要从检查点恢复
            checkpoint_path = None
            if i == start_stage_idx and resume_checkpoint:
                checkpoint_path = resume_checkpoint
            elif i > start_stage_idx:
                # 从上一阶段的最佳模型开始
                prev_stage_name = stages[i-1][0]
                checkpoint_path = str(Path(self.config.checkpoint_dir) / f'best_{prev_stage_name}.pth')
                if not Path(checkpoint_path).exists():
                    checkpoint_path = None
            
            # 训练当前阶段
            self.train_stage(stage_name, stage_config, checkpoint_path)
            
            print(f"\n=== 阶段 {stage_config.name} 完成 ===")
            print(f"最佳指标: {self.best_metrics}")
            
            # 重置早停计数器
            self.early_stopping_counter = 0
    
    def save_loss_weight_history(self, save_path: str):
        """保存损失权重历史"""
        import json
        
        history_data = {
            'loss_weights': self.loss_weight_history,
            'best_metrics': self.best_metrics,
            'config': {
                'stage1': self.config.stage1.__dict__,
                'stage2': self.config.stage2.__dict__,
                'stage3': self.config.stage3.__dict__
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Loss weight history saved to: {save_path}")
    
    def visualize_training_progress(self):
        """可视化训练进度（可选功能）"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('KawaiiSR Training Progress', fontsize=16)
            
            # 绘制损失权重变化
            for i, (loss_type, weights) in enumerate(self.loss_weight_history.items()):
                if weights:  # 只绘制有数据的损失类型
                    row, col = i // 3, i % 3
                    if row < 2 and col < 3:
                        axes[row, col].plot(weights)
                        axes[row, col].set_title(f'{loss_type.capitalize()} Loss Weight')
                        axes[row, col].set_xlabel('Training Steps')
                        axes[row, col].set_ylabel('Weight')
                        axes[row, col].grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = Path(self.config.checkpoint_dir) / 'training_progress.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Training progress visualization saved to: {save_path}")
            
        except ImportError:
            print("Matplotlib not available, skipping visualization")
    
    def get_training_summary(self) -> Dict[str, any]:
        """获取训练总结"""
        return {
            'best_metrics': self.best_metrics,
            'total_steps': self.global_step,
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage,
            'loss_weight_history': self.loss_weight_history,
            'config_summary': {
                'stage1_epochs': self.config.stage1.epochs,
                'stage2_epochs': self.config.stage2.epochs,
                'stage3_epochs': self.config.stage3.epochs,
                'total_epochs': (
                    self.config.stage1.epochs + 
                    self.config.stage2.epochs + 
                    self.config.stage3.epochs
                )
            }
        }

class ProgressiveTrainingScheduler:
    """渐进式训练调度器"""
    
    def __init__(self, trainer: KawaiiSRTrainer):
        self.trainer = trainer
        self.stage_transitions = []
    
    def should_transition_stage(
        self,
        current_stage: str,
        current_metrics: Dict[str, float],
        epoch: int
    ) -> bool:
        """判断是否应该转换到下一阶段"""
        # 基于PSNR阈值的阶段转换逻辑
        psnr = current_metrics.get('psnr', 0)
        
        if current_stage == "stage1" and psnr > 28.0:  # PSNR > 28dB
            return True
        elif current_stage == "stage2" and psnr > 30.0:  # PSNR > 30dB
            return True
        
        return False
    
    def log_stage_transition(self, from_stage: str, to_stage: str, metrics: Dict[str, float]):
        """记录阶段转换"""
        transition_info = {
            'from_stage': from_stage,
            'to_stage': to_stage,
            'metrics': metrics,
            'global_step': self.trainer.global_step
        }
        self.stage_transitions.append(transition_info)
        print(f"Stage transition: {from_stage} -> {to_stage} at step {self.trainer.global_step}")
        print(f"Transition metrics: {metrics}")