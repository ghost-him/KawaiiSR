import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class StageConfig:
    """单个训练阶段的配置"""
    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    use_tiling: bool
    tile_size: int
    tile_pad: int
    input_size: int
    loss_weights: Dict[str, float]
    freeze_hat: bool = False
    warmup_epochs: int = 5

@dataclass
class TrainingConfig:
    """完整的训练配置"""
    # 模型配置
    model_config: Dict[str, Any]

    # 数据配置
    train_data_path: str
    val_data_path: str
    num_workers: int = 4
    pin_memory: bool = True

    # 训练配置
    device: str = 'cuda'
    mixed_precision: bool = False
    gradient_clip_norm: float = 1.0

    # 检查点配置
    checkpoint_dir: str = './checkpoints'
    save_frequency: int = 5  # 每5个epoch保存一次
    keep_last_n_checkpoints: int = 3

    # 验证配置
    val_frequency: int = 1
    early_stopping_patience: int = 10

    # 日志配置
    log_frequency: int = 100  # 每100个batch记录一次
    tensorboard_dir: str = './logs'

    # 三阶段配置
    stages: Dict[str, StageConfig] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = self._get_default_stages()
    
    def _get_default_stages(self) -> Dict[str, StageConfig]:
        """获取默认的三阶段训练配置"""
        return {
            'stage1': StageConfig(
                name='基础能力训练',
                epochs=100,
                learning_rate=1e-4,
                batch_size=16,
                use_tiling=False,
                tile_size=64,
                tile_pad=32,
                input_size=128,
                loss_weights={
                    'pixel': 1.0,
                    'frequency': 0.0,
                    'perceptual': 0.1,
                    'vgg': 0.0,
                    'adversarial': 0.0
                },
                freeze_hat=False,
                warmup_epochs=5
            ),
            'stage2': StageConfig(
                name='分块适应性训练',
                epochs=50,
                learning_rate=2e-5,
                batch_size=16,
                use_tiling=True,
                tile_size=64,
                tile_pad=32,
                input_size=128,
                loss_weights={
                    'pixel': 1.0,
                    'frequency': 0.2,
                    'perceptual': 0.3,
                    'vgg': 0.2,
                    'adversarial': 0.1
                },
                freeze_hat=False,
                warmup_epochs=3
            ),
            'stage3': StageConfig(
                name='精细化调优',
                epochs=30,
                learning_rate=1e-5,
                batch_size=8,
                use_tiling=True,
                tile_size=64,
                tile_pad=32,
                input_size=256,
                loss_weights={
                    'pixel': 0.7,
                    'frequency': 0.4,
                    'perceptual': 0.8,
                    'vgg': 0.5,
                    'adversarial': 0.1
                },
                freeze_hat=True,
                warmup_epochs=2
            )
        }

def get_default_model_config() -> Dict[str, Any]:
    """获取默认的模型配置"""
    return {
        'image_size': 128,
        'in_channels': 3,
        'image_range': 1.0,
        'use_tiling': False,
        'tile_size': 64,
        'tile_pad': 32,
        'hat_patch_size': 1,
        'hat_body_hid_channels': 180,
        'hat_upsampler_hid_channels': 64,
        'hat_depths': (6, 6, 6, 6, 6, 6, 6, 6),
        'hat_num_heads': (6, 6, 6, 6, 6, 6, 6, 6),
        'hat_window_size': 16,
        'hat_compress_ratio': 3,
        'hat_squeeze_factor': 30,
        'hat_conv_scale': 0.01,
        'hat_overlap_ratio': 0.5,
        'hat_mlp_ratio': 2.0,
        'hat_qkv_bias': True,
        'hat_drop_rate': 0.0,
        'hat_attn_drop_rate': 0.0,
        'hat_drop_path_rate': 0.1,
        'hat_norm_layer': nn.LayerNorm,
        'hat_patch_norm': True,
        'hat_use_checkpoint': False,
        'hat_resi_connection': '1conv'
    }

def create_training_config(
    train_data_path: str,
    val_data_path: str,
    checkpoint_dir: str = './checkpoints',
    custom_stages: Optional[Dict[str, StageConfig]] = None
) -> TrainingConfig:
    """创建训练配置"""
    config = TrainingConfig(
        model_config=get_default_model_config(),
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        checkpoint_dir=checkpoint_dir
    )
    
    if custom_stages is not None:
        config.stages = custom_stages
    
    return config

# 示例配置
if __name__ == '__main__':
    config = create_training_config(
        train_data_path='./data/train',
        val_data_path='./data/val'
    )
    
    print("训练配置:")
    for stage_name, stage_config in config.stages.items():
        print(f"\n{stage_name}: {stage_config.name}")
        print(f"  Epochs: {stage_config.epochs}")
        print(f"  Learning Rate: {stage_config.learning_rate}")
        print(f"  Use Tiling: {stage_config.use_tiling}")
        print(f"  Freeze HAT: {stage_config.freeze_hat}")
        print(f"  Loss Weights: {stage_config.loss_weights}")