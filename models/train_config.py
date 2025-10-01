import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


@dataclass
class TrainingConfig:
    """扁平化的单阶段训练配置（已移除三阶段相关配置）"""

    # 模型配置
    model_config: Dict[str, Any]

    # 数据配置
    train_data_path: str
    val_data_path: str
    num_workers: int = 4
    pin_memory: bool = True

    # 训练配置
    device: str = 'cuda'
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    mixed_precision: bool = False
    gradient_clip_norm: float = 1.0

    # 检查点/日志配置
    checkpoint_dir: str = './checkpoints'
    keep_last_n_checkpoints: int = 3
    tensorboard_dir: str = './logs'
    log_every: int = 100

    # 验证/早停
    val_every: int = 1
    early_stopping_patience: int = 10

    # 损失权重（固定，不做动态/阶段切换）
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'pixel': 1.0,
        'perceptual': 0.1,
        'frequency': 0.0,
        'vgg': 0.0,
        'adversarial': 0.0,
    })

    # 额外损失/开关项
    loss_options: Dict[str, Any] = field(default_factory=lambda: {
        'enable_anime_loss': False
    })

    # GAN 配置（可选，默认关闭）
    gan: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'discriminator_lr': 1e-4,
        'disc_update_ratio': 1,
    })

def get_default_model_config() -> Dict[str, Any]:
    """获取默认的模型配置"""
    return {
        'image_size': 64,
        'in_channels': 3,
        'image_range': 1.0,
        'use_tiling': False,
        'tile_size': 64,
        'tile_pad': 32,
        'hat_patch_size': 1,
        'hat_body_hid_channels': 180,
        'hat_upsampler_hid_channels': 64,
        'hat_depths': (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        'hat_num_heads': (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
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
    yaml_path: Optional[str] = None
) -> TrainingConfig:
    """创建扁平训练配置，支持从 YAML 加载并与默认值合并"""
    base: Dict[str, Any] = {
        'model_config': get_default_model_config(),
        'train_data_path': train_data_path,
        'val_data_path': val_data_path,
        'checkpoint_dir': checkpoint_dir,
    }

    # 从 YAML 读取并合并
    if yaml_path:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        train_cfg = data.get('train', {})
        model_overrides = data.get('model_config', {}) or data.get('model', {}) or {}
        loss_weights = data.get('loss_weights', {})
        loss_options = data.get('loss_options', {})
        gan_cfg = data.get('gan', {})

        # 合并到 base
        base.update(train_cfg)
    # 覆盖模型配置
    base_model_cfg = get_default_model_config()
    base_model_cfg.update(model_overrides)
    base['model_config'] = base_model_cfg
    # 调用 default_factory 需要实例化一个临时 TrainingConfig 或直接重新创建默认字典
    default_loss_weights = {
        'pixel': 1.0,
        'perceptual': 0.1,
        'frequency': 0.0,
        'vgg': 0.0,
        'adversarial': 0.0,
    }
    default_loss_options = {'enable_anime_loss': False}
    default_gan = {
        'enabled': False,
        'discriminator_lr': 1e-4,
        'disc_update_ratio': 1,
    }
    base['loss_weights'] = {**default_loss_weights, **(loss_weights if yaml_path else {})}
    base['loss_options'] = {**default_loss_options, **(loss_options if yaml_path else {})}
    base['gan'] = {**default_gan, **(gan_cfg if yaml_path else {})}

    return TrainingConfig(**base)

# 示例配置输出
if __name__ == '__main__':
    cfg = create_training_config(
        train_data_path='./data/train',
        val_data_path='./data/val',
        checkpoint_dir='./checkpoints',
        yaml_path=None
    )
    print('训练配置(扁平):')
    print(f"device={cfg.device}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, lr={cfg.learning_rate}")
    print(f"loss_weights={cfg.loss_weights}")
    print(f"gan={cfg.gan}")