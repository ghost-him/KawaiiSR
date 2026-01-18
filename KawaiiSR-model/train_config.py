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
    num_workers: int
    pin_memory: bool
    # use_online_data 已移除
    online_data_options: Dict[str, Any]
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    enable_cuda_prefetch: bool
    allow_tf32: bool

    # 损失全局缩放
    loss_global_scale: float

    # 训练配置
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    mixed_precision: bool
    gradient_clip_norm: float
    torch_compile: bool
    # 保存策略（多指标开关，true/false）。若全部为 False，则自动回退到 psnr。
    save_on_psnr: bool
    save_on_ssim: bool
    save_on_lpips: bool
    save_on_val_loss: bool

    # 检查点/日志配置
    checkpoint_dir: str
    log_every: int
    auto_resume: bool  # 如果为 True，训练前自动尝试从 checkpoint_dir 加载最后的状态

    # 验证/早停
    val_every: int
    early_stopping_patience: int

    # 损失权重（固定，不做动态/阶段切换）
    loss_weights: Dict[str, float]

    # 额外损失/开关项
    loss_options: Dict[str, Any]

    # GAN 配置（可选，默认关闭）
    gan: Dict[str, Any]

def get_default_model_config() -> Dict[str, Any]:
    """获取默认的模型配置"""
    return {
        'image_size': 64,
        'in_channels': 3,
        'image_range': 1.0,
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

DEFAULT_LOSS_WEIGHTS = {
    'pixel': 1.0,
    'perceptual': 0.1,
    'frequency': 0.0,
    'vgg': 0.0,
    'adversarial': 0.0,
}

DEFAULT_LOSS_OPTIONS = {
    'enable_anime_loss': False,
}

DEFAULT_GAN_CFG = {
    'enabled': False,
    'discriminator_lr': 1e-4,
    'disc_update_ratio': 1,
}

SAVE_METRIC_KEYS = ('psnr', 'ssim', 'lpips', 'val_loss')



def _validate_dict_keys(name: str, value: Dict[str, Any], required_keys) -> None:
    missing = [k for k in required_keys if k not in value]
    if missing:
        raise ValueError(f'配置项 "{name}" 缺少必要键: {", ".join(missing)}。')


def create_training_config(
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    yaml_path: Optional[str] = None
) -> TrainingConfig:
    """创建扁平训练配置；严格遵循 YAML/参数，缺失或冲突直接报错。"""

    config_payload: Dict[str, Any] = {}
    yaml_mode = yaml_path is not None

    if yaml_mode:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f) or {}
        if not isinstance(raw_data, dict):
            raise ValueError('YAML 顶层结构必须是字典。')

        train_cfg = raw_data.get('train')
        if not isinstance(train_cfg, dict):
            raise ValueError('YAML 中必须提供 train 字段，且类型为字典。')

        # 检查 train 字段是否包含未定义的键
        allowed_field_names = set(TrainingConfig.__dataclass_fields__.keys()) - {
            'model_config',
            'loss_weights',
            'loss_options',
            'gan',
            'online_data_options',
        }
        unknown_train_keys = set(train_cfg.keys()) - allowed_field_names
        if unknown_train_keys:
            raise ValueError(f'train 段包含未知键: {", ".join(sorted(unknown_train_keys))}')

        config_payload.update(train_cfg)

        # 全局损失缩放（必须显式提供）
        if 'loss_global_scale' not in raw_data:
            raise ValueError('YAML 中必须提供 loss_global_scale 字段。')
        try:
            loss_global_scale = float(raw_data['loss_global_scale'])
        except (TypeError, ValueError):
            raise ValueError('loss_global_scale 必须是可转换为浮点数的数值。')
        if loss_global_scale <= 0:
            raise ValueError('loss_global_scale 必须大于 0。')
        config_payload['loss_global_scale'] = loss_global_scale


        # 模型配置
        model_cfg = raw_data['model_config']
        required_model_keys = set(get_default_model_config().keys())
        _validate_dict_keys('model_config', model_cfg, required_model_keys)
        norm_layer_value = model_cfg.get('hat_norm_layer')
        if isinstance(norm_layer_value, str):
            norm_key = norm_layer_value.strip().lower()
            norm_mapping = {
                'layernorm': nn.LayerNorm,
                'nn.layernorm': nn.LayerNorm,
                'torch.nn.layernorm': nn.LayerNorm,
            }
            if norm_key not in norm_mapping:
                raise ValueError('model_config.hat_norm_layer 字符串值无法识别，仅支持 LayerNorm。')
            model_cfg['hat_norm_layer'] = norm_mapping[norm_key]
        elif norm_layer_value is not None and not callable(norm_layer_value):
            raise ValueError('model_config.hat_norm_layer 必须是可调用对象或受支持的字符串名称。')
        # 清理已弃用的 tiling 配置（调用方自行负责分块）
        for deprecated_key in ('use_tiling', 'tile_size', 'tile_pad'):
            model_cfg.pop(deprecated_key, None)
        config_payload['model_config'] = model_cfg

        # 损失、GAN、在线数据等配置
        loss_weights_cfg = raw_data['loss_weights']
        _validate_dict_keys('loss_weights', loss_weights_cfg, DEFAULT_LOSS_WEIGHTS.keys())
        config_payload['loss_weights'] = loss_weights_cfg

        loss_options_cfg = raw_data['loss_options']
        _validate_dict_keys('loss_options', loss_options_cfg, DEFAULT_LOSS_OPTIONS.keys())
        config_payload['loss_options'] = loss_options_cfg

        gan_cfg = raw_data['gan']
        _validate_dict_keys('gan', gan_cfg, DEFAULT_GAN_CFG.keys())
        config_payload['gan'] = gan_cfg

        online_data_cfg = raw_data['online_data_options']
        config_payload['online_data_options'] = online_data_cfg

        metrics_cfg = raw_data.get('save_metrics')
        if metrics_cfg is None:
            raise ValueError('YAML 中必须提供 save_metrics 字段。')
        if not isinstance(metrics_cfg, dict):
            raise ValueError('save_metrics 必须是字典。')
        _validate_dict_keys('save_metrics', metrics_cfg, SAVE_METRIC_KEYS)
        for metric_key in SAVE_METRIC_KEYS:
            metric_value = metrics_cfg[metric_key]
            if not isinstance(metric_value, bool):
                raise ValueError(f'save_metrics.{metric_key} 必须是布尔值。')
            config_payload[f'save_on_{metric_key}'] = metric_value

    else:
        # 无 YAML：使用默认值，但要求必需路径参数显式提供
        if not train_data_path:
            raise ValueError('未提供 YAML 时必须显式传入 train_data_path。')
        if not val_data_path:
            raise ValueError('未提供 YAML 时必须显式传入 val_data_path。')
        config_payload.update({
            'model_config': get_default_model_config(),
            'train_data_path': train_data_path,
            'val_data_path': val_data_path,
            'checkpoint_dir': checkpoint_dir or './checkpoints',
            'loss_weights': dict(DEFAULT_LOSS_WEIGHTS),
            'loss_options': dict(DEFAULT_LOSS_OPTIONS),
            'gan': dict(DEFAULT_GAN_CFG),
            'online_data_options': {},
            'loss_global_scale': 1.0,
            
            # 必需字段的默认回退值（用于无 YAML 模式的快速测试）
            'num_workers': 4,
            'pin_memory': True,
            'dataloader_prefetch_factor': 2,
            'dataloader_persistent_workers': False,
            'enable_cuda_prefetch': False,
            'allow_tf32': True,
            'device': 'cuda',
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'mixed_precision': False,
            'gradient_clip_norm': 1.0,
            'torch_compile': False,
            'save_on_psnr': True,
            'save_on_ssim': False,
            'save_on_lpips': False,
            'save_on_val_loss': False,
            'log_every': 100,
            'auto_resume': False,
            'val_every': 1,
            'early_stopping_patience': 10,
        })

    # 覆盖优先使用函数参数
    if train_data_path is not None:
        config_payload['train_data_path'] = train_data_path
    if val_data_path is not None:
        config_payload['val_data_path'] = val_data_path
    if checkpoint_dir is not None:
        config_payload['checkpoint_dir'] = checkpoint_dir

    # 检查 dataclass 字段的完整性
    required_fields = TrainingConfig.__dataclass_fields__
    if yaml_mode:
        missing_fields = [name for name in required_fields.keys() if name not in config_payload]
        if missing_fields:
            raise ValueError(f'配置缺少以下必须字段: {", ".join(missing_fields)}')

    # 校验全局损失缩放
    loss_global_scale_val = config_payload.get('loss_global_scale', 1.0)
    try:
        loss_global_scale_val = float(loss_global_scale_val)
    except (TypeError, ValueError):
        raise ValueError('loss_global_scale 必须是可转换为浮点数的数值。')
    if loss_global_scale_val <= 0:
        raise ValueError('loss_global_scale 必须大于 0。')
    config_payload['loss_global_scale'] = loss_global_scale_val

    # 校验 torch_compile 类型
    if 'torch_compile' in config_payload and not isinstance(config_payload['torch_compile'], bool):
        raise ValueError('torch_compile 配置仅支持布尔值 true/false。')

    # 校验保存指标至少有一个为 True
    if yaml_mode:
        if not any(config_payload[f'save_on_{metric}'] for metric in SAVE_METRIC_KEYS):
            raise ValueError('save_metrics 至少需要启用一个保存指标。')

    # 校验在线数据预加载策略
    online_options = config_payload.get('online_data_options', {})
    if not isinstance(online_options, dict):
        raise ValueError('online_data_options 必须是字典。')

    # 核心路径字段检查
    for path_key in ('train_data_path', 'val_data_path', 'checkpoint_dir'):
        value = config_payload.get(path_key)
        if not value:
            raise ValueError(f'{path_key} 未在参数或配置文件中提供。')

    return TrainingConfig(**config_payload)

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