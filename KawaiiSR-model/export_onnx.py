import argparse
import os
import sys
import yaml
import torch
import torch.export
import torch.onnx
import torch.nn as nn

# Add the directory containing the model definitions to path
sys.path.append(os.path.join(os.getcwd(), 'models'))

from KawaiiSR.KawaiiSR import KawaiiSR


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config['model_config']
    raise FileNotFoundError(f"Config file not found: {config_path}")



def resolve_norm_layer(value):
    if isinstance(value, str):
        key = value.strip().lower()
        mapping = {
            'layernorm': nn.LayerNorm,
            'nn.layernorm': nn.LayerNorm,
            'torch.nn.layernorm': nn.LayerNorm,
        }
        if key not in mapping:
            raise ValueError('Unsupported hat_norm_layer value; only LayerNorm is allowed for export.')
        return mapping[key]
    if value is None:
        return nn.LayerNorm
    if callable(value):
        return value
    raise ValueError('hat_norm_layer must be callable or a supported string name.')


def extract_state_dict(ckpt):
    """Extract the actual model weights from various checkpoint formats."""
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint is not a dictionary, cannot extract state dict.')

    for key in ('model_state_dict', 'state_dict', 'model'):
        if key in ckpt:
            ckpt = ckpt[key]
            break

    if not isinstance(ckpt, dict):
        raise ValueError('Resolved checkpoint payload is not a state dict dictionary.')

    state_dict = {}
    for k, v in ckpt.items():
        # Only keep tensor-like entries; drop optimizer or metadata fields.
        if not torch.is_tensor(v):
            continue
        clean_key = k[7:] if k.startswith('module.') else k
        state_dict[clean_key] = v

    if not state_dict:
        raise ValueError('No tensor weights found in checkpoint; ensure the file is a model checkpoint.')

    return state_dict


def build_model(model_config, input_size, device):
    norm_layer = resolve_norm_layer(model_config.get('hat_norm_layer'))
    return KawaiiSR(
        image_size=input_size,
        in_channels=model_config.get('in_channels', 3),
        image_range=model_config.get('image_range', 1.0),
        hat_patch_size=model_config.get('hat_patch_size', 1),
        hat_body_hid_channels=model_config.get('hat_body_hid_channels', 96),
        hat_upsampler_hid_channels=model_config.get('hat_upsampler_hid_channels', 64),
        hat_depths=tuple(model_config.get('hat_depths', (6, 6, 6, 6))),
        hat_num_heads=tuple(model_config.get('hat_num_heads', (6, 6, 6, 6))),
        hat_window_size=model_config.get('hat_window_size', 7),
        hat_compress_ratio=model_config.get('hat_compress_ratio', 3),
        hat_squeeze_factor=model_config.get('hat_squeeze_factor', 30),
        hat_conv_scale=model_config.get('hat_conv_scale', 0.01),
        hat_overlap_ratio=model_config.get('hat_overlap_ratio', 0.5),
        hat_mlp_ratio=model_config.get('hat_mlp_ratio', 4.0),
        hat_qkv_bias=model_config.get('hat_qkv_bias', True),
        hat_drop_rate=model_config.get('hat_drop_rate', 0.0),
        hat_attn_drop_rate=model_config.get('hat_attn_drop_rate', 0.0),
        hat_drop_path_rate=model_config.get('hat_drop_path_rate', 0.1),
        hat_norm_layer=norm_layer,
        hat_patch_norm=model_config.get('hat_patch_norm', True),
        hat_use_checkpoint=model_config.get('hat_use_checkpoint', False),
        hat_resi_connection=model_config.get('hat_resi_connection', '1conv'),
        tail_num_layers=model_config.get('tail_num_layers', 4),
    ).to(device)


def export_onnx(config_path, checkpoint_path, input_size, output_path):
    device = torch.device( 'cpu')
    print(f"Using device: {device}")

    model_config = load_config(config_path)
    # 强制使用给定 input_size，导出固定尺寸 ONNX；调用方应确保与 window_size 对齐
    export_size = input_size

    print("Initializing KawaiiSR (HAT) model for export...")
    model = build_model(model_config, export_size, device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = extract_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing keys {missing}, unexpected keys {unexpected}")
        print("Checkpoint loaded successfully!")
    else:
        print("Warning: No checkpoint loaded. Exporting randomly initialized model.")

    model.eval()

    dummy_input = torch.randn(1, model_config.get('in_channels', 3), export_size, export_size, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with torch.inference_mode():
            print(f"Exporting to {output_path} (input {export_size}x{export_size}) using legacy TorchScript exporter for DirectML compatibility...")

            # 配置动态 Batch
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            torch.onnx.export(
                model,
                (dummy_input,),  # 必须以 tuple 形式传入 args
                output_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=['input'], # KawaiiSR.forward(x) -> 这里的名字对应 dynamic_axes 的 key
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False,
                dynamo=False # 显式禁用 Dynamo
            )
            
        print("Export successful!")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        print("Try reducing --input_size if OOM occurs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KawaiiSR (HAT) to ONNX")
    parser.add_argument("--config", type=str, default="models/configs/real_stage1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_size", type=int, default=256, help="Fixed input size for ONNX export")
    parser.add_argument("--output", type=str, default="models/kawaii_sr.onnx", help="Output ONNX file path")

    args = parser.parse_args()

    export_onnx(args.config, args.checkpoint, args.input_size, args.output)
