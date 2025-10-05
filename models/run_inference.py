import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from KawaiiSR.KawaiiSR import KawaiiSR
from train_config import get_default_model_config


def _load_checkpoint(weights_path: Path, device: torch.device) -> Tuple[dict, dict | None]:
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        cfg = ckpt.get('config') if isinstance(ckpt.get('config'), dict) else None
    else:
        state_dict = ckpt
        cfg = None
    return state_dict, cfg


def _resolve_model_config(cfg_from_ckpt: dict | None, config_path: Path | None) -> dict:
    if cfg_from_ckpt and isinstance(cfg_from_ckpt.get('model_config'), dict):
        model_cfg = dict(cfg_from_ckpt['model_config'])
    elif config_path and config_path.exists():
        import yaml

        with config_path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        model_cfg = get_default_model_config()
        overrides = data.get('model_config') or data.get('model') or {}
        model_cfg.update(overrides)
    else:
        model_cfg = get_default_model_config()
    return model_cfg


def load_model(weights_path: Path, device: torch.device, config_path: Path | None,
               force_tiling: bool | None, tile_size: int | None, tile_pad: int | None) -> KawaiiSR:
    state_dict, cfg_from_ckpt = _load_checkpoint(weights_path, device)
    model_cfg = _resolve_model_config(cfg_from_ckpt, config_path)

    if force_tiling is not None:
        model_cfg['use_tiling'] = force_tiling
    if tile_size is not None:
        model_cfg['tile_size'] = tile_size
    if tile_pad is not None:
        model_cfg['tile_pad'] = tile_pad

    model = KawaiiSR(**model_cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def list_images(path: Path, extensions: Iterable[str]) -> List[Path]:
    if path.is_file():
        return [path]
    norm_exts = tuple(ext.lower().lstrip('.') for ext in extensions)
    results: List[Path] = []
    if not path.exists():
        return results
    for ext in norm_exts:
        results.extend(path.rglob(f'*.{ext}'))
        results.extend(path.rglob(f'*.{ext.upper()}'))
    return sorted(set(results))


def build_transforms() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    t = (t * 255.0 + 0.5).byte()
    return transforms.ToPILImage()(t)


@torch.inference_mode()
def run_inference(model: KawaiiSR, img_paths: List[Path], device: torch.device, out_dir: Path,
                  fp16: bool, save_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tfm = build_transforms()

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as exc:
            print(f'[跳过] 无法打开 {img_path}: {exc}')
            continue

        inp = tfm(img).unsqueeze(0).to(device)
        if fp16:
            with torch.autocast(device_type=device.type, enabled=True):
                sr = model(inp)
        else:
            sr = model(inp)

        sr_img = tensor_to_image(sr[0])
        scale = getattr(model, 'scale', 2)
        save_name = f"{save_prefix}{img_path.stem}_x{scale}.png"
        sr_img.save(out_dir / save_name)
        print(f'[完成] {img_path.name} -> {save_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='KawaiiSR 推理脚本')
    parser.add_argument('--weights', type=str, required=True, help='训练好的权重路径 (best_weights.pth / last_weights.pth，兼容 best.pth / last.pth)')
    parser.add_argument('--input', type=str, required=True, help='输入图片路径或文件夹')
    parser.add_argument('--output', type=str, default='./sr_outputs', help='输出文件夹')
    parser.add_argument('--config', type=str, default=None, help='可选的训练配置 YAML（将读取其中的 model_config）')
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu / cuda:0 等')
    parser.add_argument('--fp16', action='store_true', help='启用半精度推理 (autocast)')
    parser.add_argument('--use-tiling', action='store_true', help='强制开启分块推理')
    parser.add_argument('--no-tiling', action='store_true', help='强制关闭分块推理')
    parser.add_argument('--tile-size', type=int, default=None, help='覆盖 tile_size')
    parser.add_argument('--tile-pad', type=int, default=None, help='覆盖 tile_pad')
    parser.add_argument('--exts', type=str, default='png,jpg,jpeg,webp', help='扫描的图片扩展名，逗号分隔')
    parser.add_argument('--prefix', type=str, default='', help='输出文件名前缀')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device_req = args.device
    if torch.cuda.is_available() or 'cpu' in device_req:
        device = torch.device(device_req)
    else:
        device = torch.device('cpu')

    if args.use_tiling and args.no_tiling:
        raise ValueError('不能同时指定 --use-tiling 与 --no-tiling')
    force_tiling = True if args.use_tiling else False if args.no_tiling else None

    weights_path = Path(args.weights).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f'未找到权重文件: {weights_path}')

    print(f'[信息] 加载模型权重: {weights_path}')
    model = load_model(
        weights_path=weights_path,
        device=device,
        config_path=config_path,
        force_tiling=force_tiling,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
    )
    print(f'[信息] 模型 ready。use_tiling={getattr(model, "use_tiling", False)} tile_size={getattr(model, "tile_size", "?")}'
          f' tile_pad={getattr(model, "tile_pad", "?")}')

    extensions = [ext.strip() for ext in args.exts.split(',') if ext.strip()]
    img_paths = list_images(input_path, extensions)
    if not img_paths:
        raise FileNotFoundError(f'在 {input_path} 未找到匹配的图片 (扩展名: {extensions})')

    print(f'[信息] 共 {len(img_paths)} 张图片，开始推理...')
    run_inference(
        model=model,
        img_paths=img_paths,
        device=device,
        out_dir=output_path,
        fp16=args.fp16,
        save_prefix=args.prefix,
    )
    print('[完成] 推理结束，结果已保存。')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
