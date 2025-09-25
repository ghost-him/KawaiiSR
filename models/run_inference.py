import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from KawaiiSR.KawaiiSR import KawaiiSR
from train_config import get_default_model_config


def load_model(weights_path: str, device: torch.device, force_tiling: bool | None = None,
               tile_size: int | None = None, tile_pad: int | None = None) -> KawaiiSR:
    """根据权重文件构建并加载模型。

    1. 若权重保存了完整 checkpoint（含 model_state_dict 与 config），优先使用其中的 model_config。
    2. 否则使用默认的 model_config。
    3. 可通过参数覆盖 tiling 相关设置。
    """
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        # 提取模型配置
        model_cfg = None
        if 'config' in ckpt and isinstance(ckpt['config'], dict):
            model_cfg = ckpt['config'].get('model_config')
        if model_cfg is None:
            model_cfg = get_default_model_config()
        # 覆盖 tiling
        if force_tiling is not None:
            model_cfg['use_tiling'] = force_tiling
        if tile_size is not None:
            model_cfg['tile_size'] = tile_size
        if tile_pad is not None:
            model_cfg['tile_pad'] = tile_pad
        model = KawaiiSR(**model_cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # 仅包含纯 state_dict
        model_cfg = get_default_model_config()
        if force_tiling is not None:
            model_cfg['use_tiling'] = force_tiling
        if tile_size is not None:
            model_cfg['tile_size'] = tile_size
        if tile_pad is not None:
            model_cfg['tile_pad'] = tile_pad
        model = KawaiiSR(**model_cfg).to(device)
        model.load_state_dict(ckpt)

    model.eval()
    return model


def list_images(path: Path, exts: Tuple[str, ...]) -> List[Path]:
    if path.is_file():
        return [path]
    imgs = []
    for ext in exts:
        imgs.extend(path.rglob(f'*.{ext}'))
    return sorted(imgs)


def build_transforms():
    # 与 data_loader 内保持一致: ToTensor + Normalize(0.5,0.5)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """将模型输出张量转换为 PIL.Image

    训练中指标函数假设模型输出在 [-1,1]，因此这里首先尝试按该区间反归一化；
    若检测到范围已经在 [0,1]（min>=-0.01 且 max<=1.01），则直接使用。
    """
    t = t.detach().cpu()
    mn = float(t.min())
    mx = float(t.max())
    if mn < -0.05 or mx > 1.05:  # 视为 [-1,1]
        t = (t * 0.5 + 0.5).clamp(0, 1)
    else:
        t = t.clamp(0, 1)
    t = (t * 255.0 + 0.5).byte()
    return transforms.ToPILImage()(t)


@torch.inference_mode()
def run_inference(model: KawaiiSR, img_paths: List[Path], device: torch.device, out_dir: Path,
                  fp16: bool = False, save_prefix: str = ''):
    out_dir.mkdir(parents=True, exist_ok=True)
    tfm = build_transforms()

    for p in img_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'[跳过] 无法打开 {p}: {e}')
            continue

        inp = tfm(img).unsqueeze(0).to(device)
        if fp16:
            with torch.autocast(device_type=device.type, enabled=True):
                sr = model(inp)
        else:
            sr = model(inp)
        sr_img = tensor_to_image(sr[0])
        save_name = f"{save_prefix}{p.stem}_x{getattr(model, 'scale', 2)}.png"
        sr_img.save(out_dir / save_name)
        print(f'[完成] {p.name} -> {save_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='KawaiiSR 推理脚本')
    parser.add_argument('--weights', type=str, required=True, help='训练好的权重路径 best.pth')
    parser.add_argument('--input', type=str, required=True, help='输入图片路径或文件夹')
    parser.add_argument('--output', type=str, default='./sr_outputs', help='输出文件夹')
    parser.add_argument('--exts', type=str, default='png,jpg,jpeg,webp', help='扫描的图片扩展名，逗号分隔')
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu / cuda:0 等')
    parser.add_argument('--fp16', action='store_true', help='启用半精度推理 (autocast)')
    parser.add_argument('--use-tiling', action='store_true', help='强制开启分块推理')
    parser.add_argument('--no-tiling', action='store_true', help='强制关闭分块推理')
    parser.add_argument('--tile-size', type=int, default=None, help='覆盖 tile_size')
    parser.add_argument('--tile-pad', type=int, default=None, help='覆盖 tile_pad')
    parser.add_argument('--prefix', type=str, default='', help='输出文件名前缀')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')

    if args.use_tiling and args.no_tiling:
        raise ValueError('不能同时指定 --use-tiling 与 --no-tiling')

    force_tiling = None
    if args.use_tiling:
        force_tiling = True
    elif args.no_tiling:
        force_tiling = False

    print(f'[信息] 加载模型: {args.weights}')
    model = load_model(
        weights_path=args.weights,
        device=device,
        force_tiling=force_tiling,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
    )
    print(f'[信息] 模型已加载。use_tiling={model.use_tiling} tile_size={model.tile_size} tile_pad={model.tile_pad}')

    in_path = Path(args.input)
    img_paths = list_images(in_path, tuple(e.strip() for e in args.exts.split(',')))
    if not img_paths:
        print('[错误] 未找到任何图片。')
        return
    print(f'[信息] 发现 {len(img_paths)} 张图片，开始推理...')

    run_inference(
        model=model,
        img_paths=img_paths,
        device=device,
        out_dir=Path(args.output),
        fp16=args.fp16,
        save_prefix=args.prefix,
    )
    print('[完成] 全部推理结束。')


if __name__ == '__main__':
    main()
