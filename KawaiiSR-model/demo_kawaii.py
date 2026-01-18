#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KawaiiSR 模型快速检查脚本。

- 与 train.py 保持同一目录结构与参数输入。
- 基于 YAML 配置构建模型。
- 可执行一次或多次前向推理，并输出参数规模 / FLOPs 等信息。
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

# 确保可以导入同目录下的模块
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

from train_config import create_training_config  # noqa: E402
from KawaiiSR.KawaiiSR import KawaiiSR  # noqa: E402

try:  # noqa: E402
    from thop import profile  # type: ignore
except ImportError:  # noqa: E402
    profile = None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KawaiiSR 模型检查工具 (同 train.py 接口)"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="训练数据目录（含 LR/ 与 HR/ 及 dataset_index.csv）",
    )
    parser.add_argument(
        "--val_dir", type=str, required=True, help="验证数据目录（结构同上）"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="检查点保存目录"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置 YAML 路径（扁平）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="保持与 train.py 接口一致（该脚本不会使用）",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="选择用于推理的设备，默认自动检测。",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="指定计算 FLOPs/前向推理的输入尺寸，默认使用配置中的 image_size。",
    )
    parser.add_argument(
        "--demo_sizes",
        type=int,
        nargs="+",
        metavar=("H", "W"),
        help="可选：额外指定若干组演示尺寸，格式为 H W [H W ...]。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="推理与 FLOPs 计算所使用的 batch size。",
    )
    parser.add_argument(
        "--no_demo",
        action="store_true",
        help="只输出模型统计信息，不进行前向推理演示。",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="在正式统计前进行的前向次数，用于热身。",
    )
    return parser


def ensure_required_paths(args: argparse.Namespace) -> None:
    for path in (args.train_dir, args.val_dir, args.config):
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)


def parse_demo_sizes(
    base_hw: Tuple[int, int], extra_sizes: Optional[Sequence[int]]
) -> List[Tuple[int, int]]:
    if not extra_sizes:
        return [base_hw]
    if len(extra_sizes) % 2 != 0:
        raise ValueError("--demo_sizes 需要以 H W 成对出现")
    it: Iterable[Tuple[int, int]] = zip(
        extra_sizes[0::2], extra_sizes[1::2]
    )
    sizes = [tuple(map(int, pair)) for pair in it]
    if base_hw not in sizes:
        sizes.insert(0, base_hw)
    return sizes


def select_device(preferred: str, override: str) -> torch.device:
    candidates: List[str] = []
    if override != "auto":
        candidates.append(override)
    candidates.append(preferred)
    candidates.extend(["cuda", "mps", "cpu"])
    for candidate in candidates:
        if candidate == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if candidate == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if candidate == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def build_model(model_cfg: dict) -> KawaiiSR:
    model = KawaiiSR(**model_cfg)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_profile(
    model: torch.nn.Module,
    batch_size: int,
    input_hw: Tuple[int, int],
) -> Tuple[Optional[float], Optional[float]]:
    if profile is None:
        return None, None
    original_device = next(model.parameters()).device
    dummy = torch.randn(batch_size, model.in_channels, input_hw[0], input_hw[1])
    try:
        model_cpu = model.to("cpu")
        with torch.no_grad():
            macs, _ = profile(model_cpu, inputs=(dummy,), verbose=False)
    except Exception:
        model.to(original_device)
        return None, None
    model.to(original_device)
    return macs, macs * 2 if macs is not None else None


def run_demo(
    model: KawaiiSR,
    device: torch.device,
    sizes: Sequence[Tuple[int, int]],
    batch_size: int,
    warmup: int,
) -> None:
    model.to(device)
    scale = getattr(model, "scale", 2)
    for idx, (h, w) in enumerate(sizes, start=1):
        print(f"\n--- Demo {idx}: 输入 {h}x{w} ---")
        dummy = torch.rand(batch_size, model.in_channels, h, w, device=device)
        print(f"输入张量: {tuple(dummy.shape)} | 设备: {dummy.device}")
        with torch.no_grad():
            for _ in range(max(warmup, 0)):
                _ = model(dummy)
            output = model(dummy)
        print(
            f"输出张量: {tuple(output.shape)} | 期望尺寸: (batch={batch_size}, channels={model.in_channels},"
            f" height={h * scale}, width={w * scale})"
        )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    ensure_required_paths(args)

    cfg = create_training_config(
        train_data_path=args.train_dir,
        val_data_path=args.val_dir,
        checkpoint_dir=args.ckpt_dir,
        yaml_path=args.config,
    )

    model_cfg = dict(cfg.model_config)
    base_size = (
        args.input_size[0],
        args.input_size[1],
    ) if args.input_size else (
        int(model_cfg.get("image_size", 64)),
        int(model_cfg.get("image_size", 64)),
    )
    sizes = parse_demo_sizes(base_size, args.demo_sizes)

    device = select_device(cfg.device, args.device)
    print("=" * 60)
    print("KawaiiSR 模型检查")
    print(f"配置文件: {args.config}")
    print(f"推理设备: {device}")
    print(f"Batch Size: {args.batch_size} | 输入尺寸: {base_size}")
    print("=" * 60)

    model = build_model(model_cfg)

    total_params, trainable_params = count_parameters(model)
    macs, flops = compute_profile(model, args.batch_size, base_size)

    print("模型统计:")
    print(f"  总参数量: {total_params / 1e6:.2f} M ({total_params:,})")
    print(f"  可训练参数: {trainable_params / 1e6:.2f} M")
    if macs is not None and flops is not None:
        print(f"  MACs@{base_size[0]}x{base_size[1]}: {macs / 1e9:.2f} G")
        print(f"  FLOPs@{base_size[0]}x{base_size[1]}: {flops / 1e9:.2f} G")
    else:
        if profile is None:
            print("  未安装 thop，跳过 FLOPs 计算 (pip install thop)")
        else:
            print("  FLOPs 计算失败，可能是输入尺寸或设备不受支持。")

    if not args.no_demo:
        run_demo(
            model=model,
            device=device,
            sizes=sizes,
            batch_size=args.batch_size,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
