
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小训练入口：单阶段、YAML主导配置、最小保存best/last"""
import os
import sys
import argparse
from pathlib import Path
import torch
# 确保可导入同目录模块
sys.path.append(str(Path(__file__).parent))
from train_config import create_training_config
from KawaiiTrainer import KawaiiTrainer
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='KawaiiSR Minimal Trainer')
    parser.add_argument('--train_dir', type=str, help='训练数据目录（含 LR/ 与 HR/ 及 dataset_index.csv）')
    parser.add_argument('--val_dir', type=str, help='验证数据目录（结构同上）')
    parser.add_argument('--ckpt_dir', type=str,  help='检查点保存目录')
    parser.add_argument('--config', type=str, required=True, help='训练配置 YAML 路径（扁平）')
    parser.add_argument('--weights', type=str, help='仅加载模型权重（不恢复优化器状态）')
    parser.add_argument('--resume-weights', type=str, help='恢复训练使用的模型权重路径')
    parser.add_argument('--resume-state', type=str, help='恢复训练使用的优化器/调度器状态路径')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    # 基础校验
    for p in [args.train_dir, args.val_dir]:
        if p and not os.path.exists(p):
            print(f'路径不存在: {p}')
            sys.exit(1)
    if not os.path.exists(args.config):
        print(f'配置文件不存在: {args.config}')
        sys.exit(1)
    # 构建配置
    try:
        cfg = create_training_config(
            train_data_path=args.train_dir,
            val_data_path=args.val_dir,
            checkpoint_dir=args.ckpt_dir,
            yaml_path=args.config,
        )
    except ValueError as exc:
        print(f'配置错误: {exc}')
        sys.exit(1)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for p in [cfg.train_data_path, cfg.val_data_path]:
        if not p or not os.path.exists(p):
            print(f'配置中的路径不存在: {p}')
            sys.exit(1)
    print('=' * 60)
    print('KawaiiSR Minimal Training')
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Device: {cfg.device} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}")
    print('=' * 60)
    torch.set_float32_matmul_precision('high')
    # 训练
    trainer = KawaiiTrainer(cfg)
    try:
        trainer.train(
            load_weights=args.weights,
            resume_weights=args.resume_weights,
            resume_state=args.resume_state,
        )
    except ValueError as exc:
        print(f'参数错误: {exc}')
        sys.exit(1)
if __name__ == '__main__':
    main()