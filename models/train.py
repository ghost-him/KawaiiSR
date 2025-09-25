
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
    parser.add_argument('--train_dir', type=str, required=True, help='训练数据目录（含 LR/ 与 HR/ 及 dataset_index.csv）')
    parser.add_argument('--val_dir', type=str, required=True, help='验证数据目录（结构同上）')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='检查点保存目录')
    parser.add_argument('--config', type=str, required=True, help='训练配置 YAML 路径（扁平）')
    parser.add_argument('--resume', type=str, help='继续训练的检查点或权重路径')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    # 基础校验
    for p in [args.train_dir, args.val_dir]:
        if not os.path.exists(p):
            print(f'路径不存在: {p}')
            sys.exit(1)
    if not os.path.exists(args.config):
        print(f'配置文件不存在: {args.config}')
        sys.exit(1)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    # 构建配置
    cfg = create_training_config(
        train_data_path=args.train_dir,
        val_data_path=args.val_dir,
        checkpoint_dir=args.ckpt_dir,
        yaml_path=args.config,
    )
    print('=' * 60)
    print('KawaiiSR Minimal Training')
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Device: {cfg.device} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}")
    print('=' * 60)
    # 训练
    trainer = KawaiiTrainer(cfg)
    trainer.train(resume=args.resume)
if __name__ == '__main__':
    main()