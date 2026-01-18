
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
    cfg = create_training_config(
        train_data_path=args.train_dir,
        val_data_path=args.val_dir,
        checkpoint_dir=args.ckpt_dir,
        yaml_path=args.config,
    )

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

    # 处理自动恢复 (auto_resume)
    # 优先级: 命令行参数 > 配置文件 auto_resume
    resume_weights_path = args.resume_weights
    resume_state_path = args.resume_state

    if not resume_weights_path and not resume_state_path:
        # 如果命令行没有指定恢复参数，检查配置文件中的 auto_resume
        if cfg.auto_resume:
            ckpt_dir = Path(cfg.checkpoint_dir)
            auto_weights = ckpt_dir / 'last_weights.pth'
            auto_state = ckpt_dir / 'last_state.pth'

            if auto_weights.exists() and auto_state.exists():
                print(f'[Auto Resume] 发现上次的检查点，准备恢复训练: {auto_weights}')
                resume_weights_path = str(auto_weights)
                resume_state_path = str(auto_state)
            else:
                # 用户虽然启用了 auto_resume，但因为没有找到对应文件，可能意味着是首次训练。
                # 但根据用户“配置有问题直接抛出异常”的严格要求，如果要从头开始，应该显式设置 auto_resume=False。
                # 如果设置为 True 却找不到文件，抛出异常。
                raise FileNotFoundError(
                    f'[Auto Resume Error] 配置中启用了 auto_resume=True，但在 {ckpt_dir} 下未找到 last_weights.pth 或 last_state.pth。\n'
                    f'如果是首次训练，请将 auto_resume 设置为 False。'
                )

    trainer.train(
        load_weights=args.weights,
        resume_weights=resume_weights_path,
        resume_state=resume_state_path,
    )

if __name__ == '__main__':
    main()