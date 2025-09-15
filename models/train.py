#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KawaiiSR 模型训练脚本
支持三阶段渐进式训练和断点恢复功能

使用示例:
    # 从头开始训练
    python train.py --config config.yaml --data_path /path/to/data
    
    # 从检查点恢复训练
    python train.py --resume /path/to/checkpoint.pth
    
    # 只训练特定阶段
    python train.py --stage stage2 --resume /path/to/stage1_best.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import signal
import atexit

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from KawaiiSR.KawaiiSR import KawaiiSR
from kawaii_trainer import KawaiiSRTrainer
from train_config import create_training_config, get_default_model_config
from checkpoint_manager import CheckpointManager, AutoSaveManager

class TrainingSession:
    """训练会话管理器"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.config = self._load_config()
        self.model = self._create_model()
        self.trainer = self._create_trainer()
        self.checkpoint_manager = self._create_checkpoint_manager()
        self.auto_save_manager = self._create_auto_save_manager()
        
        # 设置信号处理（优雅退出）
        self._setup_signal_handlers()
        
        # 训练状态
        self.is_training = False
        self.emergency_save_path = None
    
    def _setup_device(self) -> str:
        """设置训练设备"""
        if self.args.device:
            device = self.args.device
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _load_config(self):
        """加载训练配置"""
        if self.args.config and os.path.exists(self.args.config):
            # 从YAML文件加载配置
            with open(self.args.config, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 转换为TrainingConfig对象
            config = create_training_config(
                train_data_path=config_dict.get('train_data_path', self.args.data_path),
                val_data_path=config_dict.get('val_data_path', self.args.val_data_path),
                checkpoint_dir=config_dict.get('checkpoint_dir', self.args.checkpoint_dir),
                **config_dict.get('training_params', {})
            )
        else:
            # 使用默认配置
            config = create_training_config(
                train_data_path=self.args.data_path,
                val_data_path=self.args.val_data_path,
                checkpoint_dir=self.args.checkpoint_dir
            )
        
        # 命令行参数覆盖配置文件
        if self.args.batch_size:
            config.stages['stage1'].batch_size = self.args.batch_size
            config.stages['stage2'].batch_size = self.args.batch_size
            config.stages['stage3'].batch_size = self.args.batch_size
        
        if self.args.learning_rate:
            config.stages['stage1'].learning_rate = self.args.learning_rate
            config.stages['stage2'].learning_rate = self.args.learning_rate * 0.5
            config.stages['stage3'].learning_rate = self.args.learning_rate * 0.25
        
        return config
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        model_config = get_default_model_config()
        
        # 如果指定了模型配置文件
        if self.args.model_config and os.path.exists(self.args.model_config):
            with open(self.args.model_config, 'r', encoding='utf-8') as f:
                model_config.update(yaml.safe_load(f))
        
        model = KawaiiSR(**model_config)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1e6:.1f} MB")
        
        return model
    
    def _create_trainer(self) -> KawaiiSRTrainer:
        """创建训练器"""
        return KawaiiSRTrainer(
            model=self.model,
            config=self.config,
            device=self.device
        )
    
    def _create_checkpoint_manager(self) -> CheckpointManager:
        """创建检查点管理器"""
        return CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_checkpoints=self.args.max_checkpoints,
            auto_save_frequency=self.args.auto_save_frequency
        )
    
    def _create_auto_save_manager(self) -> AutoSaveManager:
        """创建自动保存管理器"""
        return AutoSaveManager(
            checkpoint_manager=self.checkpoint_manager,
            save_frequency=self.args.auto_save_frequency,
            save_on_improvement=True,
            improvement_threshold=0.1
        )
    
    def _setup_signal_handlers(self):
        """设置信号处理器（优雅退出）"""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, saving checkpoint and exiting...")
            self._emergency_save()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        
        # 注册退出时的清理函数
        atexit.register(self._cleanup)
    
    def _is_main_process(self) -> bool:
        """检查是否为主进程（避免多进程CUDA问题）"""
        try:
            import multiprocessing as mp
            return mp.current_process().name == 'MainProcess'
        except:
            return True
    
    def _emergency_save(self):
        """紧急保存（多进程安全）"""
        if not self.is_training or not self.trainer.optimizer:
            return
            
        # 检查是否为主进程，避免在fork的子进程中操作CUDA
        if not self._is_main_process():
            print("Skipping emergency save in subprocess to avoid CUDA errors")
            return
            
        try:
            # 确保CUDA同步
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except RuntimeError as e:
                    print(f"CUDA synchronization failed: {e}")
            
            emergency_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                scaler=self.trainer.scaler,
                epoch=self.trainer.current_epoch,
                global_step=self.trainer.global_step,
                stage=self.trainer.current_stage or 'unknown',
                metrics=self.trainer.best_metrics,
                config=self.config,
                additional_info={'emergency_save': True},
                is_auto_save=True
            )
            self.emergency_save_path = emergency_path
            print(f"Emergency checkpoint saved: {emergency_path}")
            
        except Exception as e:
            print(f"Failed to save emergency checkpoint: {e}")
            # 尝试保存最基本的模型权重
            try:
                emergency_dir = self.checkpoint_dir / "emergency"
                emergency_dir.mkdir(exist_ok=True)
                
                # 只保存模型权重，避免CUDA相关的状态
                model_path = emergency_dir / f"emergency_model_{self.trainer.global_step}.pth"
                
                # 安全获取模型状态
                if hasattr(self.model, 'state_dict'):
                    try:
                        state_dict = self.model.state_dict()
                        # 将所有tensor移到CPU
                        cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in state_dict.items()}
                        torch.save(cpu_state_dict, model_path)
                        print(f"Emergency model weights saved: {model_path}")
                    except Exception as model_e:
                        print(f"Failed to save emergency model weights: {model_e}")
                        
            except Exception as fallback_e:
                print(f"All emergency save attempts failed: {fallback_e}")
    
    def _cleanup(self):
        """清理资源"""
        if hasattr(self.trainer, 'writer') and self.trainer.writer:
            self.trainer.writer.close()
    
    def _resume_from_checkpoint(self) -> Optional[str]:
        """从检查点恢复"""
        resume_path = None
        
        if self.args.resume:
            if os.path.exists(self.args.resume):
                resume_path = self.args.resume
            else:
                print(f"Resume checkpoint not found: {self.args.resume}")
                return None
        elif self.args.auto_resume:
            # 自动查找最新的检查点
            resume_path = self.checkpoint_manager.find_latest_checkpoint(self.args.stage)
            if resume_path:
                print(f"Auto-resuming from: {resume_path}")
        
        if resume_path:
            # 验证检查点
            is_valid, message = self.checkpoint_manager.validate_checkpoint(resume_path)
            if not is_valid:
                print(f"Invalid checkpoint: {message}")
                return None
            
            print(f"Resuming training from: {resume_path}")
            return resume_path
        
        return None
    
    def train(self):
        """开始训练"""
        print("=== KawaiiSR Training Session Started ===")
        print(f"Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Data path: {self.config.train_data_path}")
        print(f"  Checkpoint dir: {self.config.checkpoint_dir}")
        print(f"  Mixed precision: {self.config.mixed_precision}")
        
        self.is_training = True
        
        try:
            # 检查是否需要恢复训练
            resume_path = self._resume_from_checkpoint()
            
            if self.args.stage:
                # 训练指定阶段
                stage_config = getattr(self.config, self.args.stage)
                self.trainer.train_stage(
                    stage_name=self.args.stage,
                    stage_config=stage_config,
                    resume_from_checkpoint=resume_path
                )
            else:
                # 训练所有阶段
                resume_stage = None
                if resume_path:
                    # 从检查点信息中获取阶段
                    checkpoint_info = self.checkpoint_manager.get_checkpoint_info(resume_path)
                    if checkpoint_info:
                        resume_stage = checkpoint_info.get('stage')
                
                self.trainer.train_all_stages(
                    resume_from_stage=resume_stage,
                    resume_checkpoint=resume_path
                )
            
            # 训练完成后的处理
            self._post_training_cleanup()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._emergency_save()
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self._emergency_save()
            raise
        finally:
            self.is_training = False
    
    def _post_training_cleanup(self):
        """训练后清理"""
        print("\n=== Training Completed ===")
        
        # 保存训练总结
        summary = self.trainer.get_training_summary()
        summary_path = Path(self.config.checkpoint_dir) / 'training_summary.json'
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Training summary saved: {summary_path}")
        
        # 保存损失权重历史
        history_path = Path(self.config.checkpoint_dir) / 'loss_weight_history.json'
        self.trainer.save_loss_weight_history(str(history_path))
        
        # 生成训练进度可视化
        self.trainer.visualize_training_progress()
        
        # 打印最终结果
        print(f"\nFinal Results:")
        print(f"  Best metrics: {self.trainer.best_metrics}")
        print(f"  Total training steps: {self.trainer.global_step}")
        print(f"  Final stage: {self.trainer.current_stage}")
        
        # 存储信息
        storage_info = self.checkpoint_manager.get_storage_info()
        print(f"\nStorage Info:")
        print(f"  Checkpoint directory: {storage_info.get('checkpoint_dir')}")
        print(f"  Total size: {storage_info.get('total_size_mb', 0):.1f} MB")
        print(f"  Number of checkpoints: {storage_info.get('checkpoint_count', 0)}")

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='KawaiiSR Model Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val_data_path', type=str,
        help='Path to validation data directory (default: use train_data_path)'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints'
    )
    
    # 配置文件
    parser.add_argument(
        '--config', type=str,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--model_config', type=str,
        help='Path to model configuration YAML file'
    )
    
    # 训练参数
    parser.add_argument(
        '--stage', type=str, choices=['stage1', 'stage2', 'stage3'],
        help='Train specific stage only'
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size (overrides config file)'
    )
    parser.add_argument(
        '--learning_rate', type=float,
        help='Learning rate (overrides config file)'
    )
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'],
        help='Training device'
    )
    
    # 恢复训练
    parser.add_argument(
        '--resume', type=str,
        help='Path to checkpoint file to resume from'
    )
    parser.add_argument(
        '--auto_resume', action='store_true',
        help='Automatically resume from latest checkpoint'
    )
    
    # 检查点管理
    parser.add_argument(
        '--max_checkpoints', type=int, default=5,
        help='Maximum number of checkpoints to keep'
    )
    parser.add_argument(
        '--auto_save_frequency', type=int, default=100,
        help='Auto-save frequency (in steps)'
    )
    
    # 其他选项
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser

def setup_logging(level: str, log_dir: str):
    """设置日志记录"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'training.log'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_args(args) -> bool:
    """验证命令行参数"""
    # 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"Error: Training data path does not exist: {args.data_path}")
        return False
    
    # 检查验证数据路径
    if args.val_data_path and not os.path.exists(args.val_data_path):
        print(f"Error: Validation data path does not exist: {args.val_data_path}")
        return False
    
    # 检查配置文件
    if args.config and not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        return False
    
    # 检查恢复检查点
    if args.resume and not os.path.exists(args.resume):
        print(f"Error: Resume checkpoint does not exist: {args.resume}")
        return False
    
    return True

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 验证参数
    if not validate_args(args):
        sys.exit(1)
    
    # 设置默认验证数据路径
    if not args.val_data_path:
        args.val_data_path = args.data_path
    
    # 设置日志
    setup_logging(args.log_level, args.checkpoint_dir)
    
    # 打印启动信息
    print("=" * 60)
    print("KawaiiSR Training Script")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("=" * 60)
    
    try:
        # 创建训练会话
        session = TrainingSession(args)
        
        # 开始训练
        session.train()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()