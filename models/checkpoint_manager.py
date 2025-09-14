import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

from train_config import TrainingConfig

class CheckpointManager:
    """检查点管理器 - 负责保存和恢复训练状态"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        auto_save_frequency: int = 10
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_save_frequency = auto_save_frequency
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 检查点历史记录
        self.checkpoint_history = self._load_checkpoint_history()
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.checkpoint_dir / 'checkpoint.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_checkpoint_history(self) -> List[Dict[str, Any]]:
        """加载检查点历史记录"""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint history: {e}")
        return []
    
    def _save_checkpoint_history(self):
        """保存检查点历史记录"""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint history: {e}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        global_step: int,
        stage: str,
        metrics: Dict[str, float],
        config: TrainingConfig,
        additional_info: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        is_auto_save: bool = False
    ) -> str:
        """保存检查点"""
        
        # 创建检查点数据
        checkpoint_data = {
            'epoch': epoch,
            'global_step': global_step,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'config': asdict(config),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
        
        # 确定文件名
        if is_best:
            filename = f'best_{stage}.pth'
        elif is_auto_save:
            filename = f'auto_save_{stage}_epoch_{epoch}_step_{global_step}.pth'
        else:
            filename = f'{stage}_epoch_{epoch}_step_{global_step}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        try:
            # 保存检查点
            torch.save(checkpoint_data, checkpoint_path)
            
            # 记录到历史
            history_entry = {
                'filename': filename,
                'path': str(checkpoint_path),
                'epoch': epoch,
                'global_step': global_step,
                'stage': stage,
                'metrics': metrics,
                'timestamp': checkpoint_data['timestamp'],
                'is_best': is_best,
                'is_auto_save': is_auto_save,
                'file_size': checkpoint_path.stat().st_size
            }
            
            self.checkpoint_history.append(history_entry)
            self._save_checkpoint_history()
            
            self.logger.info(
                f"Checkpoint saved: {filename} "
                f"(Epoch: {epoch}, Step: {global_step}, Stage: {stage})"
            )
            
            # 清理旧的检查点
            if not is_best:
                self._cleanup_old_checkpoints(stage, is_auto_save)
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = 'cuda',
        strict: bool = True
    ) -> Dict[str, Any]:
        """加载检查点"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # 加载检查点数据
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载模型状态
            if strict:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 尝试部分加载
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                
                # 过滤掉不匹配的键
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                self.logger.warning(
                    f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters"
                )
            
            # 加载优化器状态
            if optimizer and checkpoint.get('optimizer_state_dict'):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    self.logger.warning(f"Failed to load optimizer state: {e}")
            
            # 加载调度器状态
            if scheduler and checkpoint.get('scheduler_state_dict'):
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"Failed to load scheduler state: {e}")
            
            # 加载混合精度状态
            if scaler and checkpoint.get('scaler_state_dict'):
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"Failed to load scaler state: {e}")
            
            self.logger.info(
                f"Checkpoint loaded: {checkpoint_path} "
                f"(Epoch: {checkpoint.get('epoch', 'N/A')}, "
                f"Step: {checkpoint.get('global_step', 'N/A')}, "
                f"Stage: {checkpoint.get('stage', 'N/A')})"
            )
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, stage: str, is_auto_save: bool = False):
        """清理旧的检查点文件"""
        try:
            # 获取当前阶段的检查点
            if is_auto_save:
                pattern = f'auto_save_{stage}_*.pth'
            else:
                pattern = f'{stage}_epoch_*.pth'
            
            checkpoints = sorted(
                self.checkpoint_dir.glob(pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # 保留最新的N个检查点
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[self.max_checkpoints:]:
                    checkpoint.unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint.name}")
                    
                    # 从历史记录中移除
                    self.checkpoint_history = [
                        entry for entry in self.checkpoint_history
                        if entry['filename'] != checkpoint.name
                    ]
                
                self._save_checkpoint_history()
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def find_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[str]:
        """查找最新的检查点"""
        try:
            if stage:
                # 查找特定阶段的最新检查点
                checkpoints = list(self.checkpoint_dir.glob(f'{stage}_*.pth'))
                checkpoints.extend(list(self.checkpoint_dir.glob(f'best_{stage}.pth')))
            else:
                # 查找所有检查点
                checkpoints = list(self.checkpoint_dir.glob('*.pth'))
            
            if not checkpoints:
                return None
            
            # 按修改时间排序，返回最新的
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            return str(latest_checkpoint)
            
        except Exception as e:
            self.logger.error(f"Failed to find latest checkpoint: {e}")
            return None
    
    def find_best_checkpoint(self, stage: str) -> Optional[str]:
        """查找最佳检查点"""
        best_checkpoint = self.checkpoint_dir / f'best_{stage}.pth'
        if best_checkpoint.exists():
            return str(best_checkpoint)
        return None
    
    def list_checkpoints(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        if stage:
            return [
                entry for entry in self.checkpoint_history
                if entry['stage'] == stage
            ]
        return self.checkpoint_history.copy()
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """获取检查点信息（不加载模型权重）"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 移除大的权重数据，只保留元信息
            info = {
                'epoch': checkpoint.get('epoch'),
                'global_step': checkpoint.get('global_step'),
                'stage': checkpoint.get('stage'),
                'metrics': checkpoint.get('metrics'),
                'timestamp': checkpoint.get('timestamp'),
                'pytorch_version': checkpoint.get('pytorch_version'),
                'cuda_version': checkpoint.get('cuda_version'),
                'file_size': Path(checkpoint_path).stat().st_size
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint info: {e}")
            return None
    
    def backup_checkpoint(self, checkpoint_path: str, backup_dir: Optional[str] = None) -> str:
        """备份检查点"""
        if backup_dir is None:
            backup_dir = self.checkpoint_dir / 'backups'
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = Path(checkpoint_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{checkpoint_file.stem}_{timestamp}{checkpoint_file.suffix}"
        backup_path = backup_dir / backup_filename
        
        try:
            shutil.copy2(checkpoint_path, backup_path)
            self.logger.info(f"Checkpoint backed up: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to backup checkpoint: {e}")
            raise
    
    def restore_from_backup(self, backup_path: str, target_path: Optional[str] = None) -> str:
        """从备份恢复检查点"""
        if target_path is None:
            backup_file = Path(backup_path)
            # 移除时间戳后缀
            original_name = '_'.join(backup_file.stem.split('_')[:-2]) + backup_file.suffix
            target_path = self.checkpoint_dir / original_name
        
        try:
            shutil.copy2(backup_path, target_path)
            self.logger.info(f"Checkpoint restored: {target_path}")
            return str(target_path)
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """验证检查点文件的完整性"""
        try:
            # 尝试加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 检查必要的键
            required_keys = ['model_state_dict', 'epoch', 'global_step', 'stage']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"
            
            # 检查模型状态字典
            if not isinstance(checkpoint['model_state_dict'], dict):
                return False, "Invalid model_state_dict format"
            
            return True, "Checkpoint is valid"
            
        except Exception as e:
            return False, f"Checkpoint validation failed: {e}"
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        try:
            total_size = sum(
                f.stat().st_size for f in self.checkpoint_dir.rglob('*.pth')
            )
            
            checkpoint_count = len(list(self.checkpoint_dir.glob('*.pth')))
            
            return {
                'checkpoint_dir': str(self.checkpoint_dir),
                'total_size_mb': total_size / (1024 * 1024),
                'checkpoint_count': checkpoint_count,
                'max_checkpoints': self.max_checkpoints,
                'history_entries': len(self.checkpoint_history)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {}

class AutoSaveManager:
    """自动保存管理器"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_frequency: int = 10,
        save_on_improvement: bool = True,
        improvement_threshold: float = 0.1
    ):
        self.checkpoint_manager = checkpoint_manager
        self.save_frequency = save_frequency
        self.save_on_improvement = save_on_improvement
        self.improvement_threshold = improvement_threshold
        
        self.last_save_step = 0
        self.best_metric = float('-inf')
    
    def should_save(
        self,
        global_step: int,
        current_metrics: Dict[str, float],
        force_save: bool = False
    ) -> bool:
        """判断是否应该自动保存"""
        if force_save:
            return True
        
        # 基于步数的保存
        if global_step - self.last_save_step >= self.save_frequency:
            return True
        
        # 基于性能改进的保存
        if self.save_on_improvement:
            current_psnr = current_metrics.get('psnr', 0)
            if current_psnr > self.best_metric + self.improvement_threshold:
                self.best_metric = current_psnr
                return True
        
        return False
    
    def auto_save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        global_step: int,
        stage: str,
        metrics: Dict[str, float],
        config: TrainingConfig,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """执行自动保存"""
        if self.should_save(global_step, metrics):
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                stage=stage,
                metrics=metrics,
                config=config,
                additional_info=additional_info,
                is_auto_save=True
            )
            
            self.last_save_step = global_step
            return checkpoint_path
        
        return None