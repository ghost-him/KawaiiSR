import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import pandas as pd
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SRDataset(Dataset):
    """超分辨率数据集，基于CSV索引文件加载预处理的数据（已移除阶段参数）"""

    def __init__(
        self,
        data_path: str,
        csv_file: str = 'dataset_index.csv'
    ):
        self.data_path = Path(data_path)
        
        # 读取CSV索引文件
        csv_path = self.data_path / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV索引文件不存在: {csv_path}")
        
        self.data_index = pd.read_csv(csv_path)
        print(f"从CSV文件加载数据集: {len(self.data_index)} 对图像")
        
        # 验证文件存在性
        self._validate_files()
        
        # 定义变换（统一到 [0,1]，不做额外归一化）
        self.to_tensor = transforms.ToTensor()

    def _validate_files(self):
        """验证CSV中列出的文件是否存在"""
        missing_files = []
        
        for idx, row in self.data_index.iterrows():
            lr_path = self.data_path / 'LR' / row['lr_image_path']
            hr_path = self.data_path / 'HR' / row['hr_image_path']
            
            if not lr_path.exists():
                missing_files.append(str(lr_path))
            if not hr_path.exists():
                missing_files.append(str(hr_path))
        
        if missing_files:
            print(f"警告: 发现 {len(missing_files)} 个缺失文件")
            if len(missing_files) <= 10:
                for file in missing_files:
                    print(f"  缺失: {file}")
            else:
                print(f"  前10个缺失文件: {missing_files[:10]}")
                print(f"  ... 还有 {len(missing_files) - 10} 个文件")
        
        # 过滤掉缺失文件的行
        valid_indices = []
        for idx, row in self.data_index.iterrows():
            lr_path = self.data_path / 'LR' / row['lr_image_path']
            hr_path = self.data_path / 'HR' / row['hr_image_path']
            if lr_path.exists() and hr_path.exists():
                valid_indices.append(idx)
        
        self.data_index = self.data_index.iloc[valid_indices].reset_index(drop=True)
        print(f"有效数据对: {len(self.data_index)}")
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """加载图像并转换为tensor"""
        try:
            image = Image.open(path).convert('RGB')
            tensor = self.to_tensor(image)  # [0,1]
            return tensor
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个默认tensor
            default_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
            tensor = self.to_tensor(default_image)  # [0,1]
            return tensor
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从CSV索引获取文件路径
        row = self.data_index.iloc[idx]
        lr_path = self.data_path / 'LR' / row['lr_image_path']
        hr_path = self.data_path / 'HR' / row['hr_image_path']
        
        # 加载预处理的图像
        lr_tensor = self._load_image(lr_path)
        hr_tensor = self._load_image(hr_path)
        
        return lr_tensor, hr_tensor

    def _setup_transforms(self):
        """设置数据变换（统一到 [0,1]，不做额外归一化）"""
        self.to_tensor = transforms.ToTensor()

def create_data_loaders(
    train_data_path: str,
    val_data_path: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_csv: str = 'dataset_index.csv',
    val_csv: str = 'dataset_index.csv'
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器（已移除阶段参数）"""
    
    # 创建数据集
    train_dataset = SRDataset(data_path=train_data_path, csv_file=train_csv)
    
    val_dataset = SRDataset(data_path=val_data_path, csv_file=val_csv)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

# 测试代码
if __name__ == '__main__':
    # 测试数据加载器
    try:
        dataset = SRDataset(
            data_path='./data/train',
            csv_file='dataset_index.csv'
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        for i, (lr, hr) in enumerate(dataloader):
            print(f"Batch {i}: LR shape: {lr.shape}, HR shape: {hr.shape}")
            print(f"LR range: [{lr.min():.3f}, {lr.max():.3f}]")
            print(f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")
            if i >= 2:  # 只测试前几个batch
                break
    except Exception as e:
        print(f"测试失败: {e}")