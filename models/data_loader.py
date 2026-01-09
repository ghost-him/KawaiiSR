import os
import io
import random
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from typing import Tuple, Optional, List
import pandas as pd
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# SRDataset and create_data_loaders have been removed.

# ==============================
# 在线随机裁剪/退化的数据集实现
# ==============================

def _get_resample_enum(name: str):
    """将字符串映射为 Pillow 的重采样枚举，兼容旧/新 Pillow 版本。"""
    # Pillow>=10 使用 Image.Resampling，旧版本使用顶层常量
    Resampling = getattr(Image, 'Resampling', Image)
    name = name.lower()
    if name in ('nearest', 'nn'):
        return Resampling.NEAREST
    if name in ('bilinear', 'triangle'):
        return Resampling.BILINEAR
    if name in ('bicubic', 'catmullrom'):
        return Resampling.BICUBIC
    if name in ('lanczos', 'lanczos3'):
        return Resampling.LANCZOS
    # 默认使用 LANCZOS
    return Resampling.LANCZOS


@dataclass
class OnTheFlyOptions:
    crop_size: int = 128                 # HR 裁剪尺寸
    scale: int = 2                      # 放大倍数（LR 的尺寸 = crop_size // scale）
    downsample_filters: Tuple[str, ...] = (
        'nearest', 'bilinear', 'bicubic', 'lanczos'
    )
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    rot90_p: float = 0.0                # 随机旋转90°（一次，0/90/180/270）
    jpeg_p: float = 0.0                 # 对 LR 进行 JPEG 退化的概率
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 95
    webp_p: float = 0.0                 # 对 LR 进行 WebP 退化的概率
    webp_quality_min: int = 50
    webp_quality_max: int = 95
    blur_p: float = 0.0                 # 对 LR 进行高斯模糊的概率
    blur_sigma_min: float = 0.2
    blur_sigma_max: float = 1.2
    noise_p: float = 0.0                # 向 LR 添加高斯噪声的概率（添加在 [0,1] 张量域）
    noise_std_min: float = 0.001
    noise_std_max: float = 0.01
    recursive: bool = True              # 是否递归扫描文件夹
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    seed: Optional[int] = None          # 设定后可复现


class OnTheFlySRDataset(Dataset):
    """
    在线生成 LR-HR 对：
    - 从原始图像文件夹中读取单张图
    - 随机裁剪 HR 128x128（可配）
    - 随机选择下采样算法，将 HR -> LR (128/scale)
    - 可选：对 LR 施加 JPEG 压缩、高斯模糊、噪声等退化

    返回: (lr_tensor [0,1], hr_tensor [0,1])
    """

    def __init__(self, image_dir: str, options: Optional[OnTheFlyOptions] = None, is_val: bool = False):
        self.root = Path(image_dir)
        self.opts = options or OnTheFlyOptions()
        self.is_val = is_val
        if self.opts.seed is not None:
            random.seed(self.opts.seed)
        self.files = self._scan_files()
        if len(self.files) == 0:
            raise FileNotFoundError(f"未在 {self.root} 中找到图像: 扩展名 {self.opts.extensions}")
        if self.opts.crop_size % self.opts.scale != 0:
            raise ValueError(f"crop_size 必须能被 scale 整除: {self.opts.crop_size} vs {self.opts.scale}")
        self.to_tensor = transforms.ToTensor()

    def _scan_files(self) -> List[Path]:
        pattern = '**/*' if self.opts.recursive else '*'
        files = []
        for p in self.root.glob(pattern):
            if p.is_file() and p.suffix.lower() in self.opts.extensions:
                files.append(p)
        return sorted(files)

    def __len__(self):
        return len(self.files)

    def _open_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _get_image(self, idx: int) -> Image.Image:
        # 直接从磁盘读取
        return self._open_rgb(self.files[idx])

    def _rand_crop_hr(self, img: Image.Image, crop_size: int) -> Image.Image:
        w, h = img.size
        if w < crop_size or h < crop_size:
            # 过小则等比例放大到最小边 >= crop_size
            scale = max(crop_size / max(1, w), crop_size / max(1, h))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), _get_resample_enum('bicubic'))
            w, h = img.size
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        return img.crop((x, y, x + crop_size, y + crop_size))

    def _maybe_aug_hr(self, hr: Image.Image) -> Image.Image:
        if self.is_val:
            return hr
        # 随机水平/垂直翻转
        if random.random() < self.opts.hflip_p:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.opts.vflip_p:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        # 随机 0/90/180/270 旋转
        if random.random() < self.opts.rot90_p:
            k = random.choice([1, 2, 3])
            hr = hr.rotate(90 * k, expand=False)
        return hr

    def _downsample_to_lr(self, hr: Image.Image) -> Image.Image:
        lr_size = self.opts.crop_size // self.opts.scale
        resample_name = random.choice(self.opts.downsample_filters)
        resample = _get_resample_enum(resample_name)
        return hr.resize((lr_size, lr_size), resample)

    def _maybe_degrade_lr(self, lr: Image.Image) -> Image.Image:
        if self.is_val:
            return lr
        # 高斯模糊
        if self.opts.blur_p > 0 and random.random() < self.opts.blur_p:
            sigma = random.uniform(self.opts.blur_sigma_min, self.opts.blur_sigma_max)
            lr = lr.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        # 压缩退化（JPEG 或 WebP，仅选其一）
        apply_jpeg = self.opts.jpeg_p > 0 and random.random() < self.opts.jpeg_p
        apply_webp = self.opts.webp_p > 0 and random.random() < self.opts.webp_p
        mode = None
        if apply_jpeg and apply_webp:
            mode = random.choice(['jpeg', 'webp'])
        elif apply_jpeg:
            mode = 'jpeg'
        elif apply_webp:
            mode = 'webp'

        if mode == 'jpeg':
            q = random.randint(self.opts.jpeg_quality_min, self.opts.jpeg_quality_max)
            buf = io.BytesIO()
            lr.save(buf, format='JPEG', quality=int(q))
            buf.seek(0)
            lr = Image.open(buf).convert('RGB')
        elif mode == 'webp':
            q = random.randint(self.opts.webp_quality_min, self.opts.webp_quality_max)
            buf = io.BytesIO()
            lr.save(buf, format='WEBP', quality=int(q))
            buf.seek(0)
            lr = Image.open(buf).convert('RGB')
        return lr

    def _maybe_add_noise(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_val or self.opts.noise_p <= 0:
            return lr_tensor
        if random.random() < self.opts.noise_p:
            std = random.uniform(self.opts.noise_std_min, self.opts.noise_std_max)
            noise = torch.randn_like(lr_tensor) * float(std)
            lr_tensor = (lr_tensor + noise).clamp(0.0, 1.0)
        return lr_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._get_image(idx)
        hr = self._rand_crop_hr(img, self.opts.crop_size)
        hr = self._maybe_aug_hr(hr)
        lr = self._downsample_to_lr(hr)
        lr = self._maybe_degrade_lr(lr)

        hr_t = self.to_tensor(hr)
        lr_t = self.to_tensor(lr)
        lr_t = self._maybe_add_noise(lr_t)
        return lr_t, hr_t


def create_random_sr_loaders(
    train_image_dir: str,
    val_image_dir: Optional[str],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    options: Optional[OnTheFlyOptions] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    基于 OnTheFlySRDataset 创建训练/验证 DataLoader。
    - 训练集使用随机裁剪与退化
    - 验证集（若提供路径）使用同样裁剪但不做退化/增强（is_val=True）
    """

    train_ds = OnTheFlySRDataset(train_image_dir, options=options, is_val=False)
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
    }
    if num_workers > 0:
        if prefetch_factor is not None:
            train_loader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
        if persistent_workers:
            train_loader_kwargs['persistent_workers'] = True
    train_loader = DataLoader(
        train_ds,
        **train_loader_kwargs,
    )

    val_loader = None
    if val_image_dir:
        val_ds = OnTheFlySRDataset(val_image_dir, options=options, is_val=True)
        val_loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': False,
        }
        if num_workers > 0:
            if prefetch_factor is not None:
                val_loader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
            if persistent_workers:
                val_loader_kwargs['persistent_workers'] = True
        val_loader = DataLoader(
            val_ds,
            **val_loader_kwargs,
        )
    return train_loader, val_loader


if __name__ == '__main__':
    pass
