import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)
    
    Used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        
        # Encoder (Downsampling)
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        
        # Decoder (Upsampling)
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        
        # Output layers
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        # Encoder (Downsampling)
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # Decoder (Upsampling)
        x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3_up), negative_slope=0.2, inplace=True)
        
        if self.skip_connection:
            x4 = x4 + x2
            
        x4_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4_up), negative_slope=0.2, inplace=True)
        
        if self.skip_connection:
            x5 = x5 + x1
            
        x5_up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5_up), negative_slope=0.2, inplace=True)
        
        if self.skip_connection:
            x6 = x6 + x0

        # Output layers
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Return intermediate feature maps for analysis"""
        features = {}
        
        # Encoder
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=False)
        features['x0'] = x0
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=False)
        features['x1'] = x1
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=False)
        features['x2'] = x2
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=False)
        features['x3'] = x3
        
        return features
    
def demo_usage():
    """演示如何使用UNet判别器"""
    print("\n=== UNet Discriminator 使用示例 ===\n")
    
    # 创建网络
    discriminator = UNetDiscriminatorSN(
        num_in_ch=3,      # RGB图像
        num_feat=64,      # 基础特征数
        skip_connection=True  # 使用跳跃连接
    )
    
    # 打印网络信息
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"网络参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"网络大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 测试不同输入尺寸
    test_sizes = [(16, 16), (64, 64), (128, 128), (256, 256)]
    
    print("\n输入尺寸测试:")
    for h, w in test_sizes:
        try:
            # 创建随机输入
            x = torch.randn(1, 3, h, w)
            
            # 前向传播
            with torch.no_grad():
                output = discriminator(x)
            
            print(f"输入: {x.shape} -> 输出: {output.shape} ✓")
            
        except Exception as e:
            print(f"输入: (1, 3, {h}, {w}) -> 错误: {e}")
    
    # 实际使用示例
    print("\n=== 实际使用示例 ===")
    
    # 模拟真实图像和生成图像
    real_images = torch.randn(4, 3, 128, 128)  # 批次大小为4
    fake_images = torch.randn(4, 3, 128, 128)
    
    discriminator.train()
    
    # 判别真实图像
    real_output = discriminator(real_images)
    print(f"真实图像判别结果: {real_output.shape}")
    print(f"真实图像得分范围: [{real_output.min():.3f}, {real_output.max():.3f}]")
    
    # 判别生成图像
    fake_output = discriminator(fake_images)
    print(f"生成图像判别结果: {fake_output.shape}")
    print(f"生成图像得分范围: [{fake_output.min():.3f}, {fake_output.max():.3f}]")
    
    # 计算损失示例
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)
    
    criterion = nn.BCEWithLogitsLoss()
    real_loss = criterion(real_output, real_labels)
    fake_loss = criterion(fake_output, fake_labels)
    total_loss = (real_loss + fake_loss) / 2
    
    print(f"\n损失计算:")
    print(f"真实图像损失: {real_loss:.4f}")
    print(f"生成图像损失: {fake_loss:.4f}")
    print(f"总损失: {total_loss:.4f}")

def visualize_feature_maps():
    """可视化特征图"""
    print("\n=== 特征图可视化示例 ===")
    
    discriminator = UNetDiscriminatorSN(num_in_ch=3, num_feat=32)  # 使用较小的特征数便于展示
    discriminator.eval()
    
    # 创建测试输入
    x = torch.randn(1, 3, 64, 64)
    
    # 获取特征图
    with torch.no_grad():
        features = discriminator.get_feature_maps(x)
    
    print("各层特征图尺寸:")
    for name, feature in features.items():
        print(f"{name}: {feature.shape}")

# 运行demo
if __name__ == "__main__":
    demo_usage()
    visualize_feature_maps()