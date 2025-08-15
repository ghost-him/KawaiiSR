# -*- coding: utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from torch import nn
from torchvision.models import vgg

class VGGFeatureExtractor(nn.Module):
    """
    VGG网络特征提取器 (更健壮、更高效的版本)

    此实现通过动态检查VGG结构来构建网络，而不是硬编码层名。
    这使其能够自动适应不同类型的VGG网络 (vgg11, 13, 16, 19, 带或不带BN)。

    Args:
        layer_name_list (list[str]): 需要提取特征的层名列表。
            例如: ['relu1_1', 'relu2_1', 'relu3_1']。
        vgg_type (str): VGG网络类型, 例如 'vgg19', 'vgg16_bn'。
        use_input_norm (bool): 是否对输入图像进行归一化。输入图像范围应为 [0, 1]。
        range_norm (bool): 是否将范围为 [-1, 1] 的图像归一化到 [0, 1]。
        requires_grad (bool): 是否需要计算VGG参数的梯度。
        remove_pooling (bool): 是否移除VGG中的最大池化层。
        pooling_stride (int): 最大池化层的步长。
        pretrained_path (str, optional): VGG预训练权重的本地路径。如果为None，则从torchvision下载。
    """

    def __init__(self,
                 layer_name_list: list[str],
                 vgg_type: str = 'vgg19',
                 use_input_norm: bool = True,
                 range_norm: bool = False,
                 requires_grad: bool = False,
                 remove_pooling: bool = False,
                 pooling_stride: int = 2,
                 pretrained_path: str = None):
        super().__init__()

        self.layer_name_list = set(layer_name_list)
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # 加载VGG模型
        if pretrained_path and os.path.exists(pretrained_path):
            vgg_net = getattr(vgg, vgg_type)(weights=None)
            state_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            # 使用新的weights API
            vgg_net = getattr(vgg, vgg_type)(weights='IMAGENET1K_V1')

        # 动态构建特征提取网络
        self.vgg_net = self._build_vgg_net(vgg_net.features, remove_pooling, pooling_stride)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False

        if self.use_input_norm:
            # ImageNet的均值和标准差
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _build_vgg_net(self, features: nn.Sequential, remove_pooling: bool, pooling_stride: int) -> nn.Sequential:
        """动态解析并构建所需的VGG网络部分"""
        modified_net = OrderedDict()
        
        # 确定需要构建到的最深层
        max_layer_name = max(self.layer_name_list, key=lambda name: int(''.join(filter(str.isdigit, name))))

        block_idx, conv_idx, pool_idx = 1, 1, 1
        for layer in features:
            if isinstance(layer, nn.Conv2d):
                name = f'conv{block_idx}_{conv_idx}'
                modified_net[name] = layer
                conv_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = f'relu{block_idx}_{conv_idx - 1}'
                # VGG的ReLU是inplace的，为保证特征图正确，替换为非inplace
                modified_net[name] = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{block_idx}'
                if not remove_pooling:
                    modified_net[name] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
                # 一个block结束，重置计数器
                block_idx += 1
                conv_idx = 1
                pool_idx += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{block_idx}_{conv_idx - 1}'
                modified_net[name] = layer
            
            # 如果已添加了最深的所需层，则停止构建
            if name == max_layer_name:
                break
        
        return nn.Sequential(modified_net)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播，返回一个包含所需层特征的字典"""
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for name, layer in self.vgg_net.named_children():
            x = layer(x)
            if name in self.layer_name_list:
                output[name] = x
        
        return output


class VGGPerceptualLoss(nn.Module):
    """
    感知损失，包含VGG特征损失。

    Args:
        layer_weights (dict): VGG各特征层损失的权重。
            例如: {'conv5_4': 1.}。
        vgg_type (str): VGG网络类型。
        use_input_norm (bool): 是否对VGG的输入进行归一化。
        range_norm (bool): 是否将范围为 [-1, 1] 的图像归一化到 [0, 1]。
        perceptual_weight (float): 感知损失的总权重。
        criterion (str): 感知损失的评估标准, 'l1' 或 'mse'。
        device (torch.device or str): 模型运行的设备。
    """
    def __init__(self,
                 layer_weights: dict[str, float],
                 vgg_type: str = 'vgg19',
                 use_input_norm: bool = True,
                 range_norm: bool = False,
                 perceptual_weight: float = 1.0,
                 device = None):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm).to(device)

        self.criterion = nn.L1Loss()
        
    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失。

        Args:
            x (Tensor): 输入图像 (例如，生成器输出)。
            gt (Tensor): 目标图像 (真实图像)。

        Returns:
            Tensor: 感知损失值。
        """
        # 提取VGG特征
        x_features = self.vgg(x)
        with torch.no_grad():
            gt_features = self.vgg(gt)

        # 计算感知损失
        percep_loss = 0.0
        if self.perceptual_weight > 0:
            for k in x_features.keys():
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        
        return percep_loss

if __name__ == "__main__":
    # --- 配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 定义需要计算损失的层及其权重
    layer_weights = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}
    vgg_type = 'vgg19'
    
    # 实例化损失函数
    perceptual_loss = VGGPerceptualLoss(
        layer_weights, 
        vgg_type, 
        perceptual_weight=1.0, 
        device=device
    )
    gen_tensor = torch.rand(1, 3, 256, 256).to(device)
    gt_tensor = torch.rand(1, 3, 256, 256).to(device)

    # --- 计算损失 ---
    loss = perceptual_loss(gen_tensor, gt_tensor)
    print(f"Calculated Perceptual Loss: {loss.item()}")

    # --- 打印模型参数量 ---
    total_params = sum(p.numel() for p in perceptual_loss.vgg.parameters())
    print(f"VGG Feature Extractor has {total_params / 1e6:.2f} M parameters")