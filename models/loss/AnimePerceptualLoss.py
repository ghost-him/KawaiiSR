import torch
import torch.nn as nn
from torchvision import models

class AnimePerceptualLoss(nn.Module):
    """
    一个基于在Danbooru2018数据集上预训练的ResNet-50模型的感知损失。
    这个类封装了模型加载、特征提取和损失计算的所有逻辑。

    Args:
        layer_weights (dict): 一个字典，指定了用于计算损失的层及其对应的权重。
                              键是层的标识符，值是浮点型的权重。
                              例如: {'conv1': 0.5, 'layer1': 1.0, 'layer2': 1.0}
        criterion (nn.Module): 用于比较特征图的损失函数，默认为 L1Loss。
        device (str or torch.device): 模型和计算所在的设备，如 'cuda' 或 'cpu'。
    """
    def __init__(self, layer_weights: dict, criterion: nn.Module = nn.L1Loss(), device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.layer_weights = {k: v for k, v in layer_weights.items()}
        self.criterion = criterion.to(self.device)
        
        # 1. 加载ResNet-50骨干网络
        base_model = models.resnet50(weights=None) # 不使用ImageNet预训练权重
        feature_extractor_body = nn.Sequential(*list(base_model.children())[:-2])

        # b) 加载Danbooru2018数据集的预训练权重
        state_dict_url = "https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth"
        # 使用weights_only=False以兼容PyTorch 2.6+的安全限制
        state = torch.hub.load_state_dict_from_url(state_dict_url, progress=True, map_location=self.device, weights_only=False)
        
        body_state = {}
        for key, val in state.items():
            if key.startswith('0.'):  # '0.' 是原始代码中body部分的网络层前缀
                new_key = key[2:]     # 去掉 '0.' 前缀以匹配我们的模型
                body_state[new_key] = val
        
        feature_extractor_body.load_state_dict(body_state)
        
        # 2. 冻结模型参数并设置为评估模式
        feature_extractor_body.eval()
        for param in feature_extractor_body.parameters():
            param.requires_grad = False
            
        self.model = feature_extractor_body.to(self.device)

        # 3. 定义输入图像的归一化参数 (与ImageNet标准相同)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))
        
        
        # 4. 存储需要提取特征的层
        self.feature_layers = {}
        model_children = list(self.model.children())

        # 建立一个查找表，将字符串映射到具体的层
        layer_map = {
            'conv1': model_children[0],
            # 原始代码中的 "0" 对应 ResNet 的 conv1
            
            # 原始代码中的 "4_2_conv3" 对应 layer1 的第3个 Bottleneck 的第3个卷积
            # PyTorch ResNet layer1 (model[4]) 有3个 Bottleneck 块，索引是 0, 1, 2
            'layer1_2_conv3': model_children[4][2].conv3,
            
            # 原始代码中的 "5_3_conv3" 对应 layer2 的第4个 Bottleneck 的第3个卷积
            # PyTorch ResNet layer2 (model[5]) 有4个 Bottleneck 块，索引是 0, 1, 2, 3
            'layer2_3_conv3': model_children[5][3].conv3,

            # 原始代码中的 "6_5_conv3" 对应 layer3 的第6个 Bottleneck 的第3个卷积
            # PyTorch ResNet layer3 (model[6]) 有6个 Bottleneck 块，索引是 0..5
            'layer3_5_conv3': model_children[6][5].conv3,

            # 原始代码中的 "7_2_conv3" 对应 layer4 的第3个 Bottleneck 的第3个卷积
            # PyTorch ResNet layer4 (model[7]) 有3个 Bottleneck 块，索引是 0, 1, 2
            'layer4_2_conv3': model_children[7][2].conv3
        }

        # 根据传入的 layer_weights 选择要使用的层
        for name in self.layer_weights.keys():
            if name in layer_map:
                self.feature_layers[name] = layer_map[name]
            else:
                print(f"Warning: Layer '{name}' not found in the predefined map.")

    def _extract_features(self, x: torch.Tensor) -> dict:
        """
        前向传播并从指定层提取特征图。
        """
        # 归一化输入图像
        x = (x - self.mean) / self.std
        
        features = {}
        # 使用钩子（hooks）来捕获中间层的输出，这是一种高效且灵活的方式
        hooks = []
        
        def get_hook(name):
            def hook_fn(module, input, output):
                features[name] = output
            return hook_fn

        for name, layer in self.feature_layers.items():
            hooks.append(layer.register_forward_hook(get_hook(name)))
            
        self.model(x)
        
        # 移除钩子，避免内存泄漏
        for h in hooks:
            h.remove()
            
        return features

    def forward(self, gen: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算生成图像 (gen) 和目标图像 (gt) 之间的感知损失。

        Args:
            gen (torch.Tensor): 生成的图像张量，形状为 (N, C, H, W)。
            gt (torch.Tensor): 真实的图像张量，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 计算出的总感知损失（标量）。
        """
        gen_features = self._extract_features(gen)
        with torch.no_grad():
            gt_features = self._extract_features(gt)

        total_loss = 0.0
        for name, weight in self.layer_weights.items():
            if name in gen_features:
                loss = self.criterion(gen_features[name], gt_features[name])
                total_loss += loss * weight
        
        return total_loss

### 如何使用精简后的代码

if __name__ == "__main__":
    # 定义需要计算损失的层及其权重
    # 注意：这里的键需要与PerceptualLoss类中定义的self.feature_layers的键匹配
    weights = {
        'conv1': 0.5,          
        'layer1_2_conv3': 20.0,
        'layer2_3_conv3': 30.0,
        'layer3_5_conv3': 1.0, 
        'layer4_2_conv3': 1.0  
    }
    
    # 检查是否有可用的CUDA设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 实例化感知损失模块
    perceptual_loss_fn = AnimePerceptualLoss(layer_weights=weights, device=device)
    
    # 创建两个随机的图像张量作为示例输入
    # 假设图像尺寸为 224x224
    batch_size = 4
    generated_image = torch.rand(batch_size, 3, 224, 224).to(device)
    ground_truth_image = torch.rand(batch_size, 3, 224, 224).to(device)
    
    # 计算损失
    loss = perceptual_loss_fn(generated_image, ground_truth_image)
    
    print(f"Calculated Perceptual Loss: {loss.item()}")

    # 检查模型参数是否已冻结
    for param in perceptual_loss_fn.parameters():
        assert not param.requires_grad, "Model parameters should be frozen!"
    print("All parameters are correctly frozen.")