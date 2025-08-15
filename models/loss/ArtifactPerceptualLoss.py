import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtifactPerceptualLoss(nn.Module):
    """
    该损失函数旨在惩罚那些被传入的分类器模型识别为"有伪影"（或任何非目标类别）的图像。
    通过最小化这个损失，可以引导一个生成模型产生更符合分类器"期望"的图像。

    使用方法:
    1. 在主程序中，首先初始化并加载你的伪影检测器（或任何分类器）模型。
    2. 将这个模型实例传递给本损失函数的构造函数进行初始化。
    3. 在训练循环中，将生成器输出的图像传入此损失函数的 forward 方法。
    4. 将返回的损失值添加到你的总损失中。
    """
    def __init__(self, artifact_detector: nn.Module):
        """
        初始化感知损失模块。

        Args:
            artifact_detector (nn.Module): 一个预训练并已加载权重的分类器模型。
                                           该模型应已被设置为评估模式 (eval())。
                                           损失函数将确保其参数被冻结。
        """
        super().__init__()
        
        # 1. 接收一个预先初始化好的模型实例
        self.artifact_detector = artifact_detector
        
        # 2. 从传入的模型中推断设备
        try:
            self.device = next(self.artifact_detector.parameters()).device
        except StopIteration:
            # 处理模型没有参数的情况（虽然不太可能）
            print("Warning: Artifact detector has no parameters. Defaulting device to 'cpu'.")
            self.device = torch.device("cpu")
            self.artifact_detector.to(self.device)

        print(f"Artifact Perceptual Loss initialized with a detector on device: '{self.device}'")

        # 3. 关键：损失函数仍然负责确保模型被冻结，以防意外训练
        print("Freezing Artifact Detector weights within the loss function.")
        self.artifact_detector.eval() # 确保是评估模式
        for param in self.artifact_detector.parameters():
            param.requires_grad = False
        
        # 4. 定义底层的损失计算标准
        self.criterion = nn.CrossEntropyLoss()

        print("Artifact Perceptual Loss is ready.")

    def forward(self, generated_image: torch.Tensor) -> torch.Tensor:
        """
        计算一批生成图像的感知损失。

        Args:
            generated_image (torch.Tensor): 从生成器输出的图像张量，
                                             形状为 (B, 3, H, W)，值域应为 [0, 1] 或 [-1, 1]。

        Returns:
            torch.Tensor: 一个标量张量，表示该批次图像的平均感知损失。
        """
        # 确保输入图像在正确的设备上
        generated_image = generated_image.to(self.device)

        # 1. 从伪影检测器获取预测 logits
        logits = self.artifact_detector(generated_image)

        # 2. 创建目标标签
        # 我们的目标是让模型认为图片是“无伪影”的，即类别 0。
        batch_size = generated_image.shape[0]
        target = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 3. 计算交叉熵损失
        loss = self.criterion(logits, target)
        
        return loss


# ==============================================================================
# 3. 主测试程序
# ==============================================================================
if __name__ == '__main__':
    class MockArtifactDetector(nn.Module):
        """
        一个简单的CNN分类器，用于模拟伪影检测。
        类别 0: "无伪影" (Good)
        类别 1: "有伪影" (Bad/Artifact)
        """
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv_stack = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 64x64 -> 32x32
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 32x32 -> 16x16
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 16x16 -> 8x8
                nn.Flatten(),
            )
            self.classifier = nn.Linear(64 * 8 * 8, num_classes)

        def forward(self, x):
            features = self.conv_stack(x)
            logits = self.classifier(features)
            return logits
    # --- 配置 ---
    IMG_SIZE = 64
    BATCH_SIZE = 4  # 使用一个大于1的批次大小进行演示
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demonstration on device: {DEVICE}")

    # 1. 初始化并加载伪影检测器
    mock_detector = MockArtifactDetector(num_classes=2).to(DEVICE)
    
    # 2. 实例化损失函数，并将检测器传入
    perceptual_loss_fn = ArtifactPerceptualLoss(artifact_detector=mock_detector)

    # 3. 创建一批随机的、假的"生成图像"用于测试
    #    这些图像就像是从你的生成器中输出的一样
    generated_images = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    # --- 计算损失 ---
    # 在这里使用 torch.no_grad() 是正确的，因为我们只是想计算一次损失值，
    # 而不是为了后续的反向传播和优化。
    with torch.no_grad():
        loss = perceptual_loss_fn(generated_images)
    
    print(f"\nCalculated Artifact Perceptual Loss: {loss.item():.4f}")

    # --- （可选）打印模型参数量，就像你的例子中一样 ---
    total_params = sum(p.numel() for p in perceptual_loss_fn.artifact_detector.parameters())
    print(f"Mock Artifact Detector has {total_params / 1e6:.2f} M parameters")