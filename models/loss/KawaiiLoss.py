import torch
import torch.nn as nn
import torch.nn.functional as F
from .CharbonnierLoss import CharbonnierLoss
from .LaplacianLoss import LaplacianLoss
from .HingeGANLoss import HingeDiscriminatorLoss, HingeGeneratorLoss
from .AnimePerceptualLoss import AnimePerceptualLoss

from .VGGPerceptualLoss import VGGPerceptualLoss
class KawaiiLoss(nn.Module):
    """
    用于超分辨率生成器的综合损失函数。
    
    Args:
        lambda_pixel (float): 像素级损失的权重 (使用 CharbonnierLoss)。
        lambda_frequency (float): 频域级损失的权重（如 Laplacian）。
        lambda_vgg (float): VGG损失的权重。
        lambda_perceptual (float): 感知损失的权重。
        lambda_adversarial (float): 对抗性损失的权重。
        enable_anime_loss (bool): 是否启用动漫特化感知损失。
        device (str): 计算设备。
    """
    def __init__(self, 
                 lambda_pixel: float = 1.0,
                 lambda_frequency: float = 1.0,
                 lambda_perceptual: float = 1.0,
                 lambda_adversarial: float = 0.005,
                 lambda_vgg: float = 1.0,
                 enable_anime_loss: bool = False,
                 device: str = 'cuda'):
        super().__init__()
        
        self.lambda_pixel = lambda_pixel
        self.lambda_frequency = lambda_frequency
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial
        self.lambda_vgg = lambda_vgg

        # 1. 像素级损失
        self.pixel_loss = CharbonnierLoss().to(device)
        
        # 2. 频域损失
        self.frequency_loss = LaplacianLoss().to(device)

        # 3. 感知损失 (动漫特化)
        self.perc_loss = None
        if enable_anime_loss:
            anime_perc_weights = {
                'conv1': 0.5, 'layer1_2_conv3': 20.0, 'layer2_3_conv3': 30.0, 
                'layer3_5_conv3': 1.0, 'layer4_2_conv3': 1.0
            }
            self.perc_loss = AnimePerceptualLoss(layer_weights=anime_perc_weights, device=device)

        # 4. 对抗性损失
        self.adv_loss = HingeGeneratorLoss().to(device)

        # 5. vgg损失
        layer_weights = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}
        vgg_type = 'vgg19'
        
        # 实例化损失函数
        self.vgg_loss = VGGPerceptualLoss(
            layer_weights,
            vgg_type,
            use_input_norm=True,
            range_norm=False,
            perceptual_weight=1.0,
            device=device,
        )

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor, fake_logits: torch.Tensor):
        """
        计算总的生成器损失。

        Args:
            sr_img (torch.Tensor): 生成的超分辨率图像。
            hr_img (torch.Tensor): 真实的高分辨率图像。
            fake_logits (torch.Tensor): 判别器对生成图像的输出。

        Returns:
            Tuple[torch.Tensor, dict]: 总损失和包含各分量损失的字典。
        """
        loss_components = {}
        total_loss = 0
        
        # 1. 像素损失
        if self.lambda_pixel > 0:
            loss_p = self.pixel_loss(sr_img, hr_img)
            loss_components['pixel'] = loss_p.item()
            total_loss += self.lambda_pixel * loss_p
        
        # 2. 频域损失
        if self.lambda_frequency > 0:
            loss_f = self.frequency_loss(sr_img, hr_img)
            loss_components['frequency'] = loss_f.item()
            total_loss += self.lambda_frequency * loss_f

        # 3. 感知损失
        if self.lambda_perceptual > 0 and self.perc_loss is not None:
            loss_perc = self.perc_loss(sr_img, hr_img)
            loss_components['perceptual'] = loss_perc.item()
            total_loss += self.lambda_perceptual * loss_perc

        # 4. 对抗损失
        if self.lambda_adversarial > 0:
            loss_adv = self.adv_loss(fake_logits)
            loss_components['adversarial'] = loss_adv.item()
            total_loss += self.lambda_adversarial * loss_adv
            
        # 5. VGG损失
        if self.lambda_vgg > 0:
            loss_vgg = self.vgg_loss(sr_img, hr_img)
            loss_components['vgg'] = loss_vgg.item()
            total_loss += self.lambda_vgg * loss_vgg

        loss_components['total_g_loss'] = total_loss.item()

        return total_loss, loss_components

class DiscriminatorLoss(nn.Module):
    """
    用于判别器的Hinge Loss。
    """
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.criterion = HingeDiscriminatorLoss().to(device)

    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real_logits (torch.Tensor): 判别器对真实图像的输出。
            fake_logits (torch.Tensor): 判别器对生成图像的输出。

        Returns:
            torch.Tensor: 判别器损失。
        """
        return self.criterion(real_logits, fake_logits)


if __name__ == '__main__':
    # --- 配置 ---
    BATCH_SIZE = 2
    IMG_SIZE = 128
    SCALE = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running demonstration on device: {DEVICE}")

    # --- 1. 创建模拟模型 ---


    # 生成器 (超分网络)
    class MockGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 3*SCALE**2, 3, 1, 1), nn.PixelShuffle(SCALE))
        def forward(self, x): return self.main(x)
        
    # 判别器
    class MockDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2, True),
                nn.Flatten(), nn.Linear(32 * (IMG_SIZE // 4)**2, 1)
            )
        def forward(self, x): return self.main(x)

    # --- 2. 实例化模型和损失函数 ---
    generator = MockGenerator().to(DEVICE)
    discriminator = MockDiscriminator().to(DEVICE)

    
    # 初始化损失函数
    g_loss_fn = KawaiiLoss(device=DEVICE)
    d_loss_fn = DiscriminatorLoss(device=DEVICE)

    # --- 3. 创建模拟数据 ---
    # 假设输入是 32x32，目标是 128x128
    lr_image = torch.rand(BATCH_SIZE, 3, IMG_SIZE // SCALE, IMG_SIZE // SCALE).to(DEVICE)
    hr_image = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # --- 4. 模拟训练步骤 ---
    # --- 更新判别器 ---
    # 生成图像
    sr_image_detached = generator(lr_image).detach() # .detach() 避免梯度传到生成器
    # 获取判别器输出
    real_d_logits = discriminator(hr_image)
    fake_d_logits = discriminator(sr_image_detached)
    # 计算判别器损失
    d_loss = d_loss_fn(real_d_logits, fake_d_logits)
    # d_loss.backward()
    # d_optimizer.step()
    print(f"Calculated Discriminator Loss: {d_loss.item():.4f}\n")
    
    # --- 更新生成器 ---
    # 生成图像 (这次需要梯度)
    sr_image = generator(lr_image)
    # 获取判别器输出
    fake_g_logits = discriminator(sr_image)
    # 计算生成器损失
    g_loss, g_loss_components = g_loss_fn(sr_image, hr_image, fake_g_logits)
    # g_loss.backward()
    # g_optimizer.step()
    
    print("--- Generator Loss Breakdown ---")
    for name, value in g_loss_components.items():
        print(f"{name:<15}: {value:.4f}")