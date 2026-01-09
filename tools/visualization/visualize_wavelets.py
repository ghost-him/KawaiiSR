import torch
import torch.nn as nn
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from io import BytesIO

# ==============================================================================
# 您提供的 WaveletTransform2D 类
# ==============================================================================
class WaveletTransform2D(nn.Module):
    """
    Compute a two-dimensional wavelet transform.

    This revised version behaves like a standard CNN downsampling layer,
    halving the spatial dimensions for even-sized inputs.

    Example:
    loss = nn.MSELoss()
    # Use an even-sized input for perfect reconstruction example
    data = torch.rand(1, 3, 128, 256) 
    DWT = WaveletTransform2D(wavelet="sym4")
    IDWT = WaveletTransform2D(wavelet="sym4", inverse=True)

    LL, LH, HL, HH = DWT(data) # (B, C, H / 2, W / 2) * 4
    recdata = IDWT([LL, LH, HL, HH], original_size=data.shape)
    print(f"Reconstruction Loss: {loss(data, recdata)}")
    """

    def __init__(self, inverse=False, wavelet="haar", dtype=torch.float32):
        super(WaveletTransform2D, self).__init__()
        
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            # Forward DWT filters
            lo = torch.tensor(dec_lo, dtype=dtype).flip(-1).unsqueeze(0)
            hi = torch.tensor(dec_hi, dtype=dtype).flip(-1).unsqueeze(0)
        else:
            # Inverse DWT filters
            lo = torch.tensor(rec_lo, dtype=dtype).unsqueeze(0)
            hi = torch.tensor(rec_hi, dtype=dtype).unsqueeze(0)
        
        self.build_filters(lo, hi)
        
        # Calculate padding to behave like a standard strided convolution
        # P = (KernelSize - Stride) / 2 for 'same' output, but for stride=2,
        # we need P = (KernelSize - 2) / 2 to halve the dimension.
        self.padding = (self.dim_size - 2) // 2

    def build_filters(self, lo, hi):
        # construct 2d filter
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer("filters", filters)  # [4, 1, height, width]

    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul


    def forward(self, data, original_size=None):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            # We apply padding directly in the conv2d function
            for f in self.filters:
                dec_res.append(
                    F.conv2d(
                        data, f.repeat(c, 1, 1, 1), 
                        stride=2, 
                        groups=c,
                        padding=self.padding # Use calculated padding
                    )
                )
            return dec_res
        else:
            # Inverse transform
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = F.conv_transpose2d(
                data, self.filters.repeat(c, 1, 1, 1), 
                stride=2, 
                groups=c,
                padding=self.padding # Use calculated padding
            )
            
            if original_size is not None:
                _, _, H, W = original_size
                rec_res = rec_res[..., :H, :W]
            
            return rec_res



# ==============================================================================
# 新增的可视化函数
# ==============================================================================
def visualize_wavelet_transform(image_path):
    """
    加载本地图像，执行2D小波变换，并可视化四个分量。
    """
    # --- 1. 加载和预处理图像 ---
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{image_path}'")
        return
    except Exception as e:
        print(f"加载或处理图片时出错: {e}")
        return

    # 定义图像预处理流程
    # 转换为灰度图 -> 转换为Tensor (像素值缩放到[0,1]) -> 增加批次维度
    preprocess = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Lambda(lambda x: x.unsqueeze(0)) # 增加一个批次维度, 形状变为 [1, 1, H, W]
    ])
    
    img_tensor = preprocess(img)
    print(f"输入图像张量形状: {img_tensor.shape}")

    # --- 2. 执行小波变换 ---
    DWT = WaveletTransform2D(wavelet="haar")
    # 禁止梯度计算，因为我们只做前向传播
    with torch.no_grad():
        LL, LH, HL, HH = DWT(img_tensor)

    # --- 3. 准备数据用于绘图 ---
    components = {
        "Approximation (LL)": LL.squeeze().detach().cpu().numpy(),
        "Horizontal Detail (LH)": LH.squeeze().detach().cpu().numpy(),
        "Vertical Detail (HL)": HL.squeeze().detach().cpu().numpy(),
        "Diagonal Detail (HH)": HH.squeeze().detach().cpu().numpy()
    }

    # --- 4. 可视化 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("2D Wavelet Transform Components ('sym4' wavelet)", fontsize=16)

    (axes[0, 0].imshow(components["Approximation (LL)"], cmap='gray'), axes[0, 0].set_title("Approximation (LL)"))
    (axes[0, 1].imshow(components["Horizontal Detail (LH)"], cmap='gray'), axes[0, 1].set_title("Horizontal Detail (LH)"))
    (axes[1, 0].imshow(components["Vertical Detail (HL)"], cmap='gray'), axes[1, 0].set_title("Vertical Detail (HL)"))
    (axes[1, 1].imshow(components["Diagonal Detail (HH)"], cmap='gray'), axes[1, 1].set_title("Diagonal Detail (HH)"))

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.savefig('my_plot.png')


if __name__ == "__main__":
    LOCAL_IMAGE_PATH = "/home/user/gz/gz/KawaiiSR/models/test.png" 
    
    print("运行小波变换可视化...")
    print("需要的库: torch, pywavelets, matplotlib, pillow, torchvision")
    
    visualize_wavelet_transform(LOCAL_IMAGE_PATH)