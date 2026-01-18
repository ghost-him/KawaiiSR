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
    DWT = WaveletTransform2D(wavelet="bior1.3")
    IDWT = WaveletTransform2D(wavelet="bior1.3", inverse=True)

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
def visualize_wavelet_list(image_path, wavelet_list):
    """
    对列表中的每一个小波进行测试并保存结果
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"无法加载图片: {e}")
        return

    # 预处理
    preprocess = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Lambda(lambda x: x.unsqueeze(0))
    ])
    img_tensor = preprocess(img)

    # 循环测试每一个小波
    for w_name in wavelet_list:
        print(f"正在测试小波: {w_name} ...")
        
        try:
            # --- 核心修改在这里 ---
            DWT = WaveletTransform2D(wavelet=w_name) 
            
            with torch.no_grad():
                LL, LH, HL, HH = DWT(img_tensor)

            # 准备绘图
            components = {
                "LL (Approx)": LL.squeeze().cpu().numpy(),
                "LH (Horizontal)": LH.squeeze().cpu().numpy(),
                "HL (Vertical)": HL.squeeze().cpu().numpy(),
                "HH (Diagonal)": HH.squeeze().cpu().numpy()
            }

            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"Wavelet: {w_name}", fontsize=20, weight='bold')

            # 统一使用 gray colormap
            axes[0, 0].imshow(components["LL (Approx)"], cmap='gray')
            axes[0, 0].set_title("LL (Approx)")
            
            axes[0, 1].imshow(components["LH (Horizontal)"], cmap='gray')
            axes[0, 1].set_title("LH (Horizontal)")
            
            axes[1, 0].imshow(components["HL (Vertical)"], cmap='gray')
            axes[1, 0].set_title("HL (Vertical)")
            
            axes[1, 1].imshow(components["HH (Diagonal)"], cmap='gray')
            axes[1, 1].set_title("HH (Diagonal)")

            for ax in axes.ravel():
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存文件，文件名带上小波的名字
            save_name = f"result_{w_name}.png"
            plt.savefig(save_name)
            plt.close(fig) # 关闭画布以释放内存
            print(f"已保存: {save_name}")
            
        except Exception as e:
            print(f"小波 {w_name} 处理失败: {e}")


if __name__ == "__main__":
    LOCAL_IMAGE_PATH = "./test2.jpg" 
    
    # 这里定义你要测试的列表
    # 包含了之前分析的：Haar(基准), Bior系列(推荐), Db2(折中), Coif1
    WAVELETS_TO_TEST = [
        "haar",      # 基准：最清晰，但方块效应重
        "db2",       # 比haar稍平滑，比sym4锐利
        "bior1.3",   # 强烈推荐：动漫线条利器
        "bior4.4",   # 均衡型：压缩标准常用
        "rbio1.3",   # 反双正交：特定边缘提取
        "coif1"      # 细节保留较好
    ]
    
    visualize_wavelet_list(LOCAL_IMAGE_PATH, WAVELETS_TO_TEST)