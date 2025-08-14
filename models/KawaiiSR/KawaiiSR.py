import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt



class KawaiiSR(nn.Module):
    def __init__(self):
        super(KawaiiSR, self).__init__()


    def forward(self, x):
        B, C, H, W = x.shape
        assert(C == 4, "当前的通道数不是4！")
        target_size = (H * 2, W * 2)
        # 先生成一个基础的上采样的图像
        x_base_upsample = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)


        return x_base_upsample