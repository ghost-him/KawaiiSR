import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .MambaIRv2 import MambaIRv2

class KawaiiSR(nn.Module):
    def __init__(self, 
                 image_size=64,
                 in_channels=3,
                 image_range = 1.,
                 # MambaIRv2 defaults (SR2-L)
                 upscale=2,
                 embed_dim=174,
                 d_state=16,
                 depths=(6, 6, 6, 6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6),
                 window_size=16,
                 inner_rank=64,
                 num_tokens=128,
                 convffn_kernel_size=5,
                 mlp_ratio=2.0,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 # Keep old args to prevent breaks if passed, but ignore them
                 **kwargs
                 ):
        super(KawaiiSR, self).__init__()
        self.scale = upscale 
        self.window_size = window_size
        self.in_channels = in_channels
        
        # Initialize MambaIRv2
        self.mamba_model = MambaIRv2(
            img_size=image_size,
            patch_size=1, # Default in MambaIRv2
            in_chans=in_channels,
            embed_dim=embed_dim,
            d_state=d_state,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            upscale=upscale,
            img_range=image_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

    def forward(self, x):
        """
        前向传播。
        注意：输入图像 x 的长宽必须是 window_size 的倍数。
        这一点由外部调用者（如 ONNX Runtime 的预处理或其他框架）保证。
        模型内部不再进行 Padding 或 Tiling。
        """
        return self.mamba_model(x)

