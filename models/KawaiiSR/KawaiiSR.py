import torch
import torch.nn as nn
from .HAT import HAT

class KawaiiSR(nn.Module):
    def __init__(self, 
                 image_size=64,
                 in_channels=3,
                 image_range = 1.,
                 hat_patch_size=1,
                 hat_body_hid_channels=96,
                 hat_upsampler_hid_channels=64,
                 hat_depths=(6, 6, 6, 6),
                 hat_num_heads=(6, 6, 6, 6),
                 hat_window_size=7,
                 hat_compress_ratio=3,
                 hat_squeeze_factor=30,
                 hat_conv_scale=0.01,
                 hat_overlap_ratio=0.5,
                 hat_mlp_ratio=4.,
                 hat_qkv_bias=True,
                 hat_drop_rate=0.,
                 hat_attn_drop_rate=0.,
                 hat_drop_path_rate=0.1,
                 hat_norm_layer=nn.LayerNorm,
                 hat_patch_norm=True,
                 hat_use_checkpoint=False,
                 hat_resi_connection='1conv',
                 tail_num_layers=4,
                 **kwargs
                 ):
        super(KawaiiSR, self).__init__()
        self.scale = 2 # HAT 默认放大倍数
        self.window_size = hat_window_size
        self.in_channels = in_channels
        self.image_range = image_range
        
        # 初始化 HAT 基底模型
        self.hat_model = HAT(
            image_size=image_size,
            patch_size=hat_patch_size,
            in_channels=in_channels,
            out_channels=in_channels,
            body_hid_channels=hat_body_hid_channels,
            upsampler_hid_channels=hat_upsampler_hid_channels,
            depths=hat_depths,
            num_heads=hat_num_heads,
            window_size=hat_window_size,
            compress_ratio=hat_compress_ratio,
            squeeze_factor=hat_squeeze_factor,
            conv_scale=hat_conv_scale,
            overlap_ratio=hat_overlap_ratio,
            mlp_ratio=hat_mlp_ratio,
            qkv_bias=hat_qkv_bias,
            drop_rate=hat_drop_rate,
            attn_drop_rate=hat_attn_drop_rate,
            drop_path_rate=hat_drop_path_rate,
            norm_layer=hat_norm_layer,
            patch_norm=hat_patch_norm,
            use_checkpoint=hat_use_checkpoint,
            resi_connection=hat_resi_connection,
            tail_num_layers=tail_num_layers,
        )

    def forward(self, x):
        """直接执行 HAT 模型，外部负责 padding 和分块"""
        return self.hat_model(x)

