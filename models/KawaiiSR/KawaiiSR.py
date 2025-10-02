import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .HAT import HAT


class KawaiiSR(nn.Module):
    def __init__(self, 
                 image_size=64,
                 in_channels=3,
                 image_range = 1.,
                 use_tiling: bool = False,
                 tile_size: int = 256,
                 tile_pad: int = 10,
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
                 ):
        super(KawaiiSR, self).__init__()
        self.scale = 2 # 硬编码！
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        # 整个训练/推理流程统一使用 [0,1] 输入输出；此处不做额外均值归一化
        self.window_size = hat_window_size
        self.in_channels = in_channels
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

    def _calculate_padding(self, height, width):
        """计算需要添加的padding"""
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return pad_h, pad_w

    def forward_hat_block(self, x: torch.Tensor) -> torch.Tensor:
        """仅执行 HAT 模型，便于固定尺寸导出 ONNX。假设输入 x ∈ [0,1]。"""
        return self.hat_model(x)

    def _forward_hat_tiled(self, x):
        """对HAT模块进行分块前向传播。假设输入 x ∈ [0,1]，不做额外归一化。"""
        b, c, h, w = x.shape
        output_h = h * self.scale
        output_w = w * self.scale
        output_shape = (b, self.in_channels, output_h, output_w)

        # 创建一个空的输出张量用于拼接结果
        hat_output = x.new_zeros(output_shape)
        
        tiles_x = math.ceil(w / self.tile_size)
        tiles_y = math.ceil(h / self.tile_size)

        # 循环处理每个分块
        for y in range(tiles_y):
            for x_idx in range(tiles_x):
                # 提取带有重叠区域的输入分块
                ofs_x = x_idx * self.tile_size
                ofs_y = y * self.tile_size
                
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, w)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, h)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, w)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, h)

                input_tile = x[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # 通过HAT模型处理分块
                try:
                    with torch.no_grad(): # 在推理时使用no_grad以节省显存
                        output_tile = self.hat_model(input_tile)
                except RuntimeError as e:
                    print(f"处理分块时发生错误: {e}")
                    # 可以选择跳过或使用其他方式处理
                    continue
                
                # 计算输出分块的有效区域并拼接到最终输出张量中
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale
                
                hat_output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        
        return hat_output

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 3, "当前的通道数不是3！"

        # 1. 预处理：为了满足HAT的window_size要求，先对整图进行padding
        original_h, original_w = H, W
        pad_h, pad_w = self._calculate_padding(H, W)
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') # 使用reflect填充效果更好
        else:
            x_padded = x
        
        Hp, Wp = x_padded.shape[2:]

        # 2. 使用 DataLoader 的归一化（[0,1]），模型内部不再重复归一化
        x_norm = x_padded

        # 3. HAT模块处理（核心修改部分）
        # 如果启用分块，并且图像尺寸大于分块尺寸，则调用分块处理函数
        if self.use_tiling and (Hp > self.tile_size or Wp > self.tile_size):
            hat_x = self._forward_hat_tiled(x_norm)
        else:
            hat_x = self.hat_model(x_norm)

        # 4. 模型已经完成精炼，直接取输出
        x_final = hat_x

        # 5. 后处理：裁剪掉之前添加的padding
        if pad_h > 0 or pad_w > 0:
            target_h = original_h * self.scale
            target_w = original_w * self.scale
            x_final = x_final[:, :, :target_h, :target_w]
            
        return x_final