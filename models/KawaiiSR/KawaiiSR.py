import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from HAT import HAT, CAB
import math

class WaveletTransform2D(nn.Module):
    """Compute a two-dimensional wavelet transform.
    loss = nn.MSELoss()
    data = torch.rand(1, 3, 128, 256)
    DWT = WaveletTransform2D()
    IDWT = WaveletTransform2D(inverse=True)

    LL, LH, HL, HH = DWT(data) # (B, C, H / 2, W / 2) * 4
    recdata = IDWT([LL, LH, HL, HH])
    print(loss(data, recdata))
    """

    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform2D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

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

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)

        data_pad = torch.nn.functional.pad(
            data, [padl, padr, padt, padb], mode=self.mode
        )
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(
                    torch.nn.functional.conv2d(
                        data, filter.repeat(c, 1, 1, 1), stride=2, groups=c
                    )
                )
            return dec_res
        else:
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = torch.nn.functional.conv_transpose2d(
                data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c
            )
            return rec_res

class BigConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(BigConv, self).__init__()
        self.net = nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2, 
                     groups=in_channels, bias=use_bias),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """
    一个由多个 BigConv 组成的 ResidualBlock
    """
    def __init__(self, num_layers, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.down_channel_blocks = BigConv(in_channels=in_channels, out_channels=out_channels)

        self.feature_blocks = nn.ModuleList([
            BigConv(in_channels=out_channels, out_channels=out_channels)
            for _ in range(num_layers)
        ])
        self.last_conv = BigConv(in_channels=out_channels, out_channels=out_channels)

        self.cab = CAB(num_feat=out_channels, squeeze_factor=out_channels // 8)

        self.dwt = WaveletTransform2D()
        self.idwt = WaveletTransform2D(inverse=True)

        self.lf_branch = nn.Sequential(
            BigConv(in_channels=out_channels, out_channels=out_channels),
            BigConv(in_channels=out_channels, out_channels=out_channels)
        )

        self.hf_branch = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # features 列表用于存储每一层的输出和初始输入
        y = self.down_channel_blocks(x)
        LL, LH, HL, HH = self.dwt(y) # (B, C, H / 2, W / 2) * 4
        high = torch.cat([LH, HL, HH], dim = 1)
        high = self.hf_branch(high) + high
        LH, HL, HH = torch.chunk(high, 3, dim = 1)

        LL = self.lf_branch(LL) + LL
        wave_out = self.idwt([LL, LH, HL, HH])
        
        y = y + wave_out

        hid = y
        for i in range(self.num_layers):
            hid = self.feature_blocks[i](hid)
            hid = hid + y

        out_x = self.last_conv(hid)
        out_x = self.cab(out_x)
        return out_x


class KawaiiSR(nn.Module):
    def __init__(self, 
                 image_size=64,
                 in_channels=3,
                 image_range = 1.,
                 hid_channels = 32,
                 use_tiling: bool = False,
                 tile_size: int = 256,
                 tile_pad: int = 10,
                 hat_patch_size=1,
                 hat_hid_channels=96,
                 hat_out_channels = 64,
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
                 ):
        super(KawaiiSR, self).__init__()
        self.scale = 2 # 硬编码！
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        # 这里直接使用0.5 ，因为要应对不同的数据集与实际的情况
        rgb_mean = (0.5, 0.5, 0.5)
        self.img_range = image_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.hat_out_channels = hat_out_channels
        self.window_size = hat_window_size
        self.hat_model = HAT(
            image_size=image_size,
            patch_size=hat_patch_size,
            in_channels=in_channels,
            out_channels=self.hat_out_channels,
            hid_channels=hat_hid_channels,
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
        )

        self.residual_block = ResidualBlock(num_layers=4, in_channels=self.hat_out_channels + in_channels, out_channels=hid_channels)
        self.out_conv = nn.Conv2d(in_channels=hid_channels, out_channels=3, kernel_size=3, padding=1)

    def _calculate_padding(self, height, width):
        """计算需要添加的padding"""
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return pad_h, pad_w

    def _forward_hat_tiled(self, x):
        """对HAT模块进行分块前向传播"""
        b, c, h, w = x.shape
        output_h = h * self.scale
        output_w = w * self.scale
        output_shape = (b, self.hat_out_channels, output_h, output_w)

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

        # 2. 归一化
        self.mean = self.mean.type_as(x_padded)
        x_norm = (x_padded - self.mean) * self.img_range

        # 3. HAT模块处理（核心修改部分）
        # 如果启用分块，并且图像尺寸大于分块尺寸，则调用分块处理函数
        if self.use_tiling and (Hp > self.tile_size or Wp > self.tile_size):
            hat_x = self._forward_hat_tiled(x_norm)
        else:
            hat_x = self.hat_model(x_norm)
        
        # 4. 后续模块处理（不分块）
        target_size = (Hp * self.scale, Wp * self.scale)
        # 先生成一个基础的上采样的图像
        bicubic_x = F.interpolate(x_norm, size=target_size, mode='bicubic', align_corners=False)
        
        # 将HAT的输出和bicubic上采样的结果拼接
        x_concat = torch.cat([hat_x, bicubic_x], dim=1)
        
        # 通过残差模块和输出卷积
        x_res = self.residual_block(x_concat)
        x_out = self.out_conv(x_res)
        
        # 5. 反归一化
        x_final = x_out / self.img_range + self.mean

        # 6. 后处理：裁剪掉之前添加的padding
        if pad_h > 0 or pad_w > 0:
            target_h = original_h * self.scale
            target_w = original_w * self.scale
            x_final = x_final[:, :, :target_h, :target_w]
            
        return x_final