import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from HAT import HAT, CAB


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
        last_conv_in_channel = out_channels * (1 + num_layers)
        self.last_conv = BigConv(in_channels=last_conv_in_channel, out_channels=out_channels)

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
        out_x = [y]
        for i in range(self.num_layers):
            hid = self.feature_blocks[i](hid)
            hid = hid + y
            out_x.append(hid)

        out_x = torch.cat(out_x, dim = 1)
        out_x = self.last_conv(out_x)
        out_x = self.cab(out_x)
        return out_x


class KawaiiSR(nn.Module):
    """
    img_range: Image range. 1. or 255.
    """
    def __init__(self, 
                 img_size=64,
                 img_range = 1.,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 ):
        super(KawaiiSR, self).__init__()

        # 这里直接使用0.5 ，因为要应对不同的数据集与实际的情况
        rgb_mean = (0.5, 0.5, 0.5)
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.hat_out_channels = 64
        self.window_size = window_size
        self.hat_model = HAT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=self.hat_out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            resi_connection=resi_connection,
        )

        self.residual_block = ResidualBlock(num_layers=4, in_channels=self.hat_out_channels + in_chans, out_channels=32)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

    def _calculate_padding(self, height, width):
        """计算需要添加的padding"""
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return pad_h, pad_w

    def forward(self, x):
        B, C, H, W = x.shape
        assert(C == 3, "当前的通道数不是3！")

        original_h, original_w = H, W
        pad_h, pad_w = self._calculate_padding(H, W)
        if pad_h > 0 or pad_w > 0:
            # 在右边和下边添加padding，值为0
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H, W = x.shape[2], x.shape[3]  # 更新尺寸

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        hat_x = self.hat_model(x)
        target_size = (H * 2, W * 2)
        # 先生成一个基础的上采样的图像
        bicubic_x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        # 64 + 3 = 67
        x = torch.concat([hat_x, bicubic_x], dim = 1)
        x = self.residual_block(x)
        x = self.out_conv(x)
        x = x / self.img_range + self.mean

        # 如果之前添加了padding，现在需要裁剪回原始尺寸的2倍
        if pad_h > 0 or pad_w > 0:
            target_h = original_h * 2
            target_w = original_w * 2
            x = x[:, :, :target_h, :target_w]
        return x