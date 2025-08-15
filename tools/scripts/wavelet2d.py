import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
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


loss = nn.MSELoss()
# Use an even-sized input for perfect reconstruction example
data = torch.rand(1, 3, 1080, 1920) 
DWT = WaveletTransform2D(wavelet="sym4")
IDWT = WaveletTransform2D(wavelet="sym4", inverse=True)
LL, LH, HL, HH = DWT(data) # (B, C, H / 2, W / 2) * 4
print(LL.shape)
recdata = IDWT([LL, LH, HL, HH])
print(f"Reconstruction Loss: {loss(data, recdata)}")