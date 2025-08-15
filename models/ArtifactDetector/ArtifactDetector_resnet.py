import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale
import math
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


class TransformerAggregator(nn.Module):
    """
    Transformer Encoder based aggregator for fusing patch features.
    Corresponds to Section 4.2 of the design document.
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Using learnable positional encoding for simplicity and flexibility
        # Max 500 patches should be sufficient for most high-res images.
        self.pos_embed = nn.Parameter(torch.zeros(1, 501, embed_dim)) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation=F.relu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: (B, N, E), B is batch size, N is num patches, E is embed_dim
        B, N, E = x.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, E)

        # Add positional encoding
        x = x + self.pos_embed[:, :(N + 1)]

        # Pass through transformer
        output = self.transformer_encoder(x) # (B, N+1, E)

        # Get the CLS token output which serves as the global feature
        cls_output = output[:, 0] # (B, E)
        # 移除了 .squeeze(0)
        return cls_output


class ArtifactDetectorResnet(nn.Module):
    """
    High-Resolution Image Compression Artifact Detector.
    Implements the architecture described in the design document.
    """
    def __init__(self,
                 backbone_name: str = 'resnet18',
                 aggregator: str = 'pooling', # 'pooling' or 'transformer'
                 pretrained: bool = True,
                 patch_size: int = 224,
                 dropout_rate: float = 0.5):
        super(ArtifactDetectorResnet, self).__init__()
        
        # --- Validate Inputs ---
        if backbone_name not in ['resnet18', 'resnet34']:
            raise ValueError("backbone_name must be 'resnet18' or 'resnet34'")
        if aggregator not in ['pooling', 'transformer']:
            raise ValueError("aggregator must be 'pooling' or 'transformer'")

        self.patch_size = patch_size
        self.aggregator_type = aggregator

        self.gray_downsample = nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=2)
        # --- Step 1: Input Preprocessing & Feature Engineering ---
        # 采用新的Sobel滤波器设计
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.reshape(1, 1, 3, 3), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.reshape(1, 1, 3, 3), requires_grad=False)

        # Wavelet Transform (DWT)
        self.dwt = WaveletTransform2D(wavelet="sym4")

        # --- Step 3: Local Feature Extraction (Backbone) ---
        self.backbone, self.feature_dim = self._create_backbone(backbone_name, pretrained)

        # --- Step 4: Global Feature Fusion ---
        if self.aggregator_type == 'pooling':
            # Max pooling will be done in the forward pass, no module needed.
            pass
        elif self.aggregator_type == 'transformer':
            self.aggregator = TransformerAggregator(embed_dim=self.feature_dim, num_heads=8, num_layers=2, dropout=dropout_rate)

        # --- Step 5: Final Classification ---
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim // 2, 2) # Binary classification (artifact vs. no artifact)
        )

    def _create_backbone(self, backbone_name, pretrained):
        """Creates the ResNet backbone and modifies its first layer for 4-channel input."""
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = 512
        else: # resnet34
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feature_dim = 512

        # --- Network Modification for 4-channel input ---
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )

        # Smartly initialize weights of the new layer from pretrained ones
        if pretrained:
            with torch.no_grad():
                # Average the RGB weights to initialize the 4th channel's weights
                mean_weights = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                new_weights = torch.cat([original_conv1.weight.data, mean_weights], dim=1)
                new_conv1.weight.data = new_weights

        backbone.conv1 = new_conv1
        backbone.fc = nn.Identity() # Remove final classifier
        return backbone, feature_dim
    
    def freeze_backbone(self):
        """For Stage 1 of two-stage training: freezes the ResNet backbone."""
        print("Freezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """For Stage 2 of two-stage training: unfreezes the ResNet backbone."""
        print("Unfreezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Processes a batch of high-resolution images.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
        """
        # 获取原始的batch size
        B = x.shape[0]

        # --- Step 1: Feature Engineering ---
        x_gray = rgb_to_grayscale(x)
        x_gray_downsample = self.gray_downsample(x_gray)
        sobel_x_feat = self.sobel_x(x_gray_downsample)
        sobel_y_feat = self.sobel_y(x_gray_downsample)

        _, lh, hl, hh = self.dwt(x_gray)
        
        # Downsample HPF feature to match DWT output size for concatenation
        h, w = lh.shape[-2:]

        # Concatenate features: [x_gray, sobel_x, sobel_y, DWT-LH, DWT-HL, DWT-HH]
        feature_map = torch.cat([x_gray_downsample, sobel_x_feat, sobel_y_feat, lh, hl, hh], dim=1)

        # --- Step 2: Image Patching ---
        # B, C, H, W = feature_map.shape # B已经获取
        C, H, W = feature_map.shape[1:]
        S = self.patch_size
        pad_h = (S - H % S) % S
        pad_w = (S - W % S) % S
        feature_map_padded = F.pad(feature_map, (0, pad_w, 0, pad_h), "constant", 0)

        patches = feature_map_padded.unfold(2, S, S).unfold(3, S, S)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, S, S)

        # --- Step 3: Local Feature Extraction ---
        local_features = self.backbone(patches) # shape: (B * num_patches, feature_dim)

        # --- Step 4: Global Feature Aggregation ---
        # 将patches的特征重新按batch分开
        local_features = local_features.view(B, -1, self.feature_dim) # shape: (B, num_patches, feature_dim)
        
        if self.aggregator_type == 'pooling':
            # Max pooling: 在每个图像的补丁维度上取最大值
            global_feature, _ = torch.max(local_features, dim=1) # shape: (B, feature_dim)
            # 移除了 global_feature.unsqueeze(0)
        else: # 'transformer'
            # local_features已经具备正确的(B, N, E)形状
            global_feature = self.aggregator(local_features) # shape: (B, feature_dim)
            
        # --- Step 5: Final Classification ---
        logits = self.classifier(global_feature) # shape: (B, 2)

        if return_features:
            return logits, global_feature
        else:
            return logits


# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Create a dummy high-resolution image (e.g., 1080p)
    dummy_image = torch.rand(1, 3, 1080, 1920).to(device)

    print("\n--- Testing Model with Pooling Aggregator (Section 4.1) ---")

    model_pooling = ArtifactDetectorResnet(backbone_name='resnet18', aggregator='pooling').to(device)
    model_pooling.eval()
    with torch.no_grad():
        output_pooling = model_pooling(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output logits shape (pooling): {output_pooling.shape}")
    print(f"Output logits (pooling): {output_pooling.cpu().numpy()}")
    
    # Demonstrate the two-stage training functions
    model_pooling.freeze_backbone()
    model_pooling.unfreeze_backbone()


    print("\n" + "="*50 + "\n")
    
    print("--- Testing Model with Transformer Aggregator (Section 4.2) ---")
    try:
        model_transformer = ArtifactDetectorResnet(backbone_name='resnet18', aggregator='transformer').to(device)
        model_transformer.eval()
        with torch.no_grad():
            output_transformer = model_transformer(dummy_image)
        print(f"Input shape: {dummy_image.shape}")
        print(f"Output logits shape (transformer): {output_transformer.shape}")
        print(f"Output logits (transformer): {output_transformer.cpu().numpy()}")
    except Exception as e:
        print(f"Error with transformer model: {e}")