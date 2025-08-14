import torch
from torchvision.models import resnet50
from thop import profile
from KawaiiSR import KawaiiSR

window_size = 16

model = KawaiiSR(
    img_size=64,
    window_size=window_size,
    depths=(2, 2, 2, 2),     # 使用较浅的深度以加速
    num_heads=(4, 4, 4, 4),    # 使用较少的头
    embed_dim=48,            # 使用较小的维度
    # upscale 在模型中硬编码为 2
)
model.eval()  # 切换到评估模式，这会禁用Dropout等层


# 定义一个示例输入
# 注意 batch_size 应该为 1，因为我们通常计算单次推理的 FLOPs
input_tensor = torch.randn(1, 3, 222, 222)

# 计算 MACs 和参数量
# thop 计算的是 MACs，不是 FLOPs
macs, params = profile(model, inputs=(input_tensor,))

# 转换为 GFLOPs
# FLOPs = 2 * MACs
gflops = (macs * 2) / 1e9
params_m = params / 1e6

print(f"Input shape: {input_tensor.shape}")
print(f"MACs: {macs/1e9:.2f} G")
print(f"FLOPs: {gflops:.2f} GFLOPs")
print(f"Parameters: {params_m:.2f} M")

# ResNet50 for 224x224 input: ~4.1 GFLOPs