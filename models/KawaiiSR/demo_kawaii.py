import torch
from KawaiiSR import KawaiiSR

def run_inference_and_print(model, demo_name, dims, scale, device):
    """
    执行一次推理并打印输入输出信息。

    Args:
        model (torch.nn.Module): 要使用的模型。
        demo_name (str): 当前演示的名称。
        dims (tuple): 输入图片的尺寸 (height, width)。
        scale (int): 模型的上采样倍数。
        device (str): 运行的设备 ('cpu', 'cuda', 'mps')。
    """
    print(f"\n--- {demo_name}: ({dims[0]}x{dims[1]}) ---")
    h, w = dims

    # 创建一个随机的输入图片张量，并将其移动到指定设备
    input_image = torch.rand(1, 3, h, w).to(device)
    print(f"输入图片形状: {input_image.shape}, 设备: {input_image.device}")

    # 使用 torch.no_grad() 进行推理，以节省内存并加速
    with torch.no_grad():
        output_image = model(input_image)
    
    print(f"输出图片形状: {output_image.shape}, 设备: {output_image.device}")
    print(f"期望的输出形状: torch.Size([1, 3, {h * scale}, {w * scale}])")


def demo(device: str):
    """
    HAT模型的使用小demo，验证其对动态图片尺寸的支持。

    Args:
        device (str): 指定运行模型的设备 ('cpu', 'cuda', 'mps')。
    """
    print(f"将在 {device.upper()} 设备上运行 demo...")

    # 1. 实例化模型
    # 注意：初始化时的 img_size 主要用于计算相对位置编码等，但模型在前向传播时支持动态尺寸。
    # 只要初始化的 img_size 不小于 window_size，模型就能很好地处理各种尺寸的输入。
    window_size = 16
    scale = 2
    model = KawaiiSR(
        image_size = 128,
        in_channels=3,
        use_tiling= False,
        tile_size = 32,
        tile_pad = 16,
        hat_body_hid_channels=180,
        hat_upsampler_hid_channels=64,
        hat_depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        hat_num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        hat_window_size=window_size,
        hat_compress_ratio=3,
        hat_squeeze_factor=30,
        hat_conv_scale=0.01,
        hat_overlap_ratio=0.5,
        hat_mlp_ratio=2.0,
    )

    # 2. 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()  # 切换到评估模式，这会禁用Dropout等层

    print(f"模型已在 {device.upper()} 上实例化。窗口大小: {window_size}, 上采样倍数: {scale}")
    
    # 3. 运行不同尺寸的推理演示
    run_inference_and_print(model, "Demo 1: 标准输入尺寸", (64, 64), scale, device)
    run_inference_and_print(model, "Demo 2: 动态输入尺寸", (128, 128), scale, device)
    run_inference_and_print(model, "Demo 3: 较大的正方形尺寸", (128, 128), scale, device)


if __name__ == '__main__':
    # 自动选择最佳可用设备
    if torch.cuda.is_available():
        selected_device = "cuda"
    elif torch.backends.mps.is_available():  # 适用于 Apple Silicon (M1/M2/M3)
        selected_device = "mps"
    else:
        selected_device = "cpu"
    
    print(f"自动选择的设备: {selected_device.upper()}")

    # 使用选定的设备运行demo
    demo(device=selected_device)