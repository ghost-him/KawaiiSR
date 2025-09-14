import torch
from KawaiiSR import KawaiiSR

def demo():
    """
    HAT模型的使用小demo，验证其对动态图片尺寸的支持。
    """

    # 1. 实例化模型
    # 注意：初始化时的 img_size 主要用于计算相对位置编码等，但模型在前向传播时支持动态尺寸。
    # 只要初始化的 img_size 不小于 window_size，模型就能很好地处理各种尺寸的输入。
    window_size=16
    scale=2
    model = KawaiiSR(
        image_size=128,
        use_tiling= True,
        tile_size = 64, # 128 - 2 * tile_pad = 64
        tile_pad= 32,
        hat_window_size=window_size,
        hat_depths=(6, 6, 6, 6, 6, 6),     # 使用较浅的深度以加速
        hat_num_heads=(6, 6, 6, 6, 6, 6),    # 使用较少的头
        hat_hid_channels=180,            # 使用较小的维度
        hat_mlp_ratio=2.0,
        hat_compress_ratio=3,
        hat_squeeze_factor=30,
        hat_conv_scale=0.01,
        hat_overlap_ratio=0.5,
    )
    model.eval()  # 切换到评估模式，这会禁用Dropout等层

    print(f"模型已在CPU上实例化。窗口大小: {window_size}, 上采样倍数: {scale}")
    
    # 3. Demo 1: 使用一个标准的输入尺寸
    print("\n--- Demo 1: 标准输入尺寸 (64x64) ---")
    h1, w1 = 64, 64
    

    # 创建一个随机的输入图片张量 (Batch, Channels, Height, Width)
    input_image1 = torch.rand(1, 3, h1, w1)
    print(f"输入图片 1 的形状: {input_image1.shape}")

    # 使用 torch.no_grad() 进行推理，以节省内存并加速
    with torch.no_grad():
        output_image1 = model(input_image1)
    
    print(f"输出图片 1 的形状: {output_image1.shape}")
    print(f"期望的输出形状: torch.Size([1, 3, {h1 * scale}, {w1 * scale}])")

    # 4. Demo 2: 验证动态尺寸支持（使用一个不同的、非正方形的尺寸）
    print("\n--- Demo 2: 动态输入尺寸 (17x73) ---")
    h2, w2 = 17, 73
    
    input_image2 = torch.rand(1, 3, h2, w2)
    print(f"输入图片 2 的形状: {input_image2.shape}")

    with torch.no_grad():
        output_image2 = model(input_image2)
        
    print(f"输出图片 2 的形状: {output_image2.shape}")
    print(f"期望的输出形状: torch.Size([1, 3, {h2 * scale}, {w2 * scale}])")

    # 4. Demo 3: 验证超高分辨率支持
    print("\n--- Demo 3: 超高分辨率尺寸 (1280x720) ---")
    h3, w3 = 1280, 720
    

    input_image2 = torch.rand(1, 3, h3, w3)
    print(f"输入图片 3 的形状: {input_image2.shape}")

    with torch.no_grad():
        output_image2 = model(input_image2)
        
    print(f"输出图片 3 的形状: {output_image2.shape}")
    print(f"期望的输出形状: torch.Size([1, 3, {h3 * scale}, {w3 * scale}])")


if __name__ == '__main__':
    demo()