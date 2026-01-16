import os
from PIL import Image
import subprocess
import platform
import sys
import re

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize_image(input_path, output_path, width, height):
    """将图像调整为指定尺寸"""
    img = Image.open(input_path)
    # 使用 LANCZOS 进行高质量缩放
    img = img.resize((width, height), Image.LANCZOS)
    img.save(output_path, dpi=(72, 72))
    print(f"已创建: {output_path} (72 DPI)")

def create_retina_image(png_path, output_path, scale=2):
    """创建 Retina (@2x) 版本的图像"""
    img = Image.open(png_path)
    width, height = img.size
    retina_img = img.resize((width * scale, height * scale), Image.LANCZOS)
    retina_img.save(output_path, dpi=(72, 72))
    print(f"已创建: {output_path} (72 DPI)")

def create_ico(png_path, output_path):
    """创建 .ico 文件"""
    # 为每个尺寸创建 72 DPI 的临时 PNG 文件
    temp_files = []
    sizes = [ (128, 128)]

    img = Image.open(png_path)
    for size in sizes:
        temp_file = f"temp_{size[0]}x{size[1]}.png"
        temp_img = img.resize(size, Image.LANCZOS)
        temp_img.save(temp_file, dpi=(72, 72))
        temp_files.append(temp_file)

    # 使用第一个临时文件作为基础，添加其他尺寸
    base_img = Image.open(temp_files[0])
    other_imgs = [Image.open(f) for f in temp_files[1:]]

    # 保存为 ICO
    base_img.save(output_path, format='ICO', append_images=other_imgs, 
                 sizes=sizes)

    # 清理临时文件
    for file in temp_files:
        os.remove(file)

    print(f"已创建: {output_path}")

def create_icns(input_path, output_path):
    """创建 .icns 文件 (仅在 macOS 上使用 iconutil)"""
    # 创建临时目录
    temp_iconset = "icon.iconset"
    ensure_directory(temp_iconset)

    # 创建不同尺寸的图标
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for size in sizes:
        # 标准尺寸
        standard_output = os.path.join(temp_iconset, f"icon_{size}x{size}.png")
        resize_image(input_path, standard_output, size, size)

        # Retina 尺寸
        if size < 512:  # 1024 不需要 @2x 版本
            retina_output = os.path.join(temp_iconset, f"icon_{size}x{size}@2x.png")
            resize_image(input_path, retina_output, size * 2, size * 2)

    # 使用 iconutil 创建 .icns 文件 (仅在 macOS 上)
    if platform.system() == 'Darwin':
        subprocess.run(['iconutil', '-c', 'icns', temp_iconset, '-o', output_path])
        print(f"已创建: {output_path}")
    else:
        print("警告: 创建 .icns 文件需要 macOS 系统和 iconutil 工具")

    # 清理临时目录
    for file in os.listdir(temp_iconset):
        os.remove(os.path.join(temp_iconset, file))
    os.rmdir(temp_iconset)

def create_white_icon(input_path, output_path, width, height):
    """创建并转换 PNG 为白色版本的图标（将非透明像素设为白色）"""
    img = Image.open(input_path).convert("RGBA")
    img = img.resize((width, height), Image.LANCZOS)
    
    # 创建白色版本
    data = img.getdata()
    new_data = []
    for item in data:
        # item 是 (R, G, B, A)
        if item[3] > 0:
            new_data.append((255, 255, 255, item[3]))
        else:
            new_data.append(item)
    img.putdata(new_data)
    
    img.save(output_path, dpi=(72, 72))
    print(f"已创建白色图标: {output_path} (72 DPI)")

def main():
    # 输入 PNG 文件
    input_file = "icon.png"

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)

    # 创建输出目录
    output_dir = "output"
    ensure_directory(output_dir)

    # 创建指定尺寸的 PNG 图标
    sizes = {
        "32x32.png": (32, 32),
        "128x128.png": (128, 128),
        "Square30x30Logo.png": (30, 30),
        "Square44x44Logo.png": (44, 44),
        "Square71x71Logo.png": (71, 71),
        "Square89x89Logo.png": (89, 89),
        "Square107x107Logo.png": (107, 107),
        "Square142x142Logo.png": (142, 142),
        "Square150x150Logo.png": (150, 150),
        "Square284x284Logo.png": (284, 284),
        "Square310x310Logo.png": (310, 310),
        "StoreLogo.png": (50, 50)
    }

    for filename, (width, height) in sizes.items():
        output_path = os.path.join(output_dir, filename)
        resize_image(input_file, output_path, width, height)

    # 创建 Retina (@2x) 版本
    retina_path = os.path.join(output_dir, "128x128@2x.png")
    resize_image(input_file, retina_path, 256, 256)

    # 创建 .ico 文件
    ico_path = os.path.join(output_dir, "icon.ico")
    create_ico(input_file, ico_path)

    # 创建 .icns 文件 (仅在 macOS 上)
    icns_path = os.path.join(output_dir, "icon.icns")
    create_icns(input_file, icns_path)

    white_icon_path = os.path.join(output_dir, "32x32-white.png")
    create_white_icon(input_file, white_icon_path, 32, 32)

    print("所有图标已成功创建，分辨率均为 72 DPI！")

if __name__ == "__main__":
    main()