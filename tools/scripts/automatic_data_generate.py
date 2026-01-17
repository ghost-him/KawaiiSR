import os
import random
import string
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# --- 配置参数 ---
OUTPUT_DIR = "anime_sr_dataset_v3"
NUM_IMAGES = 20
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# --- 字体配置 ---
FONT_PATH_CJK = "NotoSansSC-Regular.otf" 

def get_best_font(font_path, size):
    """尝试加载指定字体，失败则返回默认字体"""
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    except:
        pass
    return ImageFont.load_default()

# 尝试探测常见的英文基础字体
def find_eng_font():
    common_paths = [
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

FONT_PATH_ENG = find_eng_font()

# --- 调色板控制 (HSV 空间确保动漫感) ---
def get_anime_color_palette(num_colors=6):
    """
    生成一个动漫感调色板
    动漫色彩通常明度较高，饱和度适中，且具有一定色相偏移
    """
    base_h = random.random()
    palette = []
    for _ in range(num_colors):
        # 让色相在基准色附近波动，确保和谐
        h = (base_h + random.uniform(-0.15, 0.15)) % 1.0
        # 动漫色：饱和度 0.3-0.7，明度 0.6-0.95
        s = random.uniform(0.25, 0.65)
        v = random.uniform(0.65, 0.95)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette.append((int(r*255), int(g*255), int(b*255)))
    return palette

def get_random_cjk_char():
    rand_val = random.random()
    if rand_val < 0.7: return chr(random.randint(0x4E00, 0x9FFF))
    elif rand_val < 0.85: return chr(random.randint(0x3040, 0x309F))
    else: return chr(random.randint(0x30A0, 0x30FF))

def generate_random_string(length, lang):
    if lang == 'eng':
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return ''.join([get_random_cjk_char() for _ in range(length)])

# --- 拟真绘制函数 (RGB) ---

def draw_complex_lines(image, palette):
    """模拟动漫特效线条，具有变粗细和透明度的发光效果"""
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    num_lines = random.randint(10, 30)
    for _ in range(num_lines):
        color = random.choice(palette)
        start_x, start_y = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
        end_x, end_y = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
        
        # 模拟变粗细：分段绘制或使用多次偏移
        base_width = random.randint(2, 12)
        base_alpha = random.randint(100, 200)
        steps = 5
        for i in range(steps):
            # 简单的宽度渐变模拟
            w = max(1, base_width - i * 2)
            alpha = max(10, base_alpha - i * 30)
            offset = i * 0.5
            overlay_draw.line([(start_x+offset, start_y+offset), (end_x+offset, end_y+offset)], 
                              fill=(*color, alpha), width=w)
    image.paste(overlay, (0, 0), overlay)

def draw_realistic_shapes(image, palette):
    """绘制装饰性几何体，支持透明度混合"""
    num_shapes = random.randint(8, 20)
    # 创建一个用于绘制透明形状的层
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    for _ in range(num_shapes):
        color = random.choice(palette)
        x1, y1 = random.randint(0, IMG_WIDTH - 100), random.randint(0, IMG_HEIGHT - 100)
        x2, y2 = random.randint(x1 + 50, IMG_WIDTH), random.randint(y1 + 50, IMG_HEIGHT)
        
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        alpha_v = random.randint(50, 180) 
        fill_color = (*color, alpha_v)
        outline_color = (*random.choice(palette), 200)
        
        if shape_type == 'rectangle':
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=random.randint(1, 5))
        elif shape_type == 'ellipse':
            overlay_draw.ellipse([x1, y1, x2, y2], fill=fill_color)
        elif shape_type == 'polygon':
            points = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(random.randint(3, 6))]
            overlay_draw.polygon(points, fill=fill_color)
    
    # 复合到原图上
    image.paste(overlay, (0, 0), overlay)

def generate_complex_background(palette):
    """生成带有细腻渐变和图案的基础背景"""
    # 使用 numpy 优化基础渐变生成
    c1 = np.array(palette[0], dtype=np.float32)
    c2 = np.array(palette[1], dtype=np.float32)
    
    # 正确的广播：生成 (IMG_HEIGHT, 1, 3) 然后沿列平铺
    ratios = np.linspace(0, 1, IMG_HEIGHT).reshape(IMG_HEIGHT, 1, 1)
    grad_2d = (1 - ratios) * c1 + ratios * c2
    grad_2d = np.tile(grad_2d, (1, IMG_WIDTH, 1)).astype(np.uint8)
    base_img = Image.fromarray(grad_2d, mode='RGB')
    
    # 叠加简单的网点 (Halftone 模拟) - 优化：使用平铺提升速度
    pattern_color = palette[2]
    dot_size = random.randint(2, 5)
    spacing = random.randint(10, 20)
    
    tile = Image.new('RGBA', (spacing, spacing), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.ellipse([0, 0, dot_size, dot_size], fill=pattern_color)
    
    pattern_overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    for x in range(0, IMG_WIDTH, spacing):
        for y in range(0, IMG_HEIGHT, spacing):
            pattern_overlay.paste(tile, (x, y))
            
    base_img.paste(pattern_overlay, (0, 0), pattern_overlay)
    return base_img
def draw_managed_text(image, palette):
    """
    改进的文字排版：
    1. 采用网格化布局（Grid Layout），按顺序填充槽位，彻底避免随机重叠。
    2. 模拟动漫菜单、属性栏或系统 UI 的有序感。
    """
    # 动态计算网格数量，例如 4行 x 3列 = 12 个槽位
    rows, cols = 4, 3
    slot_w = IMG_WIDTH // cols
    slot_h = IMG_HEIGHT // rows
    
    # 生成所有槽位的起始坐标
    slots = []
    for r in range(rows):
        for c in range(cols):
            slots.append((c * slot_w, r * slot_h))
            
    # 文字数量不超过槽位总数
    num_texts = random.randint(6, len(slots))
    # 虽是顺序排布，但我们可以随机打乱槽位顺序，或者直接按顺序填
    # 这里我们随机抽取槽位，但每个槽位只放一个文字，保证不重叠
    selected_slots = random.sample(slots, num_texts)

    for i in range(num_texts):
        lang = random.choice(['eng', 'cjk', 'cjk'])
        font_path = FONT_PATH_CJK if lang == 'cjk' else FONT_PATH_ENG
        
        # 适当减小字号以适配网格单元
        font_size = random.randint(40, 90)
        font = get_best_font(font_path, font_size)

        text = generate_random_string(random.randint(3, 8), lang)
        
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        
        # 获取当前槽位的左上角坐标
        base_x, base_y = selected_slots[i]
        
        # 在槽位内部进行居中排版，并加入少许随机偏移（让效果不至于太死板）
        x = base_x + (slot_w - tw) // 2 + random.randint(-15, 15)
        y = base_y + (slot_h - th) // 2 + random.randint(-15, 15)
        
        # 边界二次检查
        x = max(5, min(x, IMG_WIDTH - tw - 5))
        y = max(5, min(y, IMG_HEIGHT - th - 5))
        
        angle = random.uniform(-10, 10) # 减小旋转角度
        color = random.choice(palette)
        stroke_color = random.choice(palette) if random.random() > 0.5 else (255, 255, 255)
        
        # 绘制逻辑
        txt_img = Image.new('RGBA', (tw + 40, th + 40), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((20, 20), text, font=font, fill=color, 
                      stroke_width=random.randint(2, 6), stroke_fill=stroke_color)
        
        rotated = txt_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        image.paste(rotated, (int(x), int(y)), rotated)

    return image

# --- 主生成流程 ---

def generate_single_hq_image():
    """
    保留一组最复杂、最拟真的组合。
    流程：复杂背景 -> 几何形状 -> 装饰线 -> 文字UI -> 轻微整体滤镜
    """
    palette = get_anime_color_palette(num_colors=8)
    
    # 1. 生成复杂背景
    img = generate_complex_background(palette)
    
    # 2. 绘制装饰性几何体
    draw_realistic_shapes(img, palette)
    
    # 3. 绘制装饰线条
    draw_complex_lines(img, palette)
    
    # 4. 绘制文字层
    img = draw_managed_text(img, palette)
    
    # 5. 轻微的后期处理 (模拟动漫的模糊处理)
    if random.random() > 0.8:
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
    return img.convert('RGB')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"文件夹 '{OUTPUT_DIR}' 已创建。")

    print(f"开始生成 {NUM_IMAGES} 张高拟真 RGB 图像...")
    for i in tqdm(range(NUM_IMAGES)):
        final_image = generate_single_hq_image()
        filename = f"anime_hq_{i:04d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        final_image.save(filepath, 'PNG')

    print(f"\n成功完成，图像保存在 '{OUTPUT_DIR}'。格式为 RGB，已应用动漫色彩空间与复杂排版。")

if __name__ == '__main__':
    main()