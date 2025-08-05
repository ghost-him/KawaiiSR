import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- 配置参数 ---
OUTPUT_DIR = "anime_sr_dataset_v3"  # 保存图片的文件夹
NUM_IMAGES = 100                   # 希望生成的图片总数
IMG_WIDTH = 1024                   # 图片宽度
IMG_HEIGHT = 1024                  # 图片高度

# --- 字体配置 (重要！请根据你的实际情况修改) ---
# 请下载并解压 Google Noto Sans CJK SC 字体，然后将 .otf 文件的路径填入下方
# Noto字体能很好地支持中日韩字符，对于随机字符生成至关重要
# 如果脚本和字体文件在同一目录，可以直接写文件名 "NotoSansSC-Regular.otf"
FONT_PATH_CJK = "NotoSansSC-Regular.otf" 
# 对于英文，我们可以尝试使用一个常见的系统字体，或者也用Noto
FONT_PATH_ENG = "arial.ttf" # 在Windows/macOS上通常可用

# --- 加载字体 ---
try:
    # 加载一个占位字体，实际字体大小在绘制时动态设置
    font_cjk_loader = ImageFont.truetype(FONT_PATH_CJK, size=50)
    print(f"成功加载中日文字体: {FONT_PATH_CJK}")
except IOError:
    print(f"警告: 无法在中日文字体路径 '{FONT_PATH_CJK}' 找到字体文件。文字生成功能将受限。")
    font_cjk_loader = None

try:
    font_eng_loader = ImageFont.truetype(FONT_PATH_ENG, size=50)
    print(f"成功加载英文字体: {FONT_PATH_ENG}")
except IOError:
    print(f"警告: 无法在英文字体路径 '{FONT_PATH_ENG}' 找到字体文件。将使用默认字体。")
    font_eng_loader = ImageFont.load_default()

# --- 随机字符/字符串生成函数 ---

def get_random_cjk_char():
    """随机生成一个中日韩字符 (汉字, 平假名, 片假名)"""
    rand_val = random.random()
    if rand_val < 0.7: # 70% 概率为常用汉字
        return chr(random.randint(0x4E00, 0x9FFF))
    elif rand_val < 0.85: # 15% 概率为平假名
        return chr(random.randint(0x3040, 0x309F))
    else: # 15% 概率为片假名
        return chr(random.randint(0x30A0, 0x30FF))

def generate_random_string(length, lang):
    """根据语言和长度生成随机字符串"""
    if lang == 'eng':
        # 英文：包含大小写字母和数字
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))
    elif lang == 'cjk':
        # 中日韩：调用字符生成函数
        return ''.join([get_random_cjk_char() for _ in range(length)])
    return ""

# --- 辅助函数 (与上一版相同) ---

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(150, 255))

def get_random_background():
    if random.random() < 0.2: return (0, 0, 0, 0)
    else: return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255), 255)

def create_base_image():
    return Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), get_random_background())

def draw_lines(draw):
    num_lines = random.randint(5, 50)
    for _ in range(num_lines):
        start_x, start_y = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
        end_x, end_y = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
        width = random.randint(1, 15); color = get_random_color()
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
    num_curves = random.randint(2, 10)
    for _ in range(num_curves):
        points = [(random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT))]
        for i in range(1, random.randint(10, 50)):
            points.append((points[-1][0] + random.randint(-40, 40), points[-1][1] + random.randint(-40, 40)))
        width = random.randint(2, 20); color = get_random_color()
        draw.line(points, fill=color, width=width, joint='curve')

def draw_shapes(draw):
    num_shapes = random.randint(5, 30)
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        x1, y1 = random.randint(0, IMG_WIDTH - 50), random.randint(0, IMG_HEIGHT - 50)
        x2, y2 = random.randint(x1 + 50, IMG_WIDTH), random.randint(y1 + 50, IMG_HEIGHT)
        fill_color = get_random_color() if random.random() > 0.3 else None
        outline_color = get_random_color(); outline_width = random.randint(1, 15)
        if shape_type == 'rectangle': draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=outline_width)
        elif shape_type == 'ellipse': draw.ellipse([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=outline_width)
        elif shape_type == 'polygon':
            points = [(random.randint(x1, x2), random.randint(y1, y2)) for _ in range(random.randint(3, 8))]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)

def generate_gradient_image():
    image = create_base_image()
    gradient_type = random.choice(['linear', 'radial'])
    start_color, end_color = np.array(get_random_color()), np.array(get_random_color())
    if gradient_type == 'linear':
        axis = random.choice([0, 1])
        ramp = np.linspace(0, 1, IMG_HEIGHT if axis == 0 else IMG_WIDTH)
        ramp = np.repeat(ramp[:, np.newaxis], IMG_WIDTH, axis=1) if axis == 0 else np.repeat(ramp[np.newaxis, :], IMG_HEIGHT, axis=0)
    else:
        center_x, center_y = random.uniform(0, IMG_WIDTH), random.uniform(0, IMG_HEIGHT)
        x, y = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        ramp = np.clip(dist / np.max(dist), 0, 1)
    gradient_array = (start_color * (1 - ramp[..., np.newaxis]) + end_color * ramp[..., np.newaxis]).astype(np.uint8)
    gradient_image = Image.fromarray(gradient_array, 'RGBA')
    return Image.alpha_composite(image, gradient_image)

def generate_pattern_image():
    image = create_base_image(); draw = ImageDraw.Draw(image)
    pattern_type = random.choice(['dots', 'checkerboard', 'stripes']); color = get_random_color()
    if pattern_type == 'dots':
        size = random.randint(8, 40); radius = size // 4
        for x in range(0, IMG_WIDTH, size):
            for y in range(0, IMG_HEIGHT, size): draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    elif pattern_type == 'checkerboard':
        size = random.randint(20, 100)
        for x in range(0, IMG_WIDTH, size):
            for y in range(0, IMG_HEIGHT, size):
                if (x // size + y // size) % 2 == 0: draw.rectangle([x, y, x + size, y + size], fill=color, width=0)
    elif pattern_type == 'stripes':
        size = random.randint(10, 60)
        horizontal = random.random() > 0.5
        for i in range(0, IMG_HEIGHT if horizontal else IMG_WIDTH, size*2):
            if horizontal: draw.rectangle([0, i, IMG_WIDTH, i+size], fill=color, width=0)
            else: draw.rectangle([i, 0, i+size, IMG_HEIGHT], fill=color, width=0)
    return image
    
def draw_text(base_image):
    num_texts = random.randint(2, 12)
    for _ in range(num_texts):
        lang = random.choice(['eng', 'cjk', 'cjk']) # 增加CJK的权重
        if lang == 'eng' and font_eng_loader:
            font_path = FONT_PATH_ENG
        elif lang == 'cjk' and font_cjk_loader:
            font_path = FONT_PATH_CJK
        else:
            continue # 如果所需字体不可用，则跳过本次绘制

        str_len = random.randint(1, 8)
        text = generate_random_string(str_len, lang)
        
        font_size = random.randint(40, 220)
        font = ImageFont.truetype(font_path, font_size)

        fill_color = get_random_color()
        x, y = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
        angle = random.uniform(-45, 45)
        stroke_width = random.randint(0, int(font_size * 0.1)) if random.random() > 0.4 else 0
        stroke_fill = get_random_color() if stroke_width > 0 else None

        text_bbox = font.getbbox(text, stroke_width=stroke_width)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        txt_img = Image.new('RGBA', (text_width, text_height), (0,0,0,0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_fill)

        rotated_txt = txt_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        paste_x, paste_y = x - rotated_txt.width // 2, y - rotated_txt.height // 2
        base_image.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

    return base_image

# --- 主逻辑 (与上一版类似) ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"文件夹 '{OUTPUT_DIR}' 已创建。")

    # 定义生成器函数列表
    generators = {
        'lines': (lambda img: (draw_lines(ImageDraw.Draw(img)), img)[1], create_base_image),
        'shapes': (lambda img: (draw_shapes(ImageDraw.Draw(img)), img)[1], create_base_image),
        'patterns': (generate_pattern_image, None),
        'gradients': (generate_gradient_image, None),
        'text': (draw_text, create_base_image),
        'lines_and_shapes': (lambda img: (draw_lines(ImageDraw.Draw(img)), draw_shapes(ImageDraw.Draw(img)), img)[2], create_base_image),
        'text_on_gradient': (draw_text, generate_gradient_image),
        'lines_on_pattern': (lambda img: (draw_lines(ImageDraw.Draw(img)), img)[1], generate_pattern_image),
        'text_and_shapes': (lambda img: (draw_text(img), draw_shapes(ImageDraw.Draw(img)), img)[2], create_base_image)
    }
    generator_keys = list(generators.keys())

    print(f"开始生成 {NUM_IMAGES} 张图片...")
    for i in tqdm(range(NUM_IMAGES)):
        gen_key = random.choice(generator_keys)
        draw_func, base_img_func = generators[gen_key]

        if base_img_func:
            base_img = base_img_func()
            final_image = draw_func(base_img)
        else:
            final_image = draw_func()

        filename = f"{gen_key}_{i:04d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        final_image.save(filepath, 'PNG')

    print(f"\n成功生成 {NUM_IMAGES} 张图片，保存在 '{OUTPUT_DIR}' 文件夹中。")
    print("数据集现已包含随机生成的各类字符。")

if __name__ == '__main__':
    main()