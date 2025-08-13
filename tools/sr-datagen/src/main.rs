use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba, RgbaImage, imageops::FilterType};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;
pub mod strong_gt;
// 使用 lazy_static 为 resvg 设置一个全局的、一次性的字体数据库
lazy_static! {
    static ref FONT_DB: usvg::fontdb::Database = {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        db
    };
}

/// 用于创建超分辨率数据集的强大命令行工具
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// 输入图像所在的文件夹路径
    #[arg(short, long)]
    input_dir: PathBuf,

    /// 生成数据集的目标文件夹路径
    #[arg(short, long)]
    output_dir: PathBuf,

    /// 用于存放高分辨率 (HR) 图像的子文件夹名称
    #[arg(long, default_value = "HR")]
    hr_dir_name: String,

    #[arg(long, default_value = "Strong_HR")]
    strong_hr_dir_name: String,

    /// 用于存放低分辨率 (LR) 图像的子文件夹名称
    #[arg(long, default_value = "LR")]
    lr_dir_name: String,

    /// 超分辨率的缩放比例 (例如: 2, 3, 4)
    #[arg(short, long, default_value_t = 2)]
    scale: u32,

    /// [仅SVG] 以逗号分隔的渲染尺寸列表 (例如: "128,256,512")
    #[arg(long, default_value = "128,256,512,1024")]
    svg_sizes: String,

    /// [仅位图] 以逗号分隔的下采样算法列表
    /// 可选项: Nearest, Triangle, CatmullRom, Gaussian, Lanczos3
    #[arg(long, default_value = "Nearest,Lanczos3")]
    downsample_filters: String,

    /// [仅位图] 将尺寸过大的图像裁剪为指定大小的图块 (例如: 1024)
    #[arg(long)]
    patch_size: Option<u32>,

    /// [仅位图] 是否执行数据增强 (水平翻转和旋转90度)
    #[arg(long)]
    augment: bool,

    /// [仅位图] 对下采样后的LR图像应用JPEG压缩，并指定质量 (1-100)
    #[arg(long)]
    jpeg_quality: Option<String>,

    /// [仅位图] 对下采样后的LR图像应用WebP压缩，并指定质量 (1-100)
    #[arg(long)]
    webp_quality: Option<String>,

    /// 跳过确认提示，直接开始处理
    #[arg(short, long)]
    yes: bool,

    /// 要不要使用锐化来增强图片
    #[arg(long)]
    strong_hr: bool,
}

/// 解析后的配置结构体
#[derive(Debug, Clone)]
struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    hr_path: PathBuf,
    strong_hr_path: PathBuf,
    lr_path: PathBuf,
    scale: u32,
    svg_sizes: Vec<u32>,
    downsample_filters: Vec<(String, FilterType)>,
    patch_size: Option<u32>,
    augment: bool,
    jpeg_quality: Vec<u8>,
    webp_quality: Vec<u8>,
    strong_hr: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 解析和验证参数
    let svg_sizes = cli
        .svg_sizes
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect::<Vec<_>>();

    let downsample_filters = parse_filter_types(&cli.downsample_filters)?;

    let jpeg_quality = cli.jpeg_quality.map_or_else(Vec::new, |s| {
        s.split(',')
            .filter_map(|q| q.trim().parse::<u8>().ok())
            .filter(|&q| q > 0 && q <= 100) // 确保质量在有效范围内
            .collect()
    });

    let webp_quality = cli.webp_quality.map_or_else(Vec::new, |s| {
        s.split(',')
            .filter_map(|q| q.trim().parse::<u8>().ok())
            .filter(|&q| q > 0 && q <= 100) // 确保质量在有效范围内
            .collect()
    });

    let hr_path = cli.output_dir.join(&cli.hr_dir_name);
    let lr_path = cli.output_dir.join(&cli.lr_dir_name);
    let strong_hr_path = cli.output_dir.join(&cli.strong_hr_dir_name);

    let config = Config {
        input_dir: cli.input_dir,
        output_dir: cli.output_dir,
        hr_path,
        strong_hr_path: strong_hr_path,
        lr_path,
        scale: cli.scale,
        svg_sizes,
        downsample_filters,
        patch_size: cli.patch_size,
        augment: cli.augment,
        jpeg_quality,
        webp_quality,
        strong_hr: cli.strong_hr,
    };

    // 打印配置并请求用户确认
    println!("--- 数据集生成配置 ---");
    println!("输入文件夹: {}", config.input_dir.display());
    println!("输出文件夹: {}", config.output_dir.display());
    println!("  - HR 文件夹: {}", config.hr_path.display());
    println!("  - Strong_HR 文件夹: {}", config.strong_hr_path.display());
    println!("  - LR 文件夹: {}", config.lr_path.display());
    println!("缩放比例: x{}", config.scale);
    println!("\n--- 矢量图 (SVG) 配置 ---");
    println!("渲染尺寸: {:?}", config.svg_sizes);
    println!("\n--- 位图 (PNG/JPG等) 配置 ---");
    if let Some(size) = config.patch_size {
        println!("大图像裁剪尺寸: {}x{}", size, size);
    } else {
        println!("大图像裁剪: 否");
    }
    println!(
        "下采样算法: {:?}",
        config
            .downsample_filters
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    );
    println!(
        "数据增强 (翻转/旋转): {}",
        if config.augment { "是" } else { "否" }
    );
    if !config.jpeg_quality.is_empty() {
        println!("JPEG 压缩质量 (对LR图像): {:?}", config.jpeg_quality);
    } else {
        println!("JPEG 压缩 (对LR图像): 否");
    }
    if !config.webp_quality.is_empty() {
        println!("WebP 压缩质量 (对LR图像): {:?}", config.webp_quality);
    } else {
        println!("WebP 压缩 (对LR图像): 否");
    }
    if !config.jpeg_quality.is_empty() || !config.webp_quality.is_empty() {
        println!("注意: 未压缩的LR版本也将被保存。");
    }
    println!(
        "锐化HR图像: {}",
        if config.strong_hr { "是" } else { "否" }
    );
    println!("-------------------------\n");

    if !cli.yes {
        print!("是否继续? (y/N): ");
        io::stdout().flush()?;
        let mut response = String::new();
        io::stdin().read_line(&mut response)?;
        if response.trim().to_lowercase() != "y" {
            println!("操作已取消。");
            return Ok(());
        }
    }

    run_processing(&config)?;

    Ok(())
}
/// 主处理函数
fn run_processing(config: &Config) -> Result<()> {
    println!("\n[1/5] 开始设置输出环境...");
    fs::create_dir_all(&config.hr_path).context("无法创建HR文件夹")?;
    // 只有在需要时才创建 Strong_HR 文件夹，更节省资源
    if config.strong_hr {
        fs::create_dir_all(&config.strong_hr_path).context("无法创建Strong_HR文件夹")?;
    }
    fs::create_dir_all(&config.lr_path).context("无法创建LR文件夹")?;

    let index_file_path = config.output_dir.join("dataset_index.csv");
    let writer = Writer::from_path(&index_file_path).context("无法创建CSV索引文件")?;
    let csv_writer = Arc::new(Mutex::new(writer));

    // 写入CSV表头
    {
        let mut guard = csv_writer.lock();
        guard
            .write_record(&["lr_image_path", "hr_image_path"])
            .context("无法写入CSV表头")?;
    }

    println!("[2/5] 正在扫描输入文件...");
    let paths: Vec<PathBuf> = WalkDir::new(&config.input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .collect();

    let total_files = paths.len();
    if total_files == 0 {
        println!("警告：输入文件夹中未找到任何文件。");
        return Ok(());
    }
    println!("找到 {} 个文件，准备处理。", total_files);

    let processed_count = AtomicUsize::new(0);
    let log_interval = (total_files / 100).max(1); // 每处理1%的文件或至少1个文件时打印日志

    println!("[3/5] 开始并行处理图像...");

    paths.into_par_iter().for_each(|path| {
        let current_count = processed_count.fetch_add(1, Ordering::SeqCst);
        if current_count % log_interval == 0 {
            println!(
                "进度: {}/{} ({:.2}%)",
                current_count,
                total_files,
                (current_count as f32 / total_files as f32) * 100.0
            );
        }

        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        let result = match extension.to_lowercase().as_str() {
            "svg" => process_svg(&path, config),
            "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "webp" => process_bitmap(&path, config),
            _ => {
                // eprintln!("警告: 跳过不支持的文件类型: {}", path.display());
                Ok(Vec::new()) // 静默跳过
            }
        };

        match result {
            Ok(pairs) => {
                if !pairs.is_empty() {
                    let mut guard = csv_writer.lock();
                    for (lr, hr) in pairs {
                        if let Err(e) =
                            guard.write_record(&[lr.to_str().unwrap(), hr.to_str().unwrap()])
                        {
                            eprintln!("错误: 无法将记录写入CSV: {}", e);
                        }
                    }
                }
            }
            Err(e) => eprintln!("错误: 处理文件 {} 失败: {}", path.display(), e),
        }
    });

    println!("[4/5] 开始生成锐化文件...");
    if config.strong_hr {
        println!("  -> 锐化功能已启用。正在处理 HR 图像...");

        // 收集所有已生成的HR文件
        let hr_files: Vec<PathBuf> = WalkDir::new(&config.hr_path)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();

        let total_hr_files = hr_files.len();
        if total_hr_files == 0 {
            println!("  -> HR 文件夹中未找到任何文件进行锐化。");
        } else {
            println!("  -> 找到 {} 个 HR 图像进行锐化。", total_hr_files);
            let sharpened_count = AtomicUsize::new(0);
            let log_interval = (total_hr_files / 50).max(1); // 锐化进度条

            hr_files.into_par_iter().for_each(|hr_path| {
                let img = match image::open(&hr_path) {
                    Ok(img) => img,
                    Err(e) => {
                        eprintln!("警告: 无法打开HR图像 {} 进行锐化: {}", hr_path.display(), e);
                        return;
                    }
                };
                
                // 调用锐化函数，使用推荐的默认参数
                let sharpened_img = strong_gt::anime_sharpen(&img, 2, 32);

                if let Some(filename) = hr_path.file_name() {
                    let strong_hr_filepath = config.strong_hr_path.join(filename);
                    if let Err(e) = sharpened_img.save(&strong_hr_filepath) {
                        eprintln!("错误: 无法保存锐化后的图像 {}: {}", strong_hr_filepath.display(), e);
                    }
                }

                let count = sharpened_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count % log_interval == 0 || count == total_hr_files {
                     println!(
                        "  -> 锐化进度: {}/{} ({:.2}%)",
                        count,
                        total_hr_files,
                        (count as f32 / total_hr_files as f32) * 100.0
                    );
                }
            });
             println!("  -> 锐化处理完成。");
        }
    } else {
        println!("  -> 锐化功能已跳过。");
    }


    println!("[5/5] 处理完成！");
    println!("数据集已生成在: {}", config.output_dir.display());
    println!("索引文件: {}", index_file_path.display());

    Ok(())
}

/// 处理SVG矢量图
fn process_svg(path: &Path, config: &Config) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();
    let svg_data = fs::read(path)?;
    let tree = usvg::Tree::from_data(&svg_data, &usvg::Options::default())?;

    let stem = path.file_stem().unwrap().to_str().unwrap();

    for &hr_size in &config.svg_sizes {
        let lr_size = hr_size / config.scale;
        if lr_size == 0 {
            continue;
        }

        let hr_filename = format!("{}_{}_hr.png", stem, hr_size);
        let lr_filename = format!("{}_{}_x{}_lr.png", stem, hr_size, config.scale);

        let hr_filepath = config.hr_path.join(&hr_filename);
        let lr_filepath = config.lr_path.join(&lr_filename);

        // 渲染HR图像
        let hr_img = render_svg_to_rgba(&tree, hr_size)?;
        hr_img.save_with_format(&hr_filepath, ImageFormat::Png)?;

        // 渲染LR图像
        let lr_img = render_svg_to_rgba(&tree, lr_size)?;
        lr_img.save_with_format(&lr_filepath, ImageFormat::Png)?;

        pairs.push((lr_filename.into(), hr_filename.into()));
    }
    Ok(pairs)
}

/// 将 usvg Tree 渲染为 RgbaImage
fn render_svg_to_rgba(tree: &usvg::Tree, size: u32) -> Result<RgbaImage> {
    let mut pixmap = tiny_skia::Pixmap::new(size, size).context("无法创建Pixmap")?;
    pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));
    let transform = tiny_skia::Transform::from_scale(
        size as f32 / tree.size().width(),
        size as f32 / tree.size().height(),
    );
    resvg::render(tree, transform, &mut pixmap.as_mut());
    let img = RgbaImage::from_raw(size, size, pixmap.data().to_vec())
        .context("无法从Pixmap数据创建图像")?;
    Ok(img)
}
/// 处理位图
fn process_bitmap(path: &Path, config: &Config) -> Result<Vec<(PathBuf, PathBuf)>> {
    let img = image::open(path).with_context(|| format!("无法打开图像 {}", path.display()))?;
    let stem = path.file_stem().unwrap().to_str().unwrap();

    let mut patches = Vec::new();

    // 如果设置了 patch_size 并且图片尺寸超过了它，则进行裁剪
    if let Some(patch_size) = config.patch_size {
        if img.width() > patch_size || img.height() > patch_size {
            patches.extend(crop_image_with_overlap(&img, patch_size));
        } else {
            patches.push(img);
        }
    } else {
        patches.push(img);
    }

    let mut all_pairs = Vec::new();
    let is_multi_patch = patches.len() > 1;

    for (i, patch) in patches.into_iter().enumerate() {
        // 1. 首先确保图块是 RGBA 格式
        let rgba_patch = ensure_rgba(&patch);

        // 2. 然后在 RgbaImage 上进行裁剪
        let (w, h) = rgba_patch.dimensions();
        // 确保图像尺寸为偶数，这是许多模型的要求
        let new_w = w - (w % 2);
        let new_h = h - (h % 2);

        if new_w == 0 || new_h == 0 {
            continue;
        }

        // crop_imm 返回 SubImage, .to_image() 将其转换为新的 RgbaImage
        let base_hr_image = image::imageops::crop_imm(&rgba_patch, 0, 0, new_w, new_h).to_image();
        
        // 为图块创建唯一的名称
        let patch_stem = if is_multi_patch {
            format!("{}_p{}", stem, i)
        } else {
            stem.to_string()
        };

        // 处理单个图块（或完整图像）
        match process_single_patch(&base_hr_image, &patch_stem, config) {
            Ok(pairs) => all_pairs.extend(pairs),
            Err(e) => eprintln!("错误: 处理图块 {} from {} 失败: {}", i, path.display(), e),
        }
    }

    Ok(all_pairs)
}

/// 将图像裁剪成图块。
/// 如果图像维度不能被 patch_size 整除，最后一个图块将从图像的末端开始，以确保覆盖边缘。
fn crop_image_with_overlap(img: &DynamicImage, patch_size: u32) -> Vec<DynamicImage> {
    let (width, height) = img.dimensions();
    let mut patches = Vec::new();

    let mut y = 0;
    while y < height {
        let actual_y = if y + patch_size > height { height.saturating_sub(patch_size) } else { y };
        let mut x = 0;
        while x < width {
            let actual_x = if x + patch_size > width {
                width.saturating_sub(patch_size)
            } else {
                x
            };

            patches.push(img.crop_imm(actual_x, actual_y, patch_size, patch_size));

            if x >= actual_x && x + patch_size >= width {
                break;
            }
            x += patch_size;
        }
        if y >= actual_y && y + patch_size >= height {
            break;
        }
        y += patch_size;
    }
    patches
}

/// 处理单个图像（或图块），应用下采样、增强和压缩。
fn process_single_patch(
    base_hr_image: &RgbaImage,
    stem: &str,
    config: &Config,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();

    // --- 1. 处理原始（基础）图像 ---
    let base_hr_filename = format!("{}_hr.png", stem);
    let base_hr_filepath = config.hr_path.join(&base_hr_filename);
    base_hr_image.save_with_format(&base_hr_filepath, ImageFormat::Png)?;

    let (hr_w, hr_h) = base_hr_image.dimensions();
    let lr_w = hr_w / config.scale;
    let lr_h = hr_h / config.scale;

    if lr_w > 0 && lr_h > 0 {
        for (filter_name, filter_type) in &config.downsample_filters {
            let uncompressed_lr = image::imageops::resize(base_hr_image, lr_w, lr_h, *filter_type);

            // 保存未压缩的LR图像
            let lr_filename = format!("{}_{}_x{}_lr.png", stem, filter_name, config.scale);
            let lr_filepath = config.lr_path.join(&lr_filename);
            uncompressed_lr.save_with_format(&lr_filepath, ImageFormat::Png)?;
            pairs.push((lr_filename.into(), base_hr_filename.clone().into()));

            // 应用并保存JPEG压缩版本
            for &quality in &config.jpeg_quality {
                let mut buffer = Vec::new();
                JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut buffer), quality)
                    .encode_image(&uncompressed_lr)?;
                let compressed_img = image::load_from_memory(&buffer)?;

                let lr_filename_jpeg = format!(
                    "{}_{}_q{}_jpeg_x{}_lr.png",
                    stem, filter_name, quality, config.scale
                );
                let lr_filepath_jpeg = config.lr_path.join(&lr_filename_jpeg);
                ensure_rgba(&compressed_img)
                    .save_with_format(&lr_filepath_jpeg, ImageFormat::Png)?;
                pairs.push((lr_filename_jpeg.into(), base_hr_filename.clone().into()));
            }

            // 应用并保存WebP压缩版本
            for &quality in &config.webp_quality {
                // 1. 从RGBA图像数据创建 webp 编码器
                let encoder = webp::Encoder::from_rgba(
                    uncompressed_lr.as_raw(),
                    uncompressed_lr.width(),
                    uncompressed_lr.height(),
                );
                // 2. 使用指定的质量进行有损编码 (quality是0-100的浮点数)
                let memory = encoder.encode(quality as f32);
                // 3. 将编码后的webp数据加载回来，以便统一保存为png
                let compressed_img =
                    image::load_from_memory_with_format(&memory, ImageFormat::WebP)?;

                let lr_filename_webp = format!(
                    "{}_{}_q{}_webp_x{}_lr.png",
                    stem, filter_name, quality, config.scale
                );
                let lr_filepath_webp = config.lr_path.join(&lr_filename_webp);
                ensure_rgba(&compressed_img)
                    .save_with_format(&lr_filepath_webp, ImageFormat::Png)?;
                pairs.push((lr_filename_webp.into(), base_hr_filename.clone().into()));
            }
        }
    }

    // --- 2. 如果启用了数据增强，则处理增强版本 ---
    if config.augment {
        let mut augmentations = HashMap::new();
        augmentations.insert("hflip", image::imageops::flip_horizontal(base_hr_image));
        augmentations.insert("rot90", image::imageops::rotate90(base_hr_image));

        for (aug_name, aug_hr_image) in augmentations {
            let aug_hr_filename = format!("{}_{}_hr.png", stem, aug_name);
            let aug_hr_filepath = config.hr_path.join(&aug_hr_filename);
            aug_hr_image.save_with_format(&aug_hr_filepath, ImageFormat::Png)?;

            let (hr_w, hr_h) = aug_hr_image.dimensions();
            let lr_w = hr_w / config.scale;
            let lr_h = hr_h / config.scale;

            if lr_w == 0 || lr_h == 0 {
                continue;
            }

            for (filter_name, filter_type) in &config.downsample_filters {
                let uncompressed_lr =
                    image::imageops::resize(&aug_hr_image, lr_w, lr_h, *filter_type);

                // 保存未压缩的增强版LR图像
                let lr_filename = format!(
                    "{}_{}_{}_x{}_lr.png",
                    stem, aug_name, filter_name, config.scale
                );
                let lr_filepath = config.lr_path.join(&lr_filename);
                uncompressed_lr.save_with_format(&lr_filepath, ImageFormat::Png)?;
                pairs.push((lr_filename.into(), aug_hr_filename.clone().into()));

                // 应用并保存JPEG压缩的增强版
                for &quality in &config.jpeg_quality {
                    let mut buffer = Vec::new();
                    JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut buffer), quality)
                        .encode_image(&uncompressed_lr)?;
                    let compressed_img = image::load_from_memory(&buffer)?;

                    let lr_filename_jpeg = format!(
                        "{}_{}_{}_q{}_jpeg_x{}_lr.png",
                        stem, aug_name, filter_name, quality, config.scale
                    );
                    let lr_filepath_jpeg = config.lr_path.join(&lr_filename_jpeg);
                    ensure_rgba(&compressed_img)
                        .save_with_format(&lr_filepath_jpeg, ImageFormat::Png)?;
                    pairs.push((lr_filename_jpeg.into(), aug_hr_filename.clone().into()));
                }

                // 应用并保存WebP压缩的增强版
                for &quality in &config.webp_quality {
                    let encoder = webp::Encoder::from_rgba(
                        uncompressed_lr.as_raw(),
                        uncompressed_lr.width(),
                        uncompressed_lr.height(),
                    );
                    let memory = encoder.encode(quality as f32);
                    let compressed_img =
                        image::load_from_memory_with_format(&memory, ImageFormat::WebP)?;

                    let lr_filename_webp = format!(
                        "{}_{}_{}_q{}_webp_x{}_lr.png",
                        stem, aug_name, filter_name, quality, config.scale
                    );
                    let lr_filepath_webp = config.lr_path.join(&lr_filename_webp);
                    ensure_rgba(&compressed_img)
                        .save_with_format(&lr_filepath_webp, ImageFormat::Png)?;
                    pairs.push((lr_filename_webp.into(), aug_hr_filename.clone().into()));
                }
            }
        }
    }

    Ok(pairs)
}
/// 确保图像是Rgba8格式，处理灰度和RGB图像
fn ensure_rgba(img: &DynamicImage) -> RgbaImage {
    match img {
        DynamicImage::ImageRgba8(rgba) => rgba.clone(),
        DynamicImage::ImageRgb8(rgb) => {
            let mut rgba = RgbaImage::new(rgb.width(), rgb.height());
            for (x, y, pixel) in rgb.enumerate_pixels() {
                rgba.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
            }
            rgba
        }
        DynamicImage::ImageLuma8(luma) => {
            let mut rgba = RgbaImage::new(luma.width(), luma.height());
            for (x, y, pixel) in luma.enumerate_pixels() {
                rgba.put_pixel(x, y, Rgba([pixel[0], pixel[0], pixel[0], 255]));
            }
            rgba
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            let mut rgba = RgbaImage::new(luma_a.width(), luma_a.height());
            for (x, y, pixel) in luma_a.enumerate_pixels() {
                rgba.put_pixel(x, y, Rgba([pixel[0], pixel[0], pixel[0], pixel[1]]));
            }
            rgba
        }
        _ => img.to_rgba8(), // 其他格式的通用转换
    }
}

/// 解析逗号分隔的字符串为 FilterType 枚举
fn parse_filter_types(s: &str) -> Result<Vec<(String, FilterType)>> {
    let mut filters = Vec::new();
    for name in s.split(',') {
        let trimmed = name.trim();
        let filter = match trimmed.to_lowercase().as_str() {
            "nearest" => FilterType::Nearest,
            "triangle" => FilterType::Triangle,
            "catmullrom" => FilterType::CatmullRom,
            "gaussian" => FilterType::Gaussian,
            "lanczos3" => FilterType::Lanczos3,
            _ => anyhow::bail!("不支持的下采样算法: '{}'", trimmed),
        };
        filters.push((trimmed.to_string(), filter));
    }
    Ok(filters)
}
