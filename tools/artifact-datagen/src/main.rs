use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba, RgbaImage};
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

// 使用 lazy_static 为 resvg 设置一个全局的、一次性的字体数据库 (无变动)
lazy_static! {
    static ref FONT_DB: usvg::fontdb::Database = {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        db
    };
}

/// 用于创建压缩伪影检测数据集的强大命令行工具
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// 输入图像所在的文件夹路径
    #[arg(short, long)]
    input_dir: PathBuf,

    /// 生成数据集的目标文件夹路径
    #[arg(short, long)]
    output_dir: PathBuf,

    /// 用于存放所有生成图像 (原始和压缩后) 的子文件夹名称
    #[arg(long, default_value = "images")]
    images_dir_name: String,

    /// [仅SVG] 以逗号分隔的渲染尺寸列表 (例如: "128,256,512")
    #[arg(long, default_value = "128,256,512,1024")]
    svg_sizes: String,

    /// [仅位图] 将尺寸过大的图像裁剪为指定大小的图块 (例如: 1024)
    #[arg(long)]
    patch_size: Option<u32>,

    /// [仅位图] 是否执行数据增强 (水平翻转和旋转90度)
    #[arg(long)]
    augment: bool,

    /// 对图像应用JPEG压缩，并指定以逗号分隔的质量列表 (1-100)
    #[arg(long)]
    jpeg_quality: Option<String>,

    /// 对图像应用WebP压缩，并指定以逗号分隔的质量列表 (1-100)
    #[arg(long)]
    webp_quality: Option<String>,
    
    /// 跳过确认提示，直接开始处理
    #[arg(short, long)]
    yes: bool,
}

/// 解析后的配置结构体
#[derive(Debug, Clone)]
struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    images_path: PathBuf,
    svg_sizes: Vec<u32>,
    patch_size: Option<u32>,
    augment: bool,
    jpeg_quality: Vec<u8>,
    webp_quality: Vec<u8>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 解析和验证参数
    let svg_sizes = cli
        .svg_sizes
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect::<Vec<_>>();

    let jpeg_quality = cli.jpeg_quality.map_or_else(Vec::new, |s| {
        s.split(',')
            .filter_map(|q| q.trim().parse::<u8>().ok())
            .filter(|&q| q > 0 && q <= 100)
            .collect()
    });

    let webp_quality = cli.webp_quality.map_or_else(Vec::new, |s| {
        s.split(',')
            .filter_map(|q| q.trim().parse::<u8>().ok())
            .filter(|&q| q > 0 && q <= 100)
            .collect()
    });
    
    if jpeg_quality.is_empty() && webp_quality.is_empty() {
        anyhow::bail!("必须至少提供一种压缩选项: --jpeg-quality 或 --webp-quality");
    }
    
    let images_path = cli.output_dir.join(&cli.images_dir_name);

    let config = Config {
        input_dir: cli.input_dir,
        output_dir: cli.output_dir,
        images_path,
        svg_sizes,
        patch_size: cli.patch_size,
        augment: cli.augment,
        jpeg_quality,
        webp_quality,
    };

    println!("--- 压缩伪影检测数据集生成配置 (二分类模式) ---");
    println!("输入文件夹: {}", config.input_dir.display());
    println!("输出文件夹: {}", config.output_dir.display());
    println!("  - 图像文件夹: {}", config.images_path.display());
    println!("\n--- 矢量图 (SVG) 配置 ---");
    println!("渲染尺寸: {:?}", config.svg_sizes);
    println!("\n--- 位图 (PNG/JPG等) 配置 ---");
    if let Some(size) = config.patch_size {
        println!("大图像裁剪尺寸: {}x{}", size, size);
    } else {
        println!("大图像裁剪: 否");
    }
    println!(
        "数据增强 (翻转/旋转): {}",
        if config.augment { "是" } else { "否" }
    );
    println!("JPEG 压缩质量: {:?}", config.jpeg_quality);
    println!("WebP 压缩质量: {:?}", config.webp_quality);
    println!("---------------------------------------------------\n");

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


fn run_processing(config: &Config) -> Result<()> {
    println!("\n[1/4] 开始设置输出环境...");
    fs::create_dir_all(&config.images_path).context("无法创建图像文件夹")?;
    
    let index_file_path = config.output_dir.join("dataset_index.csv");
    let writer = Writer::from_path(&index_file_path).context("无法创建CSV索引文件")?;
    let csv_writer = Arc::new(Mutex::new(writer));

    {
        let mut guard = csv_writer.lock();
        guard
            .write_record(&["image_filename", "label"]) // label: 0=原始, 1=压缩
            .context("无法写入CSV表头")?;
    }

    println!("[2/4] 正在扫描输入文件...");
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
    let log_interval = (total_files / 100).max(1);

    println!("[3/4] 开始并行处理图像...");

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
        
        let result: Result<Vec<(String, u8)>> = match extension.to_lowercase().as_str() {
            "svg" => process_svg(&path, config),
            "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "webp" => process_bitmap(&path, config),
            _ => Ok(Vec::new()), 
        };

        match result {
            Ok(records) => {
                if !records.is_empty() {
                    let mut guard = csv_writer.lock();
                    for (filename, label) in records {
                        if let Err(e) = guard.write_record(&[
                            filename,
                            label.to_string(), // 将 u8 标签转为字符串
                        ]) {
                            eprintln!("错误: 无法将记录写入CSV: {}", e);
                        }
                    }
                }
            }
            Err(e) => eprintln!("错误: 处理文件 {} 失败: {}", path.display(), e),
        }
    });

    println!("[4/4] 处理完成！");
    println!("数据集已生成在: {}", config.output_dir.display());
    println!("所有图像位于: {}", config.images_path.display());
    println!("索引文件: {}", index_file_path.display());

    Ok(())
}

/// 处理SVG矢量图
fn process_svg(path: &Path, config: &Config) -> Result<Vec<(String, u8)>> {
    let mut all_records = Vec::new();
    let svg_data = fs::read(path)?;
    let tree = usvg::Tree::from_data(&svg_data, &usvg::Options::default())?;

    let stem = path.file_stem().unwrap().to_str().unwrap();

    for &size in &config.svg_sizes {
        let base_stem = format!("{}_{}", stem, size);

        let gt_image = render_svg_to_rgba(&tree, size)?;
        let records = generate_compressed_versions(&gt_image, &base_stem, config)?;
        all_records.extend(records);
    }
    Ok(all_records)
}

/// 将 usvg Tree 渲染为 RgbaImage (无变动)
fn render_svg_to_rgba(tree: &usvg::Tree, size: u32) -> Result<RgbaImage> {
    let mut pixmap = tiny_skia::Pixmap::new(size, size).context("无法创建Pixmap")?;
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
fn process_bitmap(path: &Path, config: &Config) -> Result<Vec<(String, u8)>> {
    let img = image::open(path).with_context(|| format!("无法打开图像 {}", path.display()))?;
    let stem = path.file_stem().unwrap().to_str().unwrap();

    let mut patches = Vec::new();

    if let Some(patch_size) = config.patch_size {
        if img.width() > patch_size || img.height() > patch_size {
            patches.extend(crop_image_with_overlap(&img, patch_size));
        } else {
            patches.push(img);
        }
    } else {
        patches.push(img);
    }

    let mut all_records = Vec::new();
    let is_multi_patch = patches.len() > 1;

    for (i, patch) in patches.into_iter().enumerate() {
        let rgba_patch = ensure_rgba(&patch);
        
        let (w, h) = rgba_patch.dimensions();
        let new_w = w - (w % 2);
        let new_h = h - (h % 2);
        if new_w == 0 || new_h == 0 { continue; }
        let base_gt_image = image::imageops::crop_imm(&rgba_patch, 0, 0, new_w, new_h).to_image();
        
        let patch_stem = if is_multi_patch {
            format!("{}_p{}", stem, i)
        } else {
            stem.to_string()
        };

        match process_single_image(&base_gt_image, &patch_stem, config) {
            Ok(records) => all_records.extend(records),
            Err(e) => eprintln!("错误: 处理图块 {} from {} 失败: {}", i, path.display(), e),
        }
    }

    Ok(all_records)
}

/// 将图像裁剪成图块 (无变动)
fn crop_image_with_overlap(img: &DynamicImage, patch_size: u32) -> Vec<DynamicImage> {
    let (width, height) = img.dimensions();
    let mut patches = Vec::new();

    let mut y = 0;
    while y < height {
        let actual_y = if y + patch_size > height { height.saturating_sub(patch_size) } else { y };
        let mut x = 0;
        while x < width {
            let actual_x = if x + patch_size > width { width.saturating_sub(patch_size) } else { x };
            patches.push(img.crop_imm(actual_x, actual_y, patch_size, patch_size));
            if x >= actual_x && x + patch_size >= width { break; }
            x += patch_size;
        }
        if y >= actual_y && y + patch_size >= height { break; }
        y += patch_size;
    }
    patches
}

/// 处理单个图像（或图块），应用增强并生成压缩版本。
fn process_single_image(
    base_gt_image: &RgbaImage,
    stem: &str,
    config: &Config,
) -> Result<Vec<(String, u8)>> {
    let mut all_records = Vec::new();

    // 1. 处理原始（基础）图像
    let records = generate_compressed_versions(base_gt_image, stem, config)?;
    all_records.extend(records);

    // 2. 如果启用了数据增强，则处理增强版本
    if config.augment {
        let mut augmentations = HashMap::new();
        augmentations.insert("hflip", image::imageops::flip_horizontal(base_gt_image));
        augmentations.insert("rot90", image::imageops::rotate90(base_gt_image));

        for (aug_name, aug_gt_image) in augmentations {
            let aug_stem = format!("{}_{}", stem, aug_name);
            let aug_records = generate_compressed_versions(&aug_gt_image, &aug_stem, config)?;
            all_records.extend(aug_records);
        }
    }

    Ok(all_records)
}


/// 从给定的 Ground Truth 图像生成所有配置的压缩版本，并返回 (文件名, 标签) 记录。
/// 标签 0 代表原始图像，标签 1 代表压缩图像。
fn generate_compressed_versions(
    gt_image: &RgbaImage,
    stem: &str,
    config: &Config,
) -> Result<Vec<(String, u8)>> {
    let mut records = Vec::new();

    // 1. 保存 Ground Truth 图像，并记录为标签 0
    let gt_filename = format!("{}_gt.png", stem);
    let gt_filepath = config.images_path.join(&gt_filename);
    gt_image.save_with_format(&gt_filepath, ImageFormat::Png)?;
    records.push((gt_filename, 0)); // 标签 0 表示原始 (未压缩)

    // 2. 应用并保存JPEG压缩版本, 记录为标签 1
    for &quality in &config.jpeg_quality {
        let mut buffer = Vec::new();
        JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut buffer), quality)
            .encode_image(gt_image)?;
        let compressed_img = image::load_from_memory(&buffer)?;

        let compressed_filename = format!("{}_q{}_jpeg.png", stem, quality);
        let compressed_filepath = config.images_path.join(&compressed_filename);
        ensure_rgba(&compressed_img).save_with_format(&compressed_filepath, ImageFormat::Png)?;
        records.push((compressed_filename, 1)); // 标签 1 表示已压缩
    }

    // 3. 应用并保存WebP压缩版本, 记录为标签 1
    for &quality in &config.webp_quality {
        let encoder = webp::Encoder::from_rgba(
            gt_image.as_raw(),
            gt_image.width(),
            gt_image.height(),
        );
        let memory = encoder.encode(quality as f32);
        let compressed_img = image::load_from_memory_with_format(&memory, ImageFormat::WebP)?;

        let compressed_filename = format!("{}_q{}_webp.png", stem, quality);
        let compressed_filepath = config.images_path.join(&compressed_filename);
        ensure_rgba(&compressed_img).save_with_format(&compressed_filepath, ImageFormat::Png)?;
        records.push((compressed_filename, 1)); // 标签 1 表示已压缩
    }

    Ok(records)
}


/// 确保图像是Rgba8格式 (无变动)
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
        _ => img.to_rgba8(),
    }
}