use anyhow::{Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;

pub mod strong_gt;

/// 优化后的超分辨率（SR）数据集生成器
///
/// 逻辑：
/// 1. 文件夹 A：多尺度下采样处理
///    - Level 0: 原始图像
///    - Level N: 递归 2 倍下采样，直到下一次下采样会导致短边 < 128px
///    - Final Level: 如果当前短边 > 128px，则缩放到短边正好为 128px
/// 2. 文件夹 B：针对 A 中的每一张图片进行锐化处理 (使用 strong_gt 逻辑)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// 输入图像所在的文件夹路径
    #[arg(short, long)]
    input_dir: PathBuf,

    /// 生成数据集的目标根文件夹路径
    #[arg(short, long)]
    output_dir: PathBuf,

    /// 并行线程数 (可选，默认使用所有核心)
    #[arg(short, long)]
    threads: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 配置并行线程数
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }

    let folder_a = cli.output_dir.join("A");
    let folder_b = cli.output_dir.join("B");

    // 创建输出目录
    fs::create_dir_all(&folder_a).context("无法创建文件夹 A")?;
    fs::create_dir_all(&folder_b).context("无法创建文件夹 B")?;

    println!("--- SR 数据集生成任务 ---");
    println!("输入目录: {}", cli.input_dir.display());
    println!("输出文件夹 A (HR/Multi-scale): {}", folder_a.display());
    println!("输出文件夹 B (Sharpened/GT): {}", folder_b.display());
    println!("------------------------");

    println!("\n[1/2] 正在扫描输入文件...");
    let paths: Vec<PathBuf> = WalkDir::new(&cli.input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            if !path.is_file() {
                return false;
            }
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            matches!(
                ext.as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "webp" | "tiff"
            )
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    let total_files = paths.len();
    if total_files == 0 {
        println!("错误：未找到可处理的图像文件。");
        return Ok(());
    }
    println!("找到 {} 个图像文件，开始并行处理...\n", total_files);

    let processed_count = AtomicUsize::new(0);

    // 开始并行处理
    paths.into_par_iter().for_each(|path| {
        if let Err(e) = process_single_image(&path, &folder_a, &folder_b) {
            eprintln!("\n[错误] 处理文件 {} 失败: {}", path.display(), e);
        }

        let current = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
        if current % 1 == 0 {
            // 频繁更新进度
            print!(
                "\r进度: {}/{} ({:.1}%)",
                current,
                total_files,
                (current as f32 / total_files as f32) * 100.0
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    });

    println!(
        "\n\n[2/2] 处理完成！数据集已保存至: {}",
        cli.output_dir.display()
    );
    Ok(())
}

/// 处理单张图片，执行多尺度下采样及锐化
fn process_single_image(path: &Path, folder_a: &Path, folder_b: &Path) -> Result<()> {
    let original_img =
        image::open(path).with_context(|| format!("无法加载图像: {}", path.display()))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");

    let (orig_w, orig_h) = original_img.dimensions();
    let orig_short_side = orig_w.min(orig_h);

    // 1. Level 0 (Original)
    save_and_sharpen(&original_img, stem, "lv0", folder_a, folder_b)?;

    // 2. Level N (Recursive Downsampling - Always scale from original)
    // 追踪最后一次缩放后的尺寸，用于判断 Final Level
    let mut last_processed_w = orig_w;
    let mut last_processed_h = orig_h;
    let mut level = 1;

    loop {
        // 只要“当前”尺寸的一半 >= 128px，就进行下一次 2 倍下采样
        let current_short_side = last_processed_w.min(last_processed_h);
        if current_short_side / 2 < 128 {
            break;
        }

        // 计算下采样后的目标尺寸 (1/2, 1/4, 1/8...)
        let divisor = 2u32.pow(level);
        let new_w = orig_w / divisor;
        let new_h = orig_h / divisor;

        if new_w == 0 || new_h == 0 {
            break;
        }

        // 直接从 original_img 进行缩放
        let current_img = original_img.resize_exact(new_w, new_h, FilterType::Lanczos3);
        save_and_sharpen(
            &current_img,
            stem,
            &format!("lv{}", level),
            folder_a,
            folder_b,
        )?;

        last_processed_w = new_w;
        last_processed_h = new_h;
        level += 1;
    }

    Ok(())
}

/// 将单个图像实例保存到 A 文件夹，并生成锐化版本保存到 B 文件夹
fn save_and_sharpen(
    img: &DynamicImage,
    stem: &str,
    suffix: &str,
    folder_a: &Path,
    _folder_b: &Path,
) -> Result<()> {
    let filename = format!("{}_{}.png", stem, suffix);
    let out_a_path = folder_a.join(&filename);
    //let out_b_path = folder_b.join(&filename);

    // 文件夹 A: 保存当前处理图 (PNG format)
    img.save_with_format(&out_a_path, image::ImageFormat::Png)
        .with_context(|| format!("无法保存图片到 A: {}", out_a_path.display()))?;

    // // 文件夹 B: 生成并保存锐化版本
    // // 调用 strong_gt::anime_sharpen，固定推荐参数 (extra_sharpen_times=2, outlier_threshold=32)
    // let sharpened = strong_gt::anime_sharpen(img, 2, 32);
    // sharpened.save_with_format(&out_b_path, image::ImageFormat::Png)
    //     .with_context(|| format!("无法保存锐化图片到 B: {}", out_b_path.display()))?;

    Ok(())
}
