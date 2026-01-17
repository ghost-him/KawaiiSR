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
///    - Level 0: 原始图像 (如果有损则先 Bicubic 下采样 4 倍)
///    - Level N: 递归 2 倍下采样，直到下一次下采样会导致短边 < 128px
///    - Final Level: 如果当前短边 > 128px，则缩放到短边正好为 128px
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// 输入图像所在的文件夹路径 (GT)
    #[arg(short, long)]
    input_dir: PathBuf,

    /// 有损图像所在的文件夹路径 (Lossy, 将先进行 4x Bicubic 下采样)
    #[arg(short, long)]
    lossy_dir: Option<PathBuf>,

    /// 生成数据集的目标根文件夹路径
    #[arg(short, long)]
    output_dir: PathBuf,

    /// 并行线程数 (可选，默认使用所有核心)
    #[arg(short, long)]
    threads: Option<usize>,
}

struct InputFile {
    path: PathBuf,
    is_lossy: bool,
    rel_path: PathBuf,
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
    // 创建输出目录
    fs::create_dir_all(&folder_a).context("无法创建文件夹 A")?;

    println!("--- SR 数据集生成任务 ---");
    println!("输入目录 (GT): {}", cli.input_dir.display());
    if let Some(ref lossy) = cli.lossy_dir {
        println!("输入目录 (Lossy): {}", lossy.display());
    }
    println!("输出文件夹 A (HR/Multi-scale): {}", folder_a.display());
    println!("------------------------");

    println!("\n[1/2] 正在扫描输入文件...");

    let mut input_files = scan_dir(&cli.input_dir, false);
    if let Some(ref lossy_dir) = cli.lossy_dir {
        input_files.extend(scan_dir(lossy_dir, true));
    }

    let total_files = input_files.len();
    if total_files == 0 {
        println!("错误：未找到可处理的图像文件。");
        return Ok(());
    }
    println!("找到 {} 个图像文件，开始并行处理...\n", total_files);

    let processed_count = AtomicUsize::new(0);

    // 开始并行处理
    input_files.into_par_iter().for_each(|input| {
        if let Err(e) = process_single_image(&input, &folder_a) {
            eprintln!("\n[错误] 处理文件 {} 失败: {}", input.path.display(), e);
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

fn scan_dir(root: &Path, is_lossy: bool) -> Vec<InputFile> {
    WalkDir::new(root)
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
        .map(|e| {
            let rel_path = e
                .path()
                .strip_prefix(root)
                .unwrap_or(e.path())
                .to_path_buf();
            InputFile {
                path: e.path().to_path_buf(),
                is_lossy,
                rel_path,
            }
        })
        .collect()
}

/// 处理单张图片，执行多尺度下采样及锐化
fn process_single_image(input: &InputFile, folder_a: &Path) -> Result<()> {
    let mut img = image::open(&input.path)
        .with_context(|| format!("无法加载图像: {}", input.path.display()))?;

    // 如果是有损图像，先进行 2x Bicubic 下采样
    if input.is_lossy {
        let (w, h) = img.dimensions();
        // 使用 CatmullRom 作为 Bicubic 替代
        img = img.resize_exact(w / 2, h / 2, FilterType::CatmullRom);
    }

    let (orig_w, orig_h) = img.dimensions();
    let orig_short_side = orig_w.min(orig_h);

    // 检查短边是否小于 128px
    if orig_short_side < 128 {
        println!(
            "\n[跳过] 图片 {} 尺寸过小 ({}x{})，短边小于 128px",
            input.rel_path.display(),
            orig_w,
            orig_h
        );
        return Ok(());
    }

    let stem_orig = input
        .rel_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let stem = if input.is_lossy {
        format!("lossy_{}", stem_orig)
    } else {
        stem_orig.to_string()
    };
    let rel_parent = input.rel_path.parent().unwrap_or(Path::new(""));

    // 1. Level 0 (Original or Downsampled Lossy)
    save(&img, rel_parent, &stem, "lv0", folder_a)?;

    // 2. Level N (Recursive Downsampling)
    let mut last_processed_w = orig_w;
    let mut last_processed_h = orig_h;
    let mut level = 1;

    loop {
        let current_short_side = last_processed_w.min(last_processed_h);
        // 跳过短边小于 150px 的情况
        if current_short_side / 2 < 150 {
            break;
        }

        let divisor = 2u32.pow(level);
        let new_w = orig_w / divisor;
        let new_h = orig_h / divisor;

        if new_w == 0 || new_h == 0 {
            break;
        }

        let current_img = img.resize_exact(new_w, new_h, FilterType::CatmullRom);
        save(
            &current_img,
            rel_parent,
            &stem,
            &format!("lv{}", level),
            folder_a,
        )?;

        last_processed_w = new_w;
        last_processed_h = new_h;
        level += 1;
    }

    Ok(())
}

/// 将单个图像实例保存到 A 文件夹
fn save(
    img: &DynamicImage,
    rel_parent: &Path,
    stem: &str,
    suffix: &str,
    folder_a: &Path,
) -> Result<()> {
    let filename = format!("{}_{}.png", stem, suffix);

    let target_a_dir = folder_a.join(rel_parent);

    fs::create_dir_all(&target_a_dir)?;

    let out_a_path = target_a_dir.join(&filename);

    // 文件夹 A: 保存当前处理图 (PNG format)
    img.save_with_format(&out_a_path, image::ImageFormat::Png)
        .with_context(|| format!("无法保存图片到 A: {}", out_a_path.display()))?;
    Ok(())
}
