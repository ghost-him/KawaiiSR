use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::{Array2, Array3, Axis};
use rand::Rng;
use image::{RgbaImage, DynamicImage};
use image::Rgba;
use std::collections::{HashSet, VecDeque};

// XDoG配置
struct XDoGConfig {
    sigma: f32,
    eps: f32,
    phi: f32,
    k: f32,
    gamma: f32,
}

impl Default for XDoGConfig {
    fn default() -> Self {
        let mut rng = rand::rng();
        Self {
            sigma: 0.6,
            eps: -15.0,
            phi: 10e8,
            k: 2.5,
            gamma: 0.97 + 0.01 * rng.random::<f32>(),
        }
    }
}

/// 将动漫风格的图像进行高级锐化。
///
/// 该函数实现了专为动漫设计的两阶段锐化算法。
///
/// # Arguments
///
/// * `img` - 输入的动态图像 (`image::DynamicImage`)。
/// * `extra_sharpen_times` - 额外的USM锐化迭代次数，原脚本中固定为2。
/// * `outlier_threshold` - XDoG线稿提取中用于移除噪点的阈值，原脚本默认为32。
///
/// # Returns
///
/// * 返回一个经过锐化处理的新的 `image::DynamicImage`。
pub fn anime_sharpen(
    img: &DynamicImage,
    extra_sharpen_times: u32,
    outlier_threshold: u32,
) -> DynamicImage {
    match img {
        DynamicImage::ImageRgb8(rgb_img) => {
            let sharpened = sharpen_image_internal(rgb_img, extra_sharpen_times, outlier_threshold as usize);
            DynamicImage::ImageRgb8(sharpened)
        }
        DynamicImage::ImageRgba8(rgba_img) => {
            let (rgb_img, alpha_channel) = extract_rgb_and_alpha(rgba_img);
            let sharpened_rgb = sharpen_image_internal(&rgb_img, extra_sharpen_times, outlier_threshold as usize);
            let sharpened_rgba = combine_rgb_and_alpha(&sharpened_rgb, &alpha_channel);
            DynamicImage::ImageRgba8(sharpened_rgba)
        }
        _ => {
            // 对于其他格式，转换为RGB8后处理
            let rgb_img = img.to_rgb8();
            let sharpened = sharpen_image_internal(&rgb_img, extra_sharpen_times, outlier_threshold as usize);
            DynamicImage::ImageRgb8(sharpened)
        }
    }
}

/// 从RGBA图像中提取RGB和Alpha通道
fn extract_rgb_and_alpha(rgba_img: &RgbaImage) -> (RgbImage, Vec<u8>) {
    let (width, height) = rgba_img.dimensions();
    let mut rgb_img = ImageBuffer::new(width, height);
    let mut alpha_channel = Vec::with_capacity((width * height) as usize);
    
    for (x, y, pixel) in rgba_img.enumerate_pixels() {
        let alpha = pixel[3];
        alpha_channel.push(alpha);
        
        if alpha == 0 {
            // 透明像素设为白色
            rgb_img.put_pixel(x, y, Rgb([255, 255, 255]));
        } else {
            rgb_img.put_pixel(x, y, Rgb([pixel[0], pixel[1], pixel[2]]));
        }
    }
    
    (rgb_img, alpha_channel)
}

/// 将RGB图像和Alpha通道合并为RGBA图像
fn combine_rgb_and_alpha(rgb_img: &RgbImage, alpha_channel: &[u8]) -> RgbaImage {
    let (width, height) = rgb_img.dimensions();
    let mut rgba_img = ImageBuffer::new(width, height);
    
    for (x, y, rgb_pixel) in rgb_img.enumerate_pixels() {
        let alpha = alpha_channel[(y * width + x) as usize];
        rgba_img.put_pixel(x, y, Rgba([rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], alpha]));
    }
    
    rgba_img
}


/// 内部锐化函数
fn sharpen_image_internal(input_image: &RgbImage, extra_sharpen_times: u32, outlier_threshold: usize) -> RgbImage {
    // 第一次USM锐化
    let mut img = usm_sharpen(input_image, 0.5, 10.0);
    let first_sharpened = img.clone();
    
    // 额外锐化
    for _ in 0..extra_sharpen_times {
        img = usm_sharpen(&img, 0.5, 10.0);
    }
    
    // 生成XDoG草图映射
    let sketch_map = get_xdog_sketch_map(&img, outlier_threshold);
    
    // 混合图像
    blend_images(&img, &first_sharpened, &sketch_map)
}

/// USM锐化
fn usm_sharpen(img: &RgbImage, weight: f32, threshold: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let radius = 51; // 确保是奇数
    let sigma = 0.0; // 如果sigma为0，使用默认值
    let actual_sigma = if sigma == 0.0 { radius as f32 / 6.0 } else { sigma };
    
    // 转换为浮点数组
    let img_array = rgb_to_array(img);
    
    // 高斯模糊
    let blur = gaussian_blur(&img_array, radius, actual_sigma);
    
    // 计算残差
    let residual = &img_array - &blur;
    
    // 创建掩码
    let mask = create_threshold_mask(&residual, threshold);
    
    // 软掩码（对掩码进行高斯模糊）
    let soft_mask = gaussian_blur(&mask, radius, actual_sigma);
    
    // 锐化
    let sharp = &img_array + &(&residual * weight);
    let sharp = sharp.mapv(|x| x.clamp(0.0, 1.0));
    
    // 应用软掩码
    let output = &soft_mask * &sharp + &(1.0 - &soft_mask) * &img_array;
    
    array_to_rgb(&output, width, height)
}

/// 获取XDoG草图映射
fn get_xdog_sketch_map(img: &RgbImage, outlier_threshold: usize) -> Array2<f32> {
    // 转换为灰度
    let gray = rgb_to_grayscale(img);
    
    // 生成XDoG图像
    gen_xdog_image(&gray, outlier_threshold)
}

/// 生成XDoG图像
fn gen_xdog_image(gray: &Array2<f32>, outlier_threshold: usize) -> Array2<f32> {
    let config = XDoGConfig::default();
    
    // XDoG处理
    let mut dogged = xdog(gray, &config);
    dogged = 1.0 - dogged; // 黑白反转
    
    // 移除异常值
    dogged = outlier_removal(&dogged, outlier_threshold);
    
    // 被动膨胀
    passive_dilate(&dogged)
}

/// DoG (Difference of Gaussians)
fn dog(image: &Array2<f32>, sigma: f32, k: f32, gamma: f32) -> Array2<f32> {
    let g1 = gaussian_blur_2d(image, sigma);
    let g2 = gaussian_blur_2d(image, sigma * k);
    g1 - gamma * g2
}

/// XDoG (eXtended Difference of Gaussians)
fn xdog(image: &Array2<f32>, config: &XDoGConfig) -> Array2<f32> {
    let eps = config.eps / 255.0;
    let mut d = dog(image, config.sigma, config.k, config.gamma);
    
    // 归一化
    let max_val = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val > 0.0 {
        d /= max_val;
    }
    
    // 应用tanh函数
    d.mapv(|x| {
        let e = 1.0 + (config.phi * (x - eps)).tanh();
        if e >= 1.0 { 1.0 } else { e }
    })
}

/// 异常值移除
fn outlier_removal(img: &Array2<f32>, outlier_threshold: usize) -> Array2<f32> {
    let (height, width) = img.dim();
    let mut temp = img.clone();
    let mut global_list = HashSet::new();
    
    for i in 0..height {
        for j in 0..width {
            if global_list.contains(&(i, j)) {
                continue;
            }
            if temp[[i, j]] != 1.0 {
                continue;
            }
            
            let visited = bfs(&temp, i, j, &global_list);
            
            if visited.len() < outlier_threshold {
                for (u, v) in &visited {
                    temp[[*u, *v]] = 0.0;
                }
            }
            
            for (u, v) in visited {
                global_list.insert((u, v));
            }
        }
    }
    
    temp
}

/// 广度优先搜索
fn bfs(img: &Array2<f32>, start_i: usize, start_j: usize, global_list: &HashSet<(usize, usize)>) -> HashSet<(usize, usize)> {
    let (height, width) = img.dim();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    
    if global_list.contains(&(start_i, start_j)) || img[[start_i, start_j]] != 1.0 {
        return visited;
    }
    
    visited.insert((start_i, start_j));
    queue.push_back((start_i, start_j));
    
    let directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];
    
    while let Some((base_row, base_col)) = queue.pop_front() {
        for (dx, dy) in directions.iter() {
            let row = base_row as i32 + dx;
            let col = base_col as i32 + dy;
            
            if row < 0 || col < 0 || row >= height as i32 || col >= width as i32 {
                continue;
            }
            
            let (row, col) = (row as usize, col as usize);
            
            if visited.contains(&(row, col)) || global_list.contains(&(row, col)) {
                continue;
            }
            
            if img[[row, col]] == 1.0 {
                visited.insert((row, col));
                queue.push_back((row, col));
            }
        }
    }
    
    visited
}

/// 被动膨胀
fn passive_dilate(img: &Array2<f32>) -> Array2<f32> {
    let (height, width) = img.dim();
    let mut temp = img.clone();
    
    for i in 0..height {
        for j in 0..width {
            if img[[i, j]] == 1.0 {
                continue;
            }
            
            let mut num_white = 0;
            for di in -1..=1 {
                for dj in -1..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    
                    if ni >= 0 && nj >= 0 && ni < height as i32 && nj < width as i32 {
                        if img[[ni as usize, nj as usize]] == 1.0 {
                            num_white += 1;
                        }
                    }
                }
            }
            
            if num_white >= 3 {
                temp[[i, j]] = 1.0;
            }
        }
    }
    
    temp
}

/// 高斯模糊（2D）
fn gaussian_blur_2d(image: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let kernel_size = (sigma * 6.0).ceil() as usize;
    let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
    let kernel = create_gaussian_kernel_1d(kernel_size, sigma);
    
    // 水平方向模糊
    let temp = convolve_horizontal(image, &kernel);
    // 垂直方向模糊
    convolve_vertical(&temp, &kernel)
}

/// 创建1D高斯核
fn create_gaussian_kernel_1d(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = size / 2;
    let mut sum = 0.0;
    
    for i in 0..size {
        let x = (i as i32 - center as i32) as f32;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    
    // 归一化
    for k in &mut kernel {
        *k /= sum;
    }
    
    kernel
}

/// 水平卷积
fn convolve_horizontal(image: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (height, width) = image.dim();
    let mut result = Array2::zeros((height, width));
    let kernel_size = kernel.len();
    let half_kernel = kernel_size / 2;
    
    for i in 0..height {
        for j in 0..width {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let col = j as i32 + k as i32 - half_kernel as i32;
                let col = col.clamp(0, width as i32 - 1) as usize;
                sum += image[[i, col]] * kernel[k];
            }
            result[[i, j]] = sum;
        }
    }
    
    result
}

/// 垂直卷积
fn convolve_vertical(image: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (height, width) = image.dim();
    let mut result = Array2::zeros((height, width));
    let kernel_size = kernel.len();
    let half_kernel = kernel_size / 2;
    
    for i in 0..height {
        for j in 0..width {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let row = i as i32 + k as i32 - half_kernel as i32;
                let row = row.clamp(0, height as i32 - 1) as usize;
                sum += image[[row, j]] * kernel[k];
            }
            result[[i, j]] = sum;
        }
    }
    
    result
}

/// 辅助函数：RGB图像转换为数组
fn rgb_to_array(img: &RgbImage) -> Array3<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array3::zeros((height as usize, width as usize, 3));
    
    for (x, y, pixel) in img.enumerate_pixels() {
        array[[y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
        array[[y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
        array[[y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
    }
    
    array
}

/// 辅助函数：数组转换为RGB图像
fn array_to_rgb(array: &Array3<f32>, width: u32, height: u32) -> RgbImage {
    let mut img = ImageBuffer::new(width, height);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[y as usize, x as usize, 0]] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (array[[y as usize, x as usize, 1]] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (array[[y as usize, x as usize, 2]] * 255.0).clamp(0.0, 255.0) as u8;
        *pixel = Rgb([r, g, b]);
    }
    
    img
}

/// 辅助函数：RGB转灰度
fn rgb_to_grayscale(img: &RgbImage) -> Array2<f32> {
    let (width, height) = img.dimensions();
    let mut gray = Array2::zeros((height as usize, width as usize));
    
    for (x, y, pixel) in img.enumerate_pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        gray[[y as usize, x as usize]] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    
    gray
}

/// 辅助函数：创建阈值掩码
fn create_threshold_mask(residual: &Array3<f32>, threshold: f32) -> Array3<f32> {
    residual.mapv(|x| if (x * 255.0).abs() > threshold { 1.0 } else { 0.0 })
}

/// 辅助函数：高斯模糊（3D数组）
fn gaussian_blur(array: &Array3<f32>, radius: usize, sigma: f32) -> Array3<f32> {
    let (height, width, channels) = array.dim();
    let mut result = Array3::zeros((height, width, channels));
    
    for c in 0..channels {
        let channel = array.slice(s![.., .., c]).to_owned();
        let blurred = gaussian_blur_2d(&channel, sigma);
        result.slice_mut(s![.., .., c]).assign(&blurred);
    }
    
    result
}

/// 辅助函数：混合图像
fn blend_images(img1: &RgbImage, img2: &RgbImage, mask: &Array2<f32>) -> RgbImage {
    let (width, height) = img1.dimensions();
    let mut result = ImageBuffer::new(width, height);
    
    for (x, y, pixel) in result.enumerate_pixels_mut() {
        let p1 = img1.get_pixel(x, y);
        let p2 = img2.get_pixel(x, y);
        let m = mask[[y as usize, x as usize]];
        
        let r = (p1[0] as f32 * m + p2[0] as f32 * (1.0 - m)) as u8;
        let g = (p1[1] as f32 * m + p2[1] as f32 * (1.0 - m)) as u8;
        let b = (p1[2] as f32 * m + p2[2] as f32 * (1.0 - m)) as u8;
        
        *pixel = Rgb([r, g, b]);
    }
    
    result
}

// 为了支持ndarray的切片语法
use ndarray::s;