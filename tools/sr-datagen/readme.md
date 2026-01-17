# SR-Datagen: 超分辨率数据集生成工具

这是一个专注于动漫/二次元风格的高性能超分辨率（SR）数据集生成工具。它能够自动执行多尺度下采样，并生成经过高级锐化处理的 Ground Truth (GT) 图像。

## 核心功能

*   **多尺度下采样 (Multi-scale Downsampling)**: 
    *   **Level 0**: 保持原始分辨率（或针对有损源进行 2x 下采样）。
    *   **Level N**: 递归执行 2 倍下采样（使用 Lanczos3 算法），直到分辨率达到设定的下限。
*   **多源处理**:
    *   支持高质量 GT 源文件夹。
    *   支持有损 (Lossy) 源文件夹（自动预执行 2x Bicubic 下采样，以消除压缩伪影）。
*   **高性能并行**:
    *   底层基于 `Rayon` 框架，在多核处理器上自动并行处理，极大缩短大数据集的生成时间。
*   **智能过滤**:
    *   自动跳过短边小于 128px 的图像，确保数据集质量。

## 处理逻辑

### 1. 文件夹 A (多尺度 HR 图像)
- **Lv 0**: 输入图像的原始尺寸。
- **Lv 1...n**: 每次将前一级别图像缩小 2 倍。
- **下采样算法**: 使用 Lanczos3 算法以获得最佳图像质量。

### 3. 有损图像处理
- 如果指定了 `--lossy-dir`，程序会先将其通过 Bicubic (CatmullRom) 下采样 2 倍，以大幅减少源文件中的噪声和压缩伪影。

## 快速开始

### 运行环境
- 已安装 Rust (Cargo) 环境。

### 基本用法
```bash
cargo run --release -- -i <GT目录> -o <输出目录>
```

### 完整参数说明
```bash
Options:
  -i, --input-dir <INPUT_DIR>    输入图像所在的文件夹路径 (高质量 GT)
  -l, --lossy-dir <LOSSY_DIR>    有损图像所在的文件夹路径 (将先进行 2x 下采样)
  -o, --output-dir <OUTPUT_DIR>  生成数据集的目标根文件夹路径
  -t, --threads <THREADS>        并行线程数 (默认使用所有核心)
  -h, --help                     显示帮助信息
```

## 输出结构
处理完成后，输出目录将包含以下结构：
```text
output/
├── A/ (HR/Multi-scale)
│   ├── subfolder/
│   │   ├── image_lv0.png
│   │   ├── image_lv1.png
│   │   └── ...
└── B/ (Sharpened/GT)
    ├── subfolder/
    │   ├── image_lv0.png
    │   ├── image_lv1.png
    │   └── ...
```