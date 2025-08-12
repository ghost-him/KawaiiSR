> 以下的内容由 gemini 2.5 pro 生成

# 压缩伪影检测数据集生成器

这是一个功能强大的命令行工具，使用 Rust 编写，旨在自动化地创建用于训练**压缩伪影检测 (Compression Artifact Detection)** 模型的数据集。它可以处理常见的位图（PNG, JPG等）和矢量图（SVG），并应用多种压缩算法和数据增强策略，快速生成结构化的训练数据。

## ✨ 主要功能

- **多格式支持**: 同时处理位图 (`PNG`, `JPG`, `BMP`, `WEBP` 等) 和矢量图 (`SVG`)。
- **灵活的压缩选项**: 支持多种质量等级的 `JPEG` 和 `WebP` 压缩。
- **智能 SVG 处理**: 可将 SVG 文件渲染成多种指定尺寸的图像再进行处理，有效扩充数据。
- **位图切片 (Patching)**: 自动将过大的位图裁剪为指定大小的图块，以适应模型输入尺寸。
- **数据增强**: 内置水平翻转和旋转90度的增强策略，进一步丰富数据集。
- **并行处理**: 利用 [Rayon](https://github.com/rayon-rs/rayon) 库实现多核并行处理，极大地提升了处理速度。
- **结构化输出**: 生成的图像统一存放，并附带一个 `CSV` 索引文件，清晰地标注了每张图片的路径和标签（原始或压缩）。

## ⚙️ 依赖与编译

本项目基于 Rust 开发，你需要安装 [Rust 工具链](https://www.rust-lang.org/tools/install) (包括 `cargo`) 来编译和运行它。

1.  **克隆或下载项目代码**

2.  **编译项目**
    在项目根目录下运行以下命令进行编译。推荐使用 `--release` 模式以获得最佳性能。
    ```bash
    cargo build --release
    ```
    编译完成后，可执行文件将位于 `target/release/` 目录下。

## 🚀 使用方法

本工具通过命令行接口进行操作。基本命令格式如下：

```bash
# 假设可执行文件名为 artifact-generator
./artifact-generator -i <输入文件夹> -o <输出文件夹> --jpeg-quality <质量列表> [其他选项]
```

你必须提供一个输入目录、一个输出目录，以及至少一种压缩选项 (`--jpeg-quality` 或 `--webp-quality`)。

### 示例

**示例 1: 基本用法**
处理 `raw_images` 文件夹中的所有图像，将结果保存在 `dataset_v1` 目录中。对每张图生成质量为 90 和 75 的 JPEG 压缩版本。

```bash
./target/release/artifact-generator \
    -i ./raw_images \
    -o ./dataset_v1 \
    --jpeg-quality "90,75"
```

**示例 2: 处理 SVG 和大型位图**
处理 `source_files` 文件夹。将所有 SVG 文件渲染为 `256x256` 和 `512x512` 两种尺寸。将尺寸大于 `1024x1024` 的位图裁剪成图块。同时生成 JPEG (质量85) 和 WebP (质量80) 的压缩版本。

```bash
./target/release/artifact-generator \
    -i ./source_files \
    -o ./dataset_advanced \
    --svg-sizes "256,512" \
    --patch-size 1024 \
    --jpeg-quality "85" \
    --webp-quality "80"
```

**示例 3: 启用数据增强并跳过确认**
在示例 2 的基础上，对所有位图（或其图块）额外应用水平翻转和90度旋转，并跳过开始前的确认提示，直接开始处理。

```bash
./target/release/artifact-generator \
    -i ./source_files \
    -o ./dataset_augmented \
    --svg-sizes "256,512" \
    --patch-size 1024 \
    --jpeg-quality "85" \
    --webp-quality "80" \
    --augment \
    -y
```

## 📋 参数详解

| 短参数 | 长参数                | 描述                                                                                                        | 默认值                  |
| :----- | :-------------------- | :---------------------------------------------------------------------------------------------------------- | :---------------------- |
| `-i`   | `--input-dir`         | **[必需]** 包含原始图像（SVG, PNG, JPG等）的输入文件夹路径。                                                   | -                       |
| `-o`   | `--output-dir`        | **[必需]** 用于存放生成的数据集的文件夹路径。如果不存在，程序会自动创建。                                      | -                       |
|        | `--jpeg-quality`      | **[必需(之一)]** 以逗号分隔的 JPEG 压缩质量列表 (1-100)。例如: `"95,80,60"`。                                    | -                       |
|        | `--webp-quality`      | **[必需(之一)]** 以逗号分隔的 WebP 压缩质量列表 (1-100)。例如: `"90,75"`。                                     | -                       |
|        | `--svg-sizes`         | **[仅SVG]** 以逗号分隔的列表，用于指定将 SVG 文件渲染成的正方形尺寸。                                           | `"128,256,512,1024"`    |
|        | `--patch-size`        | **[仅位图]** 如果位图的宽或高超过此值，则将其裁剪成 `N x N` 大小的图块进行处理。                                | `None` (不裁剪)         |
|        | `--augment`           | **[仅位图]** 如果设置，将对每个位图（或图块）额外生成水平翻转和旋转90度的版本。                                  | `false` (不增强)        |
|        | `--images-dir-name`   | 在输出目录中，用于存放所有生成图像的子文件夹的名称。                                                        | `"images"`              |
| `-y`   | `--yes`               | 跳过运行前的配置确认提示，直接开始处理。                                                                    | `false` (需要确认)      |

> **注意**: 必须提供 `--jpeg-quality` 和/或 `--webp-quality` 中的至少一个。

## 📦 输出结构

成功运行后，指定的输出文件夹 (`--output-dir`) 将包含以下内容：

```
<output_dir>/
├── images/  (<-- 由 --images-dir-name 指定)
│   ├── original_cat_gt.png
│   ├── original_cat_q90_jpeg.png
│   ├── original_cat_q75_jpeg.png
│   ├── photo1_p0_gt.png
│   ├── photo1_p0_hflip_gt.png
│   ├── photo1_p0_hflip_q85_webp.png
│   ├── vector_logo_512_gt.png
│   └── vector_logo_512_q95_jpeg.png
│   └── ... (更多图片)
│
└── dataset_index.csv
```

-   **`images/` 文件夹**: 包含所有生成的图像文件。
    -   **原始图像 (Ground Truth)**: 文件名以 `_gt.png` 结尾。所有原始图像都以无损的 PNG 格式保存。
    -   **压缩图像**: 文件名中包含压缩质量和类型，例如 `_q90_jpeg.png` 或 `_q80_webp.png`。为了避免二次压缩引入的伪影，这些图像在经过内存压缩后，也被保存为无损的 PNG 格式。
    -   **命名约定**: 文件名包含了原始文件名、尺寸/图块信息、增强信息等，具有高可读性。
-   **`dataset_index.csv` 文件**: 一个索引文件，记录了每张图像与它的标签。
    -   `image_filename`: 对应 `images/` 文件夹中的文件名。
    -   `label`: `0` 代表原始图像 (Ground Truth)，`1` 代表经过压缩的图像。