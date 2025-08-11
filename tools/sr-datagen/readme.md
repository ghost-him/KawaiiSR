> 以下的内容由 gemini 2.5 pro 生成

# 超分辨率数据集生成器 (Super-Resolution Dataset Generator)

这是一个功能强大的命令行工具，使用 Rust 编写，旨在帮助您快速、高效地为超分辨率（Super-Resolution）模型创建训练和测试数据集。它能够处理矢量图（SVG）和常见的位图（PNG, JPG等），并提供了丰富的选项来生成多样化的低分辨率（LR）和高分辨率（HR）图像对。

## 核心功能

*   **并行处理**: 利用 Rayon 库实现多核并行处理，极大地加快了大规模数据集的生成速度。
*   **支持多种图像格式**:
    *   **矢量图 (SVG)**: 将 SVG 按指定尺寸渲染为高质量的 PNG 图像，从源头上保证 HR 图像的清晰度。
    *   **位图 (PNG, JPG, BMP, WebP等)**: 支持所有常见位图格式。
*   **灵活的下采样**:
    *   可同时使用多种下采样算法（如 `Lanczos3`, `Nearest`）生成不同风格的 LR 图像。
    *   可自定义超分辨率的缩放比例（`scale`）。
*   **图像分块 (Patching)**: 自动将尺寸过大的图像裁剪为指定大小的图块（Patches），便于模型训练。
*   **数据增强 (Augmentation)**: 一键开启水平翻转和90度旋转，轻松扩充数据集。
*   **模拟真实世界降质**: 可对生成的 LR 图像应用不同质量的 JPEG 和 WebP 压缩，以模拟真实场景中的压缩伪影。
*   **清晰的输出**: 自动创建 `HR` 和 `LR` 文件夹，并生成一个 `dataset_index.csv` 索引文件，方便后续的数据加载。

## 环境准备与运行

### 1. 环境准备
您需要在您的系统上安装 [Rust 编程语言环境](https://www.rust-lang.org/tools/install)。安装完成后，您将拥有 `rustc` 编译器和 `cargo` 包管理器。

### 2. 运行程序
将代码保存为 `main.rs`，并确保 `Cargo.toml` 文件中包含了所有必要的依赖项（`anyhow`, `clap`, `csv`, `image`, `lazy_static`, `parking_lot`, `rayon`, `walkdir`, `usvg`, `tiny-skia`, `resvg`, `webp`）。

在您的项目根目录下，使用以下命令运行程序：

```bash
cargo run --release -- [OPTIONS]
```
*   `cargo run` 会编译并运行项目。
*   `--release` 标志会进行优化编译，强烈建议在处理大量数据时使用以获得最佳性能。
*   `--` 是一个分隔符，它告诉 Cargo 后面的所有参数都应传递给我们的程序，而不是 Cargo 本身。
*   `[OPTIONS]` 是您需要为程序指定的参数，详见下文。

## 处理流程

当您运行程序时，它会遵循以下步骤：

1.  **解析参数**: 程序首先解析您在命令行中提供的所有参数（如输入/输出路径、缩放比例等）。
2.  **确认配置**: 打印出完整的处理配置，并请求您确认（除非您使用了 `--yes` 参数跳过）。
3.  **创建输出目录**: 在指定的输出文件夹中创建 `HR` 和 `LR` 子文件夹。同时，创建一个 `dataset_index.csv` 文件并写入表头。
4.  **扫描文件**: 递归地扫描输入文件夹，找到所有支持的图像文件。
5.  **并行处理**: 将找到的文件列表分配到多个 CPU 核心上并行处理。
    *   **对于 SVG 文件**: 根据您提供的 `svg_sizes`，程序会为每个尺寸渲染一张 HR 图像，并相应地渲染一张缩小 `scale` 倍的 LR 图像。
    *   **对于位图文件**:
        a. 如果图像尺寸大于 `patch_size`，则先将其裁剪为多个重叠的图块。
        b. 对每个图块（或原始图像）进行处理。
        c. 保存原始的 HR 图块。
        d. 使用您指定的每一种 `downsample_filters` 算法，生成对应的 LR 图像。
        e. **（可选）** 对每张生成的 LR 图像，应用您指定的 `jpeg_quality` 和 `webp_quality`，生成带有压缩伪影的新 LR 图像。
        f. **（可选）** 如果开启了 `augment`，则对原始图块进行翻转和旋转，并对增强后的图像重复上述 a-e 步骤。
6.  **记录索引**: 每成功生成一对 (LR, HR) 图像，程序就会将它们的文件名（相对路径）写入 `dataset_index.csv` 文件。
7.  **完成**: 所有文件处理完毕后，程序退出。

## 使用方法

### 基本命令格式

```bash
cargo run -- --input-dir <输入路径> --output-dir <输出路径> [其他可选参数]
```

### 参数详解

| 参数 (Flag) | 速记 | 描述 | 是否必需 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `--input-dir` | `-i` | 包含源图像的文件夹路径。 | **是** | - |
| `--output-dir` | `-o` | 用于存放生成的数据集的文件夹路径。 | **是** | - |
| `--hr-dir-name` | | 高分辨率 (HR) 图像的子文件夹名称。 | 否 | `HR` |
| `--lr-dir-name` | | 低分辨率 (LR) 图像的子文件夹名称。 | 否 | `LR` |
| `--scale` | `-s` | 超分辨率的缩放比例，例如 2, 3, 4。 | 否 | `2` |
| `--patch-size`| | **[位图]** 如果图像尺寸大于此值，则裁剪为 `N x N` 大小的图块。 | 否 | 不裁剪 |
| `--augment` | | **[位图]** 是否执行数据增强（水平翻转和90度旋转）。 | 否 | `false` |
| `--downsample-filters` | | **[位图]** 以逗号分隔的下采样算法列表。可选值: `Nearest`, `Triangle`, `CatmullRom`, `Gaussian`, `Lanczos3`。 | 否 | `Nearest,Lanczos3` |
| `--jpeg-quality` | | **[位图]** 对LR图像应用JPEG压缩伪影。可提供多个值，用逗号分隔（例如 "75,85"）。质量范围 1-100。 | 否 | 不应用 |
| `--webp-quality` | | **[位图]** 对LR图像应用WebP压缩伪影。可提供多个值，用逗号分隔。质量范围 1-100。 | 否 | 不应用 |
| `--svg-sizes` | | **[SVG]** 以逗号分隔的SVG渲染尺寸列表。程序会为每个尺寸生成一个HR/LR对。 | 否 | `128,256,512,1024` |
| `--yes` | `-y` | 跳过运行前的确认提示。 | 否 | `false` |


## 使用示例

让我们详细分解您提供的示例命令，以理解其工作方式。

### 示例命令

```bash
cargo run -- --input-dir ./test_input --output-dir ./test_output --augment --jpeg-quality 80 --webp-quality 90 --patch-size 1024
```
### 命令分解

*   `--input-dir ./test_input`: 指定 `./test_input` 文件夹作为源图像的输入。
*   `--output-dir ./test_output`: 指定 `./test_output` 文件夹用于存放所有生成的文件。
*   `--augment`: 开启数据增强。程序将为每张图片额外生成水平翻转和旋转90度的版本。
*   `--jpeg-quality 80`: 除了常规的LR图像外，还会额外生成一个带有80质量JPEG压缩伪影的LR版本。
*   `--webp-quality 90`: 同样，还会生成一个带有90质量WebP压缩伪影的LR版本。
*   `--patch-size 1024`: 如果输入文件夹中的位图尺寸大于 `1024x1024`，它将被裁剪成多个 `1024x1024` 的图块进行处理。
*   **未指定的参数**:
    *   `--scale`: 将使用默认值 `2`。
    *   `--downsample-filters`: 将使用默认值 `Nearest,Lanczos3`。

### 预期输出

假设 `./test_input` 中有一张名为 `my_cat.png` 的大尺寸图片（例如 `2000x1500`）。程序将执行以下操作：

1.  **分块**: 因为 `2000x1500` 大于 `1024x1024`，图片会被裁剪为多个 `1024x1024` 的图块。我们以第一个图块 `my_cat_p0` 为例。

2.  **处理原始图块**:
    *   **HR**: 生成 `my_cat_p0_hr.png`。
    *   **LR (未压缩)**:
        *   使用 `Nearest` 算法生成 `my_cat_p0_Nearest_x2_lr.png`。
        *   使用 `Lanczos3` 算法生成 `my_cat_p0_Lanczos3_x2_lr.png`。
    *   **LR (带JPEG伪影)**:
        *   基于 `Nearest` 版本生成 `my_cat_p0_Nearest_q80_jpeg_x2_lr.png`。
        *   基于 `Lanczos3` 版本生成 `my_cat_p0_Lanczos3_q80_jpeg_x2_lr.png`。
    *   **LR (带WebP伪影)**:
        *   基于 `Nearest` 版本生成 `my_cat_p0_Nearest_q90_webp_x2_lr.png`。
        *   基于 `Lanczos3` 版本生成 `my_cat_p0_Lanczos3_q90_webp_x2_lr.png`。

3.  **处理增强图块**:
    *   对 `my_cat_p0` 进行水平翻转，得到 `my_cat_p0_hflip`。
    *   **HR**: 生成 `my_cat_p0_hflip_hr.png`。
    *   **LR**: 为其生成所有对应的 LR 版本（`Nearest`, `Lanczos3`, `JPEG`, `WebP`），文件名中会包含 `hflip` 标识，例如 `my_cat_p0_hflip_Lanczos3_q80_jpeg_x2_lr.png`。
    *   对 `my_cat_p0` 进行90度旋转，得到 `my_cat_p0_rot90`，并重复上述过程。

4.  **最终目录结构**:

    ```
    ./test_output/
    ├── HR/
    │   ├── my_cat_p0_hr.png
    │   ├── my_cat_p0_hflip_hr.png
    │   ├── my_cat_p0_rot90_hr.png
    │   └── ... (其他图块和图片的HR文件)
    ├── LR/
    │   ├── my_cat_p0_Nearest_x2_lr.png
    │   ├── my_cat_p0_Lanczos3_x2_lr.png
    │   ├── my_cat_p0_Nearest_q80_jpeg_x2_lr.png
    │   ├── my_cat_p0_Lanczos3_q80_jpeg_x2_lr.png
    │   ├── my_cat_p0_Nearest_q90_webp_x2_lr.png
    │   ├── my_cat_p0_Lanczos3_q90_webp_x2_lr.png
    │   ├── my_cat_p0_hflip_Nearest_x2_lr.png
    │   └── ... (所有其他组合的LR文件)
    └── dataset_index.csv
    ```

5.  **索引文件 (`dataset_index.csv`)**:
    文件内容会像这样，将每一张 LR 图像与其对应的 HR 图像关联起来：
    ```csv
    lr_image_path,hr_image_path
    LR/my_cat_p0_Nearest_x2_lr.png,HR/my_cat_p0_hr.png
    LR/my_cat_p0_Lanczos3_x2_lr.png,HR/my_cat_p0_hr.png
    LR/my_cat_p0_Nearest_q80_jpeg_x2_lr.png,HR/my_cat_p0_hr.png
    LR/my_cat_p0_hflip_Nearest_x2_lr.png,HR/my_cat_p0_hflip_hr.png
    ...
    ```

## 注意事项
*   所有最终输出的图像文件（包括应用了 JPEG/WebP 压缩的）都会被保存为 **PNG 格式**。这是为了保持格式统一，并确保带有压缩伪影的图像不会被再次有损压缩。
*   程序会静默跳过不支持的文件类型，不会报错中断。
*   处理大量高分辨率图像会消耗较多内存和CPU资源，请确保您的机器配置足够。