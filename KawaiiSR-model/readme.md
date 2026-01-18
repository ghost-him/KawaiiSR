# KawaiiSR - 高效动漫超分辨率模型

> 以下的内容由 Claude Haiku 4.5 编写

一个基于 **HAT** (Hybrid Attention Transformer) 的改进动漫超分辨率解决方案，融合了 **APISR** 的训练思路，专门用于低分辨率动漫图像的高保真恢复。

## 📋 项目特色

- **架构改进**：基于 HAT 的混合注意力机制，针对动漫内容优化
- **多阶段训练**：参考 APISR，采用 L1 预训练 + GAN 对抗训练的策略
- **综合损失设计**：结合像素级、频域、感知和对抗损失的多目标优化
- **高效数据处理**：支持在线动态退化、CUDA 预取、混合精度训练
- **灵活配置系统**：基于 YAML 的扁平配置，易于调整超参数

## 📁 项目结构

```
KawaiiSR-model/
├── KawaiiSR/                    # 核心模型
│   ├── HAT.py                   # HAT 骨干网络（混合注意力变换器）
│   └── KawaiiSR.py              # KawaiiSR 包装模型（HAT + 定制化设置）
├── Discriminator/
│   └── UNetDiscriminatorSN.py    # 判别器（U-Net + 谱归一化）
├── loss/                         # 损失函数模块
│   ├── KawaiiLoss.py             # 综合损失（像素+频域+感知+对抗）
│   ├── CharbonnierLoss.py        # 鲁棒像素级损失
│   ├── LaplacianLoss.py          # 频域约束损失
│   ├── HingeGANLoss.py           # Hinge 对抗损失
│   ├── AnimePerceptualLoss.py    # 动漫特化感知损失
│   └── VGGPerceptualLoss.py      # 基于 VGG19 的感知损失
├── configs/                      # 训练配置
│   ├── anime_finetune.yaml       # 动漫微调配置
│   ├── real_stage1.yaml          # 真实图像阶段1（L1预训练）
│   ├── real_stage2.yaml          # 真实图像阶段2（GAN训练）
│   ├── real_stage3.yaml          # 真实图像阶段3（fine-tune）
│   └── quick_validate.yaml       # 快速验证配置
├── test/configs/                 # 测试配置
│   ├── test_stage1.yaml
│   ├── test_stage2.yaml
│   └── test_stage3.yaml
├── train.py                      # 训练主入口
├── train_config.py               # 配置加载和管理
├── KawaiiTrainer.py              # 训练逻辑引擎
├── data_loader.py                # 数据加载和动态退化
├── run_inference.py              # 推理脚本
├── demo_kawaii.py                # 演示脚本
├── export_onnx.py                # 模型导出为 ONNX
├── onnx_superres.py              # ONNX 推理
├── flops.py                       # 计算模型 FLOPs
└── readme.md                      # 本文件
```

## 🔧 关键文件说明

### 核心模型
- **[KawaiiSR/HAT.py](KawaiiSR/HAT.py)**：HAT 骨干网络
  - 基于 Hybrid Attention Transformer，结合窗口自注意力和通道注意力
  - 支持可变深度、注意力头数配置
  - 采用 Residual Connection 设计

- **[KawaiiSR/KawaiiSR.py](KawaiiSR/KawaiiSR.py)**：顶层模型包装
  - 集成 HAT 主干网络
  - 定制化超参数传递和管理
  - 提供标准的超分辨率推理接口

### 损失函数
- **[loss/KawaiiLoss.py](loss/KawaiiLoss.py)**：综合损失函数
  - 融合多个损失：CharbonnierLoss（像素）、LaplacianLoss（频域）、感知损失、对抗损失
  - 支持动漫特化感知损失选项
  - 可灵活调整各损失的权重

- **[loss/CharbonnierLoss.py](loss/CharbonnierLoss.py)**：鲁棒像素级损失
  - 对异常值更敏感，减少锯齿和伪影

- **[loss/LaplacianLoss.py](loss/LaplacianLoss.py)**：频域约束
  - 通过拉普拉斯算子约束高频细节

- **[loss/AnimePerceptualLoss.py](loss/AnimePerceptualLoss.py)**：动漫感知损失
  - 使用动漫预训练模型（基于 Danbooru 2018 预训练）
  - 针对动漫风格特征的特化感知约束

- **[loss/VGGPerceptualLoss.py](loss/VGGPerceptualLoss.py)**：通用感知损失
  - 基于 VGG19 多层特征
  - 可配置不同层的权重

- **[loss/HingeGANLoss.py](loss/HingeGANLoss.py)**：对抗训练损失
  - Hinge 距离的生成器和判别器损失

### 判别器
- **[Discriminator/UNetDiscriminatorSN.py](Discriminator/UNetDiscriminatorSN.py)**：U-Net 判别器
  - 基于 U-Net 架构的多尺度判别器
  - 使用谱归一化稳定训练

### 训练系统
- **[train.py](train.py)**：训练入口
  - 命令行参数解析
  - 支持权重加载、训练恢复
  - 自动恢复机制（auto_resume）

- **[train_config.py](train_config.py)**：配置管理
  - YAML 配置加载（扁平结构）
  - 数据路径、超参数、增强选项配置
  - 支持优化器、调度器、增强配置

- **[KawaiiTrainer.py](KawaiiTrainer.py)**：训练引擎
  - 完整的训练循环（多阶段支持）
  - 混合精度训练（FP16）
  - CUDA 异步预取优化
  - 指标评估（PSNR、SSIM、LPIPS）
  - 检查点保存策略（best/last）

### 数据和推理
- **[data_loader.py](data_loader.py)**：数据加载
  - 在线动态退化模式
  - 支持多种退化类型（模糊、噪声、压缩）
  - 高效的数据预处理

- **[run_inference.py](run_inference.py)**：推理脚本
  - 支持单张图片和批量推理
  - Tile 推理模式（低显存）

- **[demo_kawaii.py](demo_kawaii.py)**：交互式演示
  - 快速测试和可视化

- **[export_onnx.py](export_onnx.py)**：ONNX 导出
  - 模型转换为 ONNX 格式，支持跨平台推理

- **[onnx_superres.py](onnx_superres.py)**：ONNX 推理
  - 使用 ONNX Runtime 进行推理

### 配置文件
- **configs/** 目录
  - `anime_finetune.yaml`：用于动漫微调的配置
  - `real_stage1.yaml`、`real_stage2.yaml`、`real_stage3.yaml`：真实图像的多阶段训练配置
  - `quick_validate.yaml`：快速验证配置

## 🚀 快速开始

### 训练

#### 多阶段训练
```bash
# 阶段1：L1 预训练
python train.py --config configs/real_stage1.yaml \
                --train_dir /path/to/train \
                --val_dir /path/to/val \
                --ckpt_dir ./checkpoints_stage1

# 阶段2：GAN 对抗训练（从阶段1的最优权重开始）
python train.py --config configs/real_stage2.yaml \
                --train_dir /path/to/train \
                --val_dir /path/to/val \
                --ckpt_dir ./checkpoints_stage2 \
                --weights ./checkpoints_stage1/best_weights.pth

# 阶段3：Fine-tune
python train.py --config configs/real_stage3.yaml \
                --train_dir /path/to/train \
                --val_dir /path/to/val \
                --ckpt_dir ./checkpoints_stage3 \
                --weights ./checkpoints_stage2/best_weights.pth
```

### 推理
```bash
python run_inference.py --model_path ./checkpoints/best_weights.pth \
                        --input_dir ./inputs \
                        --output_dir ./outputs \
                        --scale 2
```

### 模型导出
```bash
python export_onnx.py --model_path ./checkpoints/best_weights.pth \
                       --output_path ./model.onnx
```

## 🎯 相比 HAT 的改进点

1. **动漫特化优化**
   - 集成动漫特化的感知损失（AnimePerceptualLoss）
   - 支持多种动漫风格数据的预训练

2. **综合损失设计**
   - 多目标损失权重优化
   - 频域约束（LaplacianLoss）减少伪影
   - Hinge GAN 用于更稳定的对抗训练

3. **训练策略改进**（参考 APISR）
   - **多阶段训练**：L1 预训练 → GAN 对抗 → Fine-tune

4. **代码易用性**
   - 扁平化 YAML 配置，易于理解和修改
   - 完整的日志和指标跟踪
   - 支持训练中断恢复


## 📚 参考论文

- **HAT**: [Activating More Pixels in Image Super-Resolution Transformer](https://arxiv.org/abs/2205.04437)
  - 混合注意力机制（通道注意力 + 窗口自注意力）
  - 像素激活策略

- **APISR**: [Anime Production Inspired Real-World Anime Super-Resolution](https://arxiv.org/abs/2403.01598)
  - 多阶段训练策略
  - 动漫特化数据策略
  - 在线动态退化


## 🤝 致谢

感谢 [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) 和 [Kiteretsu77/APISR](https://github.com/Kiteretsu77/APISR) 的开源工作。

