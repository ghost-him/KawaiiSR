## KawaiiSR 训练与推理说明书

> 以下内容使用 GPT-5 书写

本说明书面向想要使用本仓库进行 2x 超分辨率训练与推理的用户，涵盖：数据准备、配置说明、训练启动、恢复与监控、推理与常见问题排查。当前实现为“单阶段”最小化训练框架（旧的多阶段/阶段参数已移除）。

---
### 目录
1. 快速开始概览
2. 数据集准备
3. 配置文件 (YAML) 详解
4. 训练脚本参数
5. 开始训练 / 恢复训练示例
6. 日志与检查点结构
7. 推理（单图 / 批量）
8. 可选：分块推理 (Tiling) 与显存优化
9. 关键超参数影响说明
10. 常见问题 (FAQ)

---
### 1. 快速开始概览
```bash
# 1) 准备数据目录结构（见第 2 节）
# 2) 准备 YAML 配置（可参考 configs/quick_validate.yaml）
# 3) 启动训练
python models/train.py \
	--train_dir /path/to/train \
	--val_dir /path/to/validation \
	--ckpt_dir /path/to/save_ckpts \
	--config models/configs/quick_validate.yaml

# 4) 推理：使用生成的 best.pth 或 last.pth
python models/run_inference.py \
	--weights /path/to/save_ckpts/best.pth \
	--input /path/to/lr_image_or_dir \
	--output ./sr_outputs
```

---
### 2. 数据集准备（该内容可以由 `tools/sr-datagen` 自动化生成）
目录结构（训练与验证目录都相同模式）：
```
<split>/
	├─ dataset_index.csv
	├─ LR/  (低分辨率图像)
	└─ HR/  (高分辨率图像)
```
`dataset_index.csv` 至少需要两列（列名固定）：
```
lr_image_path,hr_image_path
0001.png,0001.png
0002.png,0002.png
...
```
其中每行的路径是相对各自 `LR/` 与 `HR/` 目录的文件名。程序会：
1. 读取 CSV；
2. 校验文件是否存在；
3. 过滤掉缺失的样本；
4. 使用 `transforms.ToTensor()+Normalize(mean=0.5,std=0.5)` 将像素归一化到 [-1,1]。

要求：HR 尺寸应是 LR 的 2 倍（当前模型内部固定 scale=2）。

---
### 3. 配置文件 (YAML) 详解
参考 `models/configs/quick_validate.yaml`。结构包含三个一级键：`train`, `loss_weights`, `gan`。

示例关键项说明：
| 项 | 作用 | 影响与建议 |
|----|------|------------|
| device | 设备 (cuda / cpu) | 自动 fallback；GPU 优先 |
| epochs | 总训练轮次 | 增大提升上限但耗时更久 |
| batch_size | 每批样本数 | 过大显存爆；过小统计不稳 |
| learning_rate | 初始学习率 | 过大震荡/发散；过小收敛慢 |
| mixed_precision | 是否开启 AMP | True 可降显存/提速；需 GPU 支持 |
| gradient_clip_norm | 梯度裁剪阈值 | 控制梯度爆炸；0 或 None 代表不裁剪 |
| num_workers | DataLoader 进程数 | SSD: 4~8；HDD: 2~4；CPU 核心多可适当调高 |
| pin_memory | DataLoader pin memory | GPU 训练建议开启 |
| val_every | 每隔多少 epoch 验证 | 较小可更快监控，稍增耗时 |
| log_every | 每隔多少 global step 写 TB | 太小 IO 频繁，太大曲线稀疏 |
| early_stopping_patience | 早停容忍 | 当前代码未实现早停逻辑，仅保留字段 |
| tensorboard_dir | TensorBoard 日志目录 | 用于可视化曲线 |
| checkpoint_dir | 模型权重保存目录 | 生成 last.pth / best.pth |

`loss_weights`：
| 键 | 含义 | 调大影响 |
|----|------|----------|
| pixel | L1/Charbonnier 主像素项 | 更锐或易过拟合细节 |
| perceptual | 感知/特征层损失 | 增强主观质量；过大偏离颜色 |
| frequency | 频域/Laplacian | 强化纹理；过大易噪点 |
| vgg | VGG 感知 | 同上（若与 perceptual 并存需平衡） |
| adversarial | 对抗项权重 | 提升质感；过大会产生伪影 |

`gan.enabled` 为 False 时不会实例化判别器，加快训练。

模型结构参数（在权重中保存）由 `train_config.get_default_model_config()` 定义：
| 参数 | 说明 |
|------|------|
| hat_body_hid_channels / hat_upsampler_hid_channels | 主干与上采样隐藏通道数 |
| hat_depths / hat_num_heads | Transformer 层数与多头数（可加深但显存线性增加） |
| hat_window_size | 窗口大小，影响内存与需要的 pad |
| use_tiling / tile_size / tile_pad | 是否启用推理分块与参数 |

---
### 4. 训练脚本参数
入口：`models/train.py`
```
--train_dir   训练数据根目录
--val_dir     验证数据根目录
--ckpt_dir    检查点输出目录（内层还会引用配置里的 checkpoint_dir）
--config      YAML 配置文件路径
--resume      （可选）已有 checkpoint（best/last）继续训练
```
注意：YAML 中的 `checkpoint_dir` 会覆盖命令行 `--ckpt_dir` 的作用范围（两者需保持一致以免混淆）。建议：命令行的 `--ckpt_dir` 与 YAML 里的 `train.checkpoint_dir` 设为相同路径。

---
### 5. 开始训练 / 恢复训练
```bash
# 正常训练
python models/train.py \
	--train_dir /data/SR/train \
	--val_dir /data/SR/validation \
	--ckpt_dir /data/SR/exp1_ckpts \
	--config models/configs/quick_validate.yaml

# 使用已有 best.pth 继续
python models/train.py \
	--train_dir /data/SR/train \
	--val_dir /data/SR/validation \
	--ckpt_dir /data/SR/exp1_ckpts \
	--config models/configs/quick_validate.yaml \
	--resume /data/SR/exp1_ckpts/best.pth

# 切换新实验（不同输出目录）
python models/train.py \
	--train_dir /data/SR/train \
	--val_dir /data/SR/validation \
	--ckpt_dir /data/SR/exp2_ckpts \
	--config models/configs/quick_validate.yaml
```

---
### 6. 日志与检查点结构
在 `checkpoint_dir` 下：
```
last.pth        # 最近一次保存的状态（含优化器、调度器、scaler、config）
best.pth        # 验证 PSNR 最优的完整状态
training.log    # 每个 epoch 的聚合日志
```
TensorBoard：位于 `tensorboard_dir`（如 `./models/logs/quick`），包含：
```
train/total_loss
train/loss_pixel, ...
train/psnr, train/ssim
val/loss, val/psnr, val/ssim
train/lr
```
查看：
```bash
tensorboard --logdir ./models/logs/quick --port 6006
```

`*.pth` 内容（字典）：
```
{
	epoch, global_step,
	model_state_dict,
	optimizer_state_dict,
	scheduler_state_dict,
	scaler_state_dict,
	best_metrics: {psnr, ssim, loss},
	config: <TrainingConfig.__dict__>
}
```

---
### 7. 推理
推理脚本：`models/run_inference.py`（已添加）。
```bash
# 单张 / 目录自动递归
python models/run_inference.py \
	--weights models/checkpoints/quick/best.pth \
	--input ./some_lr.png \
	--output ./sr_results

# 目录批处理 + 半精度 + 分块
python models/run_inference.py \
	--weights models/checkpoints/quick/best.pth \
	--input ./LR_folder \
	--output ./sr_results \
	--fp16 --use-tiling --tile-size 256 --tile-pad 32
```
输出文件命名：`<原文件名>_x2.png`（可用 `--prefix` 添加前缀）。

模型内部：
1. 输入张量使用与训练一致的归一化（0.5/0.5）。
2. 模型内部再做一次基于 `self.mean` 的中心化与缩放（保持训练一致）。
3. 输出自动映射回 [0,1] 并保存。

---
### 8. 分块推理 (Tiling) 与显存优化
适用于超大分辨率图：
| 参数 | 说明 | 建议 |
|------|------|------|
| --use-tiling / --no-tiling | 强制开或关 | 大图开，小图可关 |
| --tile-size | 单块尺寸（输入尺度） | 256~512 视显存而定 |
| --tile-pad | 重叠边界像素 | 16~64；增大可缓和接缝 |
| --fp16 | 开启 autocast 半精度 | 对 RTX 系列显卡友好 |

过小 tile 会导致速度变慢，过大可能 OOM。`tile_pad` 过小可能出现块状边缘，过大增加重复计算。

---
### 9. 关键超参数影响说明
| 类别 | 变量 | 影响 |
|------|------|------|
| 数据 | batch_size | 影响收敛稳定性与显存；小 batch 可适配显存但需调低 LR |
| 优化 | learning_rate | 决定初始收敛速度；配合 CosineAnnealing 调度平滑下降 |
| 优化 | gradient_clip_norm | 稳定训练，防止梯度爆炸；过低会限制学习 |
| 模型 | hat_depths/num_heads | 更深更多头 → 表达力↑ / 速度与显存成本↑ |
| 模型 | hat_window_size | 过大显存激增；过小限制感受野 |
| 推理 | use_tiling | 解决超大图 OOM；略降低吞吐 |
| 推理 | fp16 | 减少显存 + 提速；极少数情况下带来数值轻微差异 |
| 损失 | adversarial | 打开 GAN 可提升纹理；需更长调参周期 |

---
### 10. 常见问题 (FAQ)
1. 训练很慢：减少 `hat_depths` 长度或减小 `hat_body_hid_channels`；降低 batch；开启 AMP。
2. 显存不足：`--fp16` + 减小 batch；推理阶段用 `--use-tiling`。
3. 接缝/分块痕迹：适度增大 `--tile-pad`（32→48→64）。
4. 推理颜色偏差：确认未重复对输入做自定义归一化；脚本已内置正确流程。
5. 继续训练报错：旧权重不含优化器状态 → 可忽略优化器加载失败（代码已 try/except），继续训练即可。
6. 想改倍率：当前代码在 `KawaiiSR` 中 `self.scale = 2` 为硬编码；需手动修改并重新训练（含数据 HR 尺寸匹配）。

---

