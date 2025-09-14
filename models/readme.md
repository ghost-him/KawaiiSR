```
# 从头开始训练
python train.py --data_path /path/to/data --config config.yaml

# 从检查点恢复训练
python train.py --resume /path/to/checkpoint.pth

# 训练特定阶段
python train.py --stage stage2 --resume /path/to/stage1_best.pth

# 自动恢复最新检查点
python train.py --auto_resume --data_path /path/to/data
```