# 断点续训功能说明

本项目实现了训练、量化感知训练(QAT)和微调流程的断点续训功能，允许在训练过程中保存检查点，并在需要时从中断的地方继续训练。

## 功能特性

1. **训练过程中的检查点保存**：支持在训练过程中定期保存模型状态
2. **断点续训**：可以从保存的检查点恢复训练
3. **统一的检查点格式**：所有训练流程使用一致的检查点格式
4. **灵活的检查点管理**：支持列出、查看、清理检查点

## 使用方法

### 1. 训练时启用检查点保存

```bash
# 基本训练，每1个epoch保存一次检查点
python main.py train --data dataset.yaml --checkpoint-period 1

# 每5个epoch保存一次检查点
python main.py train --data dataset.yaml --checkpoint-period 5
```

### 2. 从检查点恢复训练

```bash
# 自动从最新检查点恢复
python main.py train --data dataset.yaml --resume

# 或者在QAT中恢复
python main.py optimize --model model.pt --method qat --resume --checkpoint-period 2
```

### 3. 微调时使用检查点

```bash
# 微调模型并启用检查点
python main.py train --data dataset.yaml --fine-tune --checkpoint-period 1
```

## 检查点管理命令

训练器还提供了额外的检查点管理功能：

```python
# 列出所有检查点
checkpoints = trainer.list_checkpoints()

# 获取检查点详细信息
info = trainer.get_checkpoint_info(checkpoint_path)

# 清理旧检查点，只保留最新的3个
trainer.cleanup_checkpoints(keep_last_n=3)

# 删除特定检查点
trainer.remove_checkpoint(checkpoint_path)
```

## 技术实现

### 检查点格式

所有检查点采用统一格式，包含：
- `epoch`: 当前训练轮次
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态（如果提供）
- `metrics`: 训练指标
- `timestamp`: 保存时间戳

### 目录结构

```
experiments/
├── train_20250728_235959/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_2.pt
│   │   └── ...
│   └── ...
```

## 注意事项

1. 检查点文件包含完整的模型状态和优化器状态，文件较大
2. 建议定期清理旧检查点以节省磁盘空间
3. 恢复训练时确保使用相同的模型架构和训练配置