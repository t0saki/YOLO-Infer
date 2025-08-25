# Error Handling in YOLO-Infer

This document explains how error handling works in the YOLO-Infer project, particularly for handling shape mismatch errors during training.

## Overview

The YOLO-Infer project includes robust error handling mechanisms to ensure training can continue even when individual batches cause issues. This is especially useful for handling `shape mismatch` errors that can occur during training.

## Key Components

### 1. RobustYOLO11Trainer

The `RobustYOLO11Trainer` class in `core/robust_trainer.py` provides enhanced error handling for YOLO11 model training.

Key features:
- Handles shape mismatch errors gracefully
- Continues training even when individual batches fail
- Logs detailed information about skipped batches
- Uses the `--skip-errors` flag by default in the CLI

### 2. BatchErrorSkippingTrainer

The `BatchErrorSkippingTrainer` class extends Ultralytics' `BaseTrainer` to provide batch-level error handling:

- Overrides the training loop to catch exceptions
- Specifically handles `shape mismatch` errors
- Skips problematic batches and continues training
- Tracks the number of skipped batches

## Usage

### Command Line Interface

When using the main CLI, error skipping is enabled by default:

```bash
# Error skipping is enabled by default
python main.py train --data dataset.yaml --epochs 100

# Explicitly enable error skipping (default behavior)
python main.py train --data dataset.yaml --epochs 100 --skip-errors

# Disable error skipping (training stops on any error)
python main.py train --data dataset.yaml --epochs 100 --no-skip-errors
```

### Programmatic Usage

```python
from core.robust_trainer import RobustYOLO11Trainer, create_robust_trainer

# Create a robust trainer
trainer = create_robust_trainer(
    model_type='detect',
    model_size='n',
    device='cuda'
)

# Train with error handling enabled (default)
results = trainer.train(
    data='dataset.yaml',
    epochs=100,
    skip_errors=True  # Enable error skipping
)
```

## Error Handling Behavior

When a shape mismatch error occurs:

1. The error is logged with details about the problematic batch
2. The batch is skipped without updating model weights
3. Training continues with the next batch
4. The total number of skipped batches is tracked and reported
5. Training completes normally, with information about skipped batches in the logs

Example log output:
```
ERROR - Shape mismatch error detected: shape mismatch: value tensor of shape [49750] cannot be broadcast to indexing result of shape [77395]
INFO - Skipping problematic batch and continuing training...
```

## Benefits

- **Resilience**: Training continues even when individual batches fail
- **Productivity**: No need to restart training from scratch due to intermittent errors
- **Transparency**: Detailed logging of skipped batches for analysis
- **Compatibility**: Works with existing Ultralytics YOLO training workflows

## Limitations

- Only shape mismatch errors are automatically skipped
- Other types of errors will still stop training
- Performance may be slightly reduced due to error handling overhead