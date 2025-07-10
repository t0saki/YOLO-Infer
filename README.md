# YOLO11 Project

A comprehensive YOLO11 deep learning platform supporting training, fine-tuning, quantization, and deployment with advanced optimization techniques.

## Features

- **Multi-task Support**: Object detection, segmentation, classification, pose estimation, and oriented bounding box detection
- **Complete Training Pipeline**: Full training, fine-tuning, and transfer learning capabilities
- **Advanced Optimization**: Modular optimization framework with quantization, pruning, and knowledge distillation
- **Performance Benchmarking**: Comprehensive speed and accuracy benchmarking tools
- **Interactive Demos**: Real-time inference on images, videos, and webcam streams
- **Flexible Configuration**: YAML-based configuration system
- **Multi-GPU Support**: Distributed training and inference

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/t0saki/YOLO-Infer.git
cd YOLO-Infer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Run Detection Demo
```bash
# Image detection
python main.py demo --input image.jpg

# Video detection
python main.py demo --input video.mp4

# Webcam detection
python main.py demo --input webcam
```

#### Train a Model
```bash
# Train on COCO dataset
python main.py train --data coco8.yaml --epochs 100

# Fine-tune existing model
python main.py train --model yolo11n.pt --data custom.yaml --fine-tune
```

#### Validate Model
```bash
python main.py val --model yolo11n.pt --data coco8.yaml
```

#### Optimize Model
```bash
# Dynamic quantization
python main.py optimize --model yolo11n.pt --method dynamic

# Post-training quantization
python main.py optimize --model yolo11n.pt --method ptq --calibration-data val_data.yaml
```

#### Benchmark Performance
```bash
# Benchmark different model sizes
python main.py benchmark --type model_sizes

# Benchmark quantization methods
python main.py benchmark --type quantization

# Throughput benchmark
python main.py benchmark --type throughput --duration 60
```

## Project Structure

```
YOLO-Infer/
├── core/                    # Core functionality
│   ├── model.py            # YOLO11 model wrapper
│   ├── trainer.py          # Training and fine-tuning
│   └── validator.py        # Model validation
├── optimization/           # Model optimization
│   ├── base.py            # Abstract optimization classes
│   ├── quantization/      # Quantization methods
│   ├── pruning/           # Model pruning (extensible)
│   └── distillation/      # Knowledge distillation (extensible)
├── demos/                 # Demo applications
│   └── detection_demo.py  # Object detection demo
├── benchmarks/           # Performance benchmarking
│   └── speed_benchmark.py # Speed and efficiency tests
├── utils/                # Utility functions
│   ├── visualization.py  # Visualization tools
│   ├── data_loader.py    # Data loading utilities
│   └── helpers.py        # General helper functions
├── configs/              # Configuration files
│   └── default.yaml     # Default configuration
├── requirements.txt      # Python dependencies
└── main.py              # Main CLI interface
```

## Configuration

The project uses YAML configuration files for flexible setup. See `configs/default.yaml` for all available options:

```yaml
# Model configuration
model:
  task: 'detect'
  size: 'n'
  device: null

# Training configuration
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  
# Optimization configuration
optimization:
  quantization:
    backend: 'fbgemm'
    dtype: 'qint8'
```

## Supported Tasks

- **Object Detection**: Detect and locate objects in images
- **Instance Segmentation**: Detect objects with pixel-level masks
- **Image Classification**: Classify entire images
- **Pose Estimation**: Detect human keypoints and poses
- **Oriented Object Detection**: Detect rotated objects

## Optimization Methods

### Quantization
- **Dynamic Quantization**: Runtime quantization without calibration
- **Post-Training Quantization (PTQ)**: Static quantization with calibration
- **Quantization-Aware Training (QAT)**: Training with fake quantization

### Extensible Framework
The optimization framework is designed for easy extension:
- Modular base classes for new optimization methods
- Plugin-style architecture for optimization techniques
- Comprehensive evaluation and comparison tools

## Examples

### Advanced Training
```python
from core.trainer import YOLO11Trainer
from core.model import YOLO11Model

# Create model and trainer
model = YOLO11Model(task='detect', size='s')
trainer = YOLO11Trainer(model, device='cuda')

# Train with custom configuration
results = trainer.train(
    data='custom_dataset.yaml',
    epochs=200,
    batch=32,
    lr=0.001
)
```

### Model Optimization
```python
from optimization.quantization.quantizers import create_quantizer

# Load model
model = YOLO11Model('yolo11n.pt')

# Create quantizer
quantizer = create_quantizer('dynamic', model)

# Optimize model
optimized_model = quantizer.optimize()

# Compare performance
comparison = quantizer.compare_models(test_data)
```

### Custom Validation
```python
from core.validator import YOLO11Validator

# Create validator
validator = YOLO11Validator('model.pt')

# Run comprehensive validation
results = validator.validate('test_data.yaml')

# Benchmark speed
speed_results = validator.benchmark_speed(test_data)

# Cross-validation
cv_results = validator.cross_validate('dataset.yaml', k_folds=5)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLO
- OpenCV
- NumPy
- See `requirements.txt` for complete list

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For questions and support:
- Check the documentation
- Open an issue on GitHub
- Review example configurations in `configs/`

## Acknowledgments

- Ultralytics for the YOLO implementation
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools