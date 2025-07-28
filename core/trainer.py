"""
YOLO11 Training Module

This module provides comprehensive training capabilities for YOLO11 models,
including standard training, fine-tuning, and transfer learning.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, List, Callable
from pathlib import Path
import logging
import time
from datetime import datetime

from ultralytics import YOLO
from .model import YOLO11Model

logger = logging.getLogger(__name__)


class YOLO11Trainer:
    """
    Advanced trainer for YOLO11 models with support for:
    - Standard training from scratch
    - Fine-tuning pretrained models
    - Transfer learning
    - Custom training configurations
    - Training monitoring and logging
    """
    
    def __init__(
        self,
        model: Union[YOLO11Model, str, Path],
        device: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: YOLO11Model instance or path to model weights
            device: Device to train on ('cpu', 'cuda', 'mps', etc.)
            output_dir: Directory to save training outputs
        """
        self.device = device or self._get_default_device()
        self.output_dir = Path(output_dir) if output_dir else Path("experiments") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if isinstance(model, (str, Path)):
            self.model = YOLO11Model(model_path=model, device=self.device)
        elif isinstance(model, YOLO11Model):
            self.model = model
        else:
            raise ValueError("Model must be YOLO11Model instance or path to weights")
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        self.current_epoch = 0
        self.is_training = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"YOLO11Trainer initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_default_device(self) -> str:
        """Get default device based on availability."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _setup_logging(self):
        """Setup training-specific logging."""
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train(
        self,
        data: Union[str, Path],
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr: float = 0.01,
        patience: int = 100,
        save_period: int = -1,
        val: bool = True,
        plots: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the YOLO11 model.
        
        Args:
            data: Path to dataset configuration file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            lr: Learning rate
            patience: Early stopping patience
            save_period: Save checkpoint every n epochs (-1 to disable)
            val: Whether to validate during training
            plots: Whether to generate training plots
            **kwargs: Additional training arguments
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting YOLO11 training...")
        logger.info(f"Training parameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}")
        
        self.is_training = True
        self.current_epoch = 0
        
        # Prepare training arguments
        # Note: We don't pass 'device' here since it's already set in the model during initialization
        train_args = {
            'data': data,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr,
            'patience': patience,
            'save_period': save_period,
            'val': val,
            'plots': plots,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            **kwargs
        }
        
        try:
            # Start training
            start_time = time.time()
            results = self.model.train(**train_args)
            end_time = time.time()
            
            # Record training completion
            training_time = end_time - start_time
            self._record_training_completion(results, training_time)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
    
    def fine_tune(
        self,
        data: Union[str, Path],
        pretrained_path: Optional[Union[str, Path]] = None,
        freeze_layers: Optional[List[str]] = None,
        epochs: int = 50,
        lr: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune a pretrained YOLO11 model.
        
        Args:
            data: Path to fine-tuning dataset
            pretrained_path: Path to pretrained weights (if different from current model)
            freeze_layers: List of layer names to freeze during fine-tuning
            epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning (usually lower than training from scratch)
            **kwargs: Additional training arguments
        
        Returns:
            Fine-tuning results
        """
        logger.info("Starting YOLO11 fine-tuning...")
        
        # Load pretrained weights if specified
        if pretrained_path:
            logger.info(f"Loading pretrained weights from: {pretrained_path}")
            self.model.load(pretrained_path)
        
        # Freeze specified layers
        if freeze_layers:
            self._freeze_layers(freeze_layers)
        
        # Fine-tune with reduced learning rate
        fine_tune_args = {
            'data': data,
            'epochs': epochs,
            'lr0': lr,
            'patience': 50,  # Reduced patience for fine-tuning
            'save_period': 10,
            'val': True,
            'plots': True,
            **kwargs
        }
        
        return self.train(**fine_tune_args)
    
    def transfer_learn(
        self,
        source_data: Union[str, Path],
        target_data: Union[str, Path],
        source_epochs: int = 50,
        target_epochs: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform transfer learning: train on source domain, then fine-tune on target.
        
        Args:
            source_data: Source domain dataset
            target_data: Target domain dataset
            source_epochs: Epochs for source domain training
            target_epochs: Epochs for target domain fine-tuning
            **kwargs: Additional arguments
        
        Returns:
            Transfer learning results
        """
        logger.info("Starting transfer learning...")
        
        # Phase 1: Train on source domain
        logger.info("Phase 1: Training on source domain")
        source_results = self.train(
            data=source_data,
            epochs=source_epochs,
            **kwargs
        )
        
        # Phase 2: Fine-tune on target domain
        logger.info("Phase 2: Fine-tuning on target domain")
        target_results = self.fine_tune(
            data=target_data,
            epochs=target_epochs,
            freeze_layers=['backbone.conv1', 'backbone.layer1'],  # Example layer freezing
            **kwargs
        )
        
        return {
            'source_results': source_results,
            'target_results': target_results,
            'transfer_learning_summary': self._create_transfer_summary(source_results, target_results)
        }
    
    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional training arguments
        
        Returns:
            Resumed training results
        """
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        self.model.load(checkpoint_path)
        
        # Resume training
        # Note: We don't pass 'device' here since it's already set in the model during initialization
        resume_args = {
            'resume': True,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            **kwargs
        }
        
        return self.model.train(**resume_args)
    
    def validate(
        self,
        data: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the model.
        
        Args:
            data: Validation dataset path
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results
        """
        logger.info("Running model validation...")
        
        # Note: We don't pass 'device' here since it's already set in the model during initialization
        val_args = {
            'data': data,
            **kwargs
        }
        
        return self.model.val(**val_args)
    
    def _freeze_layers(self, layer_names: List[str]):
        """
        Freeze specified layers for fine-tuning.
        
        Args:
            layer_names: List of layer names to freeze
        """
        if not hasattr(self.model.model, 'model'):
            logger.warning("Cannot freeze layers: model structure not accessible")
            return
        
        model = self.model.model.model
        frozen_count = 0
        
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    logger.debug(f"Frozen parameter: {name}")
        
        logger.info(f"Frozen {frozen_count} parameters in {len(layer_names)} layer groups")
    
    def _record_training_completion(self, results: Any, training_time: float):
        """Record training completion information."""
        completion_info = {
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'final_epoch': getattr(results, 'epoch', 'unknown'),
            'device_used': self.device,
            'output_directory': str(self.output_dir)
        }
        
        # Extract metrics if available
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                completion_info['final_mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                completion_info['final_mAP50'] = float(results.box.map50)
        
        self.training_history.append(completion_info)
        
        # Save training summary
        self._save_training_summary(completion_info)
    
    def _create_transfer_summary(self, source_results: Any, target_results: Any) -> Dict[str, Any]:
        """Create transfer learning summary."""
        summary = {
            'transfer_learning_completed': True,
            'source_domain_training': 'completed',
            'target_domain_fine_tuning': 'completed'
        }
        
        # Add metrics comparison if available
        if hasattr(source_results, 'box') and hasattr(target_results, 'box'):
            if hasattr(source_results.box, 'map') and hasattr(target_results.box, 'map'):
                summary.update({
                    'source_mAP50-95': float(source_results.box.map),
                    'target_mAP50-95': float(target_results.box.map),
                    'improvement': float(target_results.box.map) - float(source_results.box.map)
                })
        
        return summary
    
    def _save_training_summary(self, completion_info: Dict[str, Any]):
        """Save training summary to file."""
        summary_file = self.output_dir / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in completion_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nModel info:\n")
            model_info = self.model.get_model_info()
            for key, value in model_info.items():
                f.write(f"  {key}: {value}\n")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'output_directory': str(self.output_dir),
            'device': self.device,
            'training_history_count': len(self.training_history),
            'model_info': self.model.get_model_info()
        }
    
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None):
        """Save training checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        
        self.model.save(path)
        logger.info(f"Checkpoint saved to: {path}")
    
    def export_model(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> str:
        """
        Export trained model to different formats.
        
        Args:
            format: Export format ('onnx', 'tensorrt', 'coreml', etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        export_path = self.model.export(format=format, **kwargs)
        logger.info(f"Model exported to: {export_path}")
        return export_path
    
    def __repr__(self) -> str:
        return (f"YOLO11Trainer(model={self.model.task}, device={self.device}, "
                f"output_dir={self.output_dir}, is_training={self.is_training})")


class TrainingConfig:
    """Configuration class for YOLO11 training."""
    
    def __init__(self):
        # Basic training parameters
        self.epochs = 100
        self.batch_size = 16
        self.image_size = 640
        self.learning_rate = 0.01
        self.patience = 100
        
        # Data parameters
        self.data_path = None
        self.workers = 8
        self.cache = False
        
        # Optimization parameters
        self.optimizer = 'auto'
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.warmup_epochs = 3
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1
        
        # Augmentation parameters
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = 0.5
        self.shear = 0.0
        self.perspective = 0.0
        self.flipud = 0.0
        self.fliplr = 0.5
        self.mosaic = 1.0
        self.mixup = 0.0
        self.copy_paste = 0.0
        
        # Validation parameters
        self.val = True
        self.split = 'val'
        self.save_json = False
        
        # Logging parameters
        self.verbose = True
        self.plots = True
        self.save = True
        self.save_period = -1
        
        # Device parameters
        self.device = None
        self.amp = True
        self.fraction = 1.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self, path: Union[str, Path]):
        """Save config to file."""
        import json
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_config(self, path: Union[str, Path]):
        """Load config from file."""
        import json
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)


class MultiGPUTrainer(YOLO11Trainer):
    """
    Multi-GPU trainer for distributed training.
    """
    
    def __init__(
        self,
        model: Union[YOLO11Model, str, Path],
        device_ids: Optional[List[int]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize multi-GPU trainer.
        
        Args:
            model: YOLO11Model instance or path to model weights
            device_ids: List of GPU device IDs to use
            output_dir: Directory to save training outputs
        """
        # Initialize with first GPU as primary device
        primary_device = f"cuda:{device_ids[0]}" if device_ids else "cuda:0"
        super().__init__(model, device=primary_device, output_dir=output_dir)
        
        self.device_ids = device_ids or [0]
        if len(self.device_ids) > 1:
            logger.info(f"Multi-GPU training enabled with devices: {self.device_ids}")
        
    def train(
        self,
        data: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model using multiple GPUs.
        
        Args:
            data: Path to dataset configuration file
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        # Set device for multi-GPU training
        if len(self.device_ids) > 1:
            device_str = ','.join(map(str, self.device_ids))
            kwargs['device'] = device_str
            logger.info(f"Starting multi-GPU training on devices: {device_str}")
        
        return super().train(data=data, **kwargs)


class TrainingCallbacks:
    """Callback system for training monitoring."""
    
    def __init__(self):
        self.callbacks = {
            'on_train_start': [],
            'on_train_end': [],
            'on_epoch_start': [],
            'on_epoch_end': [],
            'on_batch_start': [],
            'on_batch_end': [],
            'on_validation_start': [],
            'on_validation_end': []
        }
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a specific event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for a specific event."""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback error in {event}: {e}")


def create_trainer(
    model_type: str = 'detect',
    model_size: str = 'n',
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    multi_gpu: bool = False,
    device_ids: Optional[List[int]] = None
) -> YOLO11Trainer:
    """
    Factory function to create YOLO11 trainers.
    
    Args:
        model_type: Type of model ('detect', 'segment', 'classify', 'pose', 'obb')
        model_size: Size of model ('n', 's', 'm', 'l', 'x')
        device: Device to use for training
        output_dir: Output directory for training results
        multi_gpu: Whether to use multi-GPU training
        device_ids: List of GPU device IDs for multi-GPU training
    
    Returns:
        YOLO11Trainer instance
    """
    # Create model
    model = YOLO11Model(task=model_type, size=model_size, device=device)
    
    # Create appropriate trainer
    if multi_gpu and device_ids and len(device_ids) > 1:
        trainer = MultiGPUTrainer(
            model=model,
            device_ids=device_ids,
            output_dir=output_dir
        )
    else:
        trainer = YOLO11Trainer(
            model=model,
            device=device,
            output_dir=output_dir
        )
    
    return trainer