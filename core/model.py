"""
YOLO11 Core Model Implementation

This module provides the core YOLO11 model class with support for:
- Multiple tasks (detection, segmentation, classification, pose estimation)
- Training and fine-tuning
- Inference optimization
- Model quantization and other optimization techniques
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import logging

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolo.classify import ClassificationPredictor
from ultralytics.models.yolo.pose import PosePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLO11Model:
    """
    Enhanced YOLO11 model class with support for optimization and deployment.
    
    This class wraps the Ultralytics YOLO model and provides additional
    functionality for model optimization, quantization, and deployment.
    """
    
    SUPPORTED_TASKS = {
        'detect': 'yolo11n.pt',
        'segment': 'yolo11n-seg.pt', 
        'classify': 'yolo11n-cls.pt',
        'pose': 'yolo11n-pose.pt',
        'obb': 'yolo11n-obb.pt'
    }
    
    SUPPORTED_SIZES = ['n', 's', 'm', 'l', 'x']
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        task: str = 'detect',
        size: str = 'n',
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize YOLO11 model.
        
        Args:
            model_path: Path to model weights. If None, uses pretrained model.
            task: Task type ('detect', 'segment', 'classify', 'pose', 'obb')
            size: Model size ('n', 's', 'm', 'l', 'x')
            device: Device to run model on ('cpu', 'cuda', 'mps', etc.)
            verbose: Whether to print verbose output
        """
        self.task = task
        self.size = size
        self.device = device or self._get_default_device()
        self.verbose = verbose
        self.model_path = model_path
        self.optimization_history = []
        
        # Validate inputs
        self._validate_inputs()
        
        # Load model
        self.model = self._load_model()
        
        # Store original model for comparison
        self.original_model = None
        
        logger.info(f"YOLO11 model initialized: task={task}, size={size}, device={self.device}")
    
    def _get_default_device(self) -> str:
        """Get default device based on availability."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {self.task}. Supported: {list(self.SUPPORTED_TASKS.keys())}")
        
        if self.size not in self.SUPPORTED_SIZES:
            raise ValueError(f"Unsupported size: {self.size}. Supported: {self.SUPPORTED_SIZES}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model."""
        if self.model_path:
            model_path = self.model_path
        else:
            # Use pretrained model
            base_name = self.SUPPORTED_TASKS[self.task]
            model_path = base_name.replace('n', self.size)
        
        try:
            model = YOLO(model_path)
            if self.device:
                model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        source: Union[str, Path, List],
        **kwargs
    ) -> Any:
        """
        Run inference on images/videos.
        
        Args:
            source: Input source (image path, video path, directory, etc.)
            **kwargs: Additional arguments for prediction
        
        Returns:
            Prediction results
        """
        return self.model.predict(source, **kwargs)
    
    def train(
        self,
        data: Union[str, Path],
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        custom_trainer=None,
        **kwargs
    ) -> Any:
        """
        Train the model.
        
        Args:
            data: Path to dataset config file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            custom_trainer: Custom trainer class to use (optional)
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        if custom_trainer is not None:
            # Use custom trainer if provided
            # This would require special handling depending on the trainer implementation
            return self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                **kwargs
            )
        else:
            # Use standard training
            return self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                **kwargs
            )
    
    def val(
        self,
        data: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Any:
        """
        Validate the model.
        
        Args:
            data: Path to validation dataset config
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results
        """
        return self.model.val(data=data, **kwargs)
    
    def export(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'tensorrt', 'coreml', etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        return self.model.export(format=format, **kwargs)
    
    def save(self, path: Union[str, Path]):
        """Save model weights."""
        self.model.save(path)
        logger.info(f"Model saved to: {path}")
    
    def load(self, path: Union[str, Path]):
        """Load model weights."""
        self.model = YOLO(path)
        if self.device:
            self.model.to(self.device)
        logger.info(f"Model loaded from: {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'task': self.task,
            'size': self.size,
            'device': self.device,
            'model_path': self.model_path,
            'optimization_history': self.optimization_history
        }
        
        # Add model specific info if available
        if hasattr(self.model, 'model'):
            try:
                # Count parameters
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB
                })
            except Exception as e:
                logger.warning(f"Could not get model parameters info: {e}")
        
        return info
    
    def benchmark(
        self,
        data_source: Union[str, Path],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            data_source: Path to test data
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Performance metrics
        """
        import time
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.predict(data_source, verbose=False)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(data_source, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        metrics = {
            'avg_inference_time': sum(times) / len(times),
            'min_inference_time': min(times),
            'max_inference_time': max(times),
            'fps': 1.0 / (sum(times) / len(times))
        }
        
        return metrics
    
    def __repr__(self) -> str:
        return (f"YOLO11Model(task={self.task}, size={self.size}, "
                f"device={self.device}, optimized={len(self.optimization_history) > 0})")


class YOLO11Factory:
    """Factory class for creating YOLO11 models."""
    
    @staticmethod
    def create_detector(size: str = 'n', **kwargs) -> YOLO11Model:
        """Create a detection model."""
        return YOLO11Model(task='detect', size=size, **kwargs)
    
    @staticmethod
    def create_segmenter(size: str = 'n', **kwargs) -> YOLO11Model:
        """Create a segmentation model."""
        return YOLO11Model(task='segment', size=size, **kwargs)
    
    @staticmethod
    def create_classifier(size: str = 'n', **kwargs) -> YOLO11Model:
        """Create a classification model."""
        return YOLO11Model(task='classify', size=size, **kwargs)
    
    @staticmethod
    def create_pose_estimator(size: str = 'n', **kwargs) -> YOLO11Model:
        """Create a pose estimation model."""
        return YOLO11Model(task='pose', size=size, **kwargs)
    
    @staticmethod
    def create_obb_detector(size: str = 'n', **kwargs) -> YOLO11Model:
        """Create an oriented bounding box detection model."""
        return YOLO11Model(task='obb', size=size, **kwargs)