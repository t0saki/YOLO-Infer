"""
Quantization optimizers for YOLO11 models.

This module implements various quantization techniques including:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Dynamic quantization
"""

import torch
import torch.nn as nn
import torch.quantization as torch_quantization
from torch.quantization import QuantStub, DeQuantStub
from typing import Any, Dict, Optional, List, Callable
import copy
import logging
from pathlib import Path

from ..base import QuantizationOptimizer

logger = logging.getLogger(__name__)

class PostTrainingQuantizer(QuantizationOptimizer):
    """
    Post-Training Quantization (PTQ) optimizer.
    
    This quantizer performs static quantization using calibration data
    to determine optimal quantization parameters.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, config, device)
        
        # PTQ specific configurations
        self.num_calibration_batches = self.config.get('num_calibration_batches', 100)
        self.quantization_backend = self.config.get('backend', 'qnnpack')
        self.quantization_dtype = self.config.get('dtype', torch.qint8)
        
        # Set quantization backend
        torch.backends.quantized.engine = self.quantization_backend
        
    def optimize(self, calibration_loader: Any = None, **kwargs) -> Any:
        """
        Perform post-training quantization.
        
        Args:
            calibration_loader: Data loader for calibration
            **kwargs: Additional arguments
            
        Returns:
            Quantized model
        """
        if calibration_loader is None and self.calibration_data is None:
            raise ValueError("Calibration data is required for post-training quantization")
        
        calibration_loader = calibration_loader or self.calibration_data
        
        logger.info("Starting post-training quantization...")
        
        # Prepare model for quantization
        model_to_quantize = self._prepare_model_for_quantization()
        
        # Set to evaluation mode
        model_to_quantize.eval()
        
        # Calibrate the model
        quantized_model = self._calibrate_model(model_to_quantize, calibration_loader)
        
        # Convert to quantized model
        self.optimized_model = torch.quantization.convert(quantized_model)
        
        # Record optimization metrics
        self._record_optimization_metrics()
        
        logger.info("Post-training quantization completed")
        return self.optimized_model
    
    def _prepare_model_for_quantization(self) -> Any:
        """Prepare model for quantization."""
        # Create a copy of the model
        model_copy = copy.deepcopy(self.original_model)
        
        # Get the underlying PyTorch model
        if hasattr(model_copy, 'model'):
            pytorch_model = model_copy.model
        else:
            pytorch_model = model_copy
        
        # Set quantization configuration
        if self.quantization_backend == 'fbgemm':
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif self.quantization_backend == 'qnnpack':
            qconfig = torch.quantization.get_default_qconfig('qnnpack')
        else:
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        pytorch_model.qconfig = qconfig
        
        # Prepare for quantization
        prepared_model = torch.quantization.prepare(pytorch_model)
        
        return prepared_model
    
    def _calibrate_model(self, model: Any, calibration_loader: Any) -> Any:
        """Calibrate the model using calibration data."""
        logger.info("Calibrating model...")
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= self.num_calibration_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Move to device
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                
                # Forward pass for calibration
                try:
                    _ = model(images)
                except Exception as e:
                    logger.warning(f"Calibration batch {i} failed: {e}")
                    continue
                
                if i % 20 == 0:
                    logger.info(f"Calibration progress: {i}/{self.num_calibration_batches}")
        
        logger.info("Model calibration completed")
        return model
    
    def evaluate(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the quantized model."""
        if self.optimized_model is None:
            raise ValueError("No optimized model to evaluate. Run optimize() first.")
        
        # Use the model's validation method if available
        if hasattr(self.optimized_model, 'val'):
            results = self.optimized_model.val(data=test_data)
            return self._extract_metrics_from_results(results)
        
        # Fallback to basic evaluation
        return self._basic_evaluation(test_data, metrics)
    
    def _basic_evaluation(self, test_data: Any, metrics: Optional[List[str]]) -> Dict[str, float]:
        """Basic evaluation implementation."""
        # This is a simplified evaluation
        # In practice, you would implement proper metric computation
        return {
            'accuracy': 0.0,
            'inference_time': 0.0,
            'model_size': self._get_model_size()
        }
    
    def _extract_metrics_from_results(self, results: Any) -> Dict[str, float]:
        """Extract metrics from YOLO validation results."""
        metrics = {}
        
        # Extract common YOLO metrics
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                metrics['mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                metrics['mAP50'] = float(results.box.map50)
            if hasattr(results.box, 'map75'):
                metrics['mAP75'] = float(results.box.map75)
        
        # Add model size
        metrics['model_size_mb'] = self._get_model_size()
        
        return metrics
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if self.optimized_model is None:
            return 0.0
        
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        if hasattr(self.optimized_model, 'model'):
            model = self.optimized_model.model
        else:
            model = self.optimized_model
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def _record_optimization_metrics(self):
        """Record optimization metrics."""
        self.optimization_metrics = {
            'optimization_type': 'post_training_quantization',
            'backend': self.quantization_backend,
            'dtype': str(self.quantization_dtype),
            'num_calibration_batches': self.num_calibration_batches,
            'model_size_mb': self._get_model_size()
        }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information."""
        return {
            'optimizer_type': 'PostTrainingQuantizer',
            'config': self.config,
            'metrics': self.optimization_metrics,
            'quantization_backend': self.quantization_backend,
            'quantization_dtype': str(self.quantization_dtype)
        }


class DynamicQuantizer(QuantizationOptimizer):
    """
    Dynamic Quantization optimizer.
    
    This quantizer performs dynamic quantization without requiring
    calibration data. It quantizes weights statically but activations
    dynamically during inference.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, config, device)
        
        # Dynamic quantization specific configurations
        self.qconfig_dict = self.config.get('qconfig_dict', None)
        self.dtype = self.config.get('dtype', torch.qint8)
        
    def optimize(self, **kwargs) -> Any:
        """
        Perform dynamic quantization.
        
        Returns:
            Dynamically quantized model
        """
        logger.info("Starting dynamic quantization...")
        
        # Get the underlying PyTorch model
        if hasattr(self.original_model, 'model'):
            pytorch_model = self.original_model.model
        else:
            pytorch_model = self.original_model
        
        # Perform dynamic quantization
        self.optimized_model = torch.quantization.quantize_dynamic(
            pytorch_model,
            qconfig_spec=self.qconfig_dict,
            dtype=self.dtype
        )
        
        # Wrap back if needed
        if hasattr(self.original_model, 'model'):
            # Create a new instance with quantized model
            quantized_wrapper = copy.deepcopy(self.original_model)
            quantized_wrapper.model = self.optimized_model
            self.optimized_model = quantized_wrapper
        
        # Record optimization metrics
        self._record_optimization_metrics()
        
        logger.info("Dynamic quantization completed")
        return self.optimized_model
    
    def _prepare_model_for_quantization(self) -> Any:
        """Dynamic quantization doesn't need preparation."""
        return self.original_model
    
    def _calibrate_model(self, model: Any) -> Any:
        """Dynamic quantization doesn't need calibration."""
        return model
    
    def evaluate(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the dynamically quantized model."""
        if self.optimized_model is None:
            raise ValueError("No optimized model to evaluate. Run optimize() first.")
        
        # Use similar evaluation as PTQ
        if hasattr(self.optimized_model, 'val'):
            results = self.optimized_model.val(data=test_data)
            return self._extract_metrics_from_results(results)
        
        return self._basic_evaluation(test_data, metrics)
    
    def _extract_metrics_from_results(self, results: Any) -> Dict[str, float]:
        """Extract metrics from YOLO validation results."""
        # Same implementation as PTQ
        metrics = {}
        
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                metrics['mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                metrics['mAP50'] = float(results.box.map50)
            if hasattr(results.box, 'map75'):
                metrics['mAP75'] = float(results.box.map75)
        
        metrics['model_size_mb'] = self._get_model_size()
        return metrics
    
    def _basic_evaluation(self, test_data: Any, metrics: Optional[List[str]]) -> Dict[str, float]:
        """Basic evaluation implementation."""
        return {
            'accuracy': 0.0,
            'inference_time': 0.0,
            'model_size': self._get_model_size()
        }
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if self.optimized_model is None:
            return 0.0
        
        param_size = 0
        buffer_size = 0
        
        if hasattr(self.optimized_model, 'model'):
            model = self.optimized_model.model
        else:
            model = self.optimized_model
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def _record_optimization_metrics(self):
        """Record optimization metrics."""
        self.optimization_metrics = {
            'optimization_type': 'dynamic_quantization',
            'dtype': str(self.dtype),
            'model_size_mb': self._get_model_size()
        }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information."""
        return {
            'optimizer_type': 'DynamicQuantizer',
            'config': self.config,
            'metrics': self.optimization_metrics,
            'dtype': str(self.dtype)
        }


class QATQuantizer(QuantizationOptimizer):
    """
    Quantization-Aware Training (QAT) optimizer.
    
    This quantizer fine-tunes a model with fake quantization operations
    to achieve better accuracy after quantization.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, config, device)
        
        # QAT specific configurations
        self.num_epochs = self.config.get('num_epochs', 10)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.quantization_backend = self.config.get('backend', 'qnnpack')

        # Set quantization backend
        torch.backends.quantized.engine = self.quantization_backend
    
    def optimize(self, train_loader: Any = None, **kwargs) -> Any:
        """
        Perform quantization-aware training.
        
        Args:
            train_loader: Training data loader
            **kwargs: Additional arguments
            
        Returns:
            QAT trained and quantized model
        """
        if train_loader is None:
            raise ValueError("Training data loader is required for QAT")
        
        logger.info("Starting quantization-aware training...")
        
        # Prepare model for QAT
        model_to_train = self._prepare_model_for_quantization()
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=self.learning_rate)
        
        # Training loop
        model_to_train.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images, targets = batch[0], batch[1]
                else:
                    images = batch
                    targets = None
                
                # Move to device
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model_to_train(images)
                
                # Compute loss (simplified - in practice use proper YOLO loss)
                if targets is not None:
                    loss = self._compute_loss(outputs, targets)
                else:
                    # Dummy loss for demonstration
                    loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed, Average Loss: {avg_loss:.4f}")
        
        # Convert to quantized model
        model_to_train.eval()
        self.optimized_model = torch.quantization.convert(model_to_train)
        
        # Record optimization metrics
        self._record_optimization_metrics()
        
        logger.info("Quantization-aware training completed")
        return self.optimized_model
    
    def _prepare_model_for_quantization(self) -> Any:
        """Prepare model for QAT."""
        # Create a copy of the model
        model_copy = copy.deepcopy(self.original_model)
        
        # Get the underlying PyTorch model
        if hasattr(model_copy, 'model'):
            pytorch_model = model_copy.model
        else:
            pytorch_model = model_copy
        
        # Set quantization configuration for QAT
        if self.quantization_backend == 'fbgemm':
            qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        elif self.quantization_backend == 'qnnpack':
            qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        else:
            qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        pytorch_model.qconfig = qconfig
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(pytorch_model)
        
        return prepared_model
    
    def _calibrate_model(self, model: Any) -> Any:
        """QAT doesn't need separate calibration."""
        return model
    
    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """Compute training loss (simplified implementation)."""
        # This is a placeholder - implement proper YOLO loss
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def evaluate(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the QAT model."""
        if self.optimized_model is None:
            raise ValueError("No optimized model to evaluate. Run optimize() first.")
        
        if hasattr(self.optimized_model, 'val'):
            results = self.optimized_model.val(data=test_data)
            return self._extract_metrics_from_results(results)
        
        return self._basic_evaluation(test_data, metrics)
    
    def _extract_metrics_from_results(self, results: Any) -> Dict[str, float]:
        """Extract metrics from YOLO validation results."""
        metrics = {}
        
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                metrics['mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                metrics['mAP50'] = float(results.box.map50)
            if hasattr(results.box, 'map75'):
                metrics['mAP75'] = float(results.box.map75)
        
        metrics['model_size_mb'] = self._get_model_size()
        return metrics
    
    def _basic_evaluation(self, test_data: Any, metrics: Optional[List[str]]) -> Dict[str, float]:
        """Basic evaluation implementation."""
        return {
            'accuracy': 0.0,
            'inference_time': 0.0,
            'model_size': self._get_model_size()
        }
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if self.optimized_model is None:
            return 0.0
        
        param_size = 0
        buffer_size = 0
        
        if hasattr(self.optimized_model, 'model'):
            model = self.optimized_model.model
        else:
            model = self.optimized_model
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def _record_optimization_metrics(self):
        """Record optimization metrics."""
        self.optimization_metrics = {
            'optimization_type': 'quantization_aware_training',
            'backend': self.quantization_backend,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'model_size_mb': self._get_model_size()
        }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information."""
        return {
            'optimizer_type': 'QATQuantizer',
            'config': self.config,
            'metrics': self.optimization_metrics,
            'quantization_backend': self.quantization_backend,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate
        }


class QuantizationUtils:
    """Utility functions for quantization."""
    
    @staticmethod
    def compare_model_sizes(original_model: Any, quantized_model: Any) -> Dict[str, float]:
        """Compare sizes of original and quantized models."""
        def get_model_size(model):
            param_size = 0
            buffer_size = 0
            
            if hasattr(model, 'model'):
                model = model.model
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / (1024 * 1024)
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0.0,
            'size_reduction_percent': ((original_size - quantized_size) / original_size) * 100 if original_size > 0 else 0.0
        }
    
    @staticmethod
    def benchmark_inference_speed(
        model: Any,
        test_input: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark inference speed of a model."""
        import time
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(test_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(test_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'avg_inference_time': sum(times) / len(times),
            'min_inference_time': min(times),
            'max_inference_time': max(times),
            'std_inference_time': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5,
            'fps': 1.0 / (sum(times) / len(times))
        }
    
    @staticmethod
    def get_quantization_info(model: Any) -> Dict[str, Any]:
        """Get information about model quantization."""
        info = {
            'is_quantized': False,
            'quantized_layers': [],
            'quantization_scheme': None
        }
        
        if hasattr(model, 'model'):
            model = model.model
        
        # Check if model is quantized
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                if 'qint' in str(module.weight.dtype):
                    info['is_quantized'] = True
                    info['quantized_layers'].append(name)
        
        return info


# Register quantization optimizers
from ..base import OptimizationRegistry

OptimizationRegistry.register('ptq', PostTrainingQuantizer)
OptimizationRegistry.register('dynamic_quantization', DynamicQuantizer)
OptimizationRegistry.register('qat', QATQuantizer)


def create_quantizer(
    quantization_type: str,
    model: Any,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> QuantizationOptimizer:
    """
    Factory function to create quantization optimizers.
    
    Args:
        quantization_type: Type of quantization ('ptq', 'dynamic', 'qat')
        model: Model to quantize
        config: Configuration dictionary
        **kwargs: Additional arguments
    
    Returns:
        Quantization optimizer instance
    """
    quantizer_map = {
        'ptq': PostTrainingQuantizer,
        'dynamic': DynamicQuantizer,
        'qat': QATQuantizer
    }
    
    if quantization_type not in quantizer_map:
        raise ValueError(f"Unsupported quantization type: {quantization_type}. "
                        f"Supported types: {list(quantizer_map.keys())}")
    
    quantizer_class = quantizer_map[quantization_type]
    return quantizer_class(model=model, config=config, **kwargs)