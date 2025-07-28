"""
Base classes for model optimization.

This module provides abstract base classes for different model optimization
techniques such as quantization, pruning, knowledge distillation, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for model optimization techniques.
    
    This class defines the interface that all optimization techniques
    (quantization, pruning, distillation, etc.) should implement.
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            model: The model to optimize
            config: Configuration parameters for optimization
            device: Device to run optimization on
        """
        self.original_model = model
        self.optimized_model = None
        self.config = config or {}
        self.device = device or self._get_default_device()
        self.optimization_metrics = {}
        self.optimization_history = []
        
    def _get_default_device(self) -> str:
        """Get default device based on availability."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    @abstractmethod
    def optimize(self, **kwargs) -> Any:
        """
        Perform the optimization.
        
        Returns:
            Optimized model
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the optimized model.
        
        Args:
            test_data: Test dataset or data loader
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about the optimization process.
        
        Returns:
            Dictionary containing optimization details
        """
        pass
    
    def save_optimized_model(self, path: Union[str, Path]) -> None:
        """
        Save the optimized model.
        
        Args:
            path: Path to save the model
        """
        if self.optimized_model is None:
            raise ValueError("No optimized model to save. Run optimize() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and optimization info
        save_dict = {
            'optimization_info': self.get_optimization_info(),
            'config': self.config
        }
        
        # Handle model saving with error handling for non-picklable objects
        try:
            # Try to save the entire model first
            save_dict['model'] = self.optimized_model
            torch.save(save_dict, path)
        except (TypeError, AttributeError) as e:
            if "cannot pickle" in str(e):
                logger.warning("Cannot pickle entire model, saving model state dict instead")
                # Save only the model state dict and necessary info to reconstruct
                model_to_save = self.optimized_model
                
                # Handle YOLO11 model wrapper
                if hasattr(model_to_save, 'model'):
                    # Extract the underlying PyTorch model
                    save_dict['model_state_dict'] = model_to_save.model.state_dict()
                    save_dict['model_class'] = model_to_save.__class__.__name__
                    save_dict['model_config'] = {
                        'task': getattr(model_to_save, 'task', None),
                        'size': getattr(model_to_save, 'size', None),
                        'device': getattr(model_to_save, 'device', None)
                    }
                else:
                    # Direct PyTorch model
                    save_dict['model_state_dict'] = model_to_save.state_dict()
                    save_dict['model_class'] = model_to_save.__class__.__name__
                
                torch.save(save_dict, path)
            else:
                raise
        
        logger.info(f"Optimized model saved to: {path}")
    
    def load_optimized_model(self, path: Union[str, Path]) -> None:
        """
        Load an optimized model.
        
        Args:
            path: Path to the saved model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        save_dict = torch.load(path, map_location=self.device)
        
        # Handle both old and new save formats
        if 'model' in save_dict:
            # Full model was saved
            self.optimized_model = save_dict['model']
        elif 'model_state_dict' in save_dict:
            # Only state dict was saved, reconstruct model
            logger.warning("Loading model from state dict - reconstruction may not be exact")
            # In this case, we'll need to reconstruct the model
            # This is a simplified approach - in practice, you'd want to properly reconstruct the model
            self.optimized_model = None
            logger.info("Model state dict loaded - full model reconstruction not implemented in base class")
        else:
            raise ValueError("Invalid save file format: no model or state dict found")
        
        self.optimization_metrics = save_dict.get('optimization_info', {})
        self.config.update(save_dict.get('config', {}))
        
        logger.info(f"Optimized model loaded from: {path}")
    
    def compare_models(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare original and optimized models.
        
        Args:
            test_data: Test data for comparison
            metrics: Metrics to compute for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if self.optimized_model is None:
            raise ValueError("No optimized model to compare. Run optimize() first.")
        
        # Evaluate original model
        original_metrics = self._evaluate_model(self.original_model, test_data, metrics)
        
        # Evaluate optimized model
        optimized_metrics = self._evaluate_model(self.optimized_model, test_data, metrics)
        
        return {
            'original': original_metrics,
            'optimized': optimized_metrics,
            'improvement': {
                key: optimized_metrics[key] - original_metrics[key]
                for key in original_metrics.keys()
            }
        }
    
    def _evaluate_model(
        self,
        model: Any,
        test_data: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Helper method to evaluate a model.
        
        Args:
            model: Model to evaluate
            test_data: Test data
            metrics: Metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        # This is a placeholder - should be implemented by subclasses
        # or use the evaluate method of the specific optimizer
        return {}


class QuantizationOptimizer(BaseOptimizer):
    """Base class for quantization optimizers."""
    
    SUPPORTED_BACKENDS = ['fbgemm', 'qnnpack', 'onednn']
    SUPPORTED_DTYPES = [torch.qint8, torch.quint8, torch.qint32]
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, config, device)
        self.quantization_backend = self.config.get('backend', 'qnnpack')
        self.quantization_dtype = self.config.get('dtype', torch.qint8)
        self.calibration_data = None
        
    def set_calibration_data(self, calibration_data: Any) -> None:
        """Set calibration data for post-training quantization."""
        self.calibration_data = calibration_data
    
    @abstractmethod
    def _prepare_model_for_quantization(self) -> Any:
        """Prepare model for quantization."""
        pass
    
    @abstractmethod
    def _calibrate_model(self, model: Any) -> Any:
        """Calibrate the model using calibration data."""
        pass


class PruningOptimizer(BaseOptimizer):
    """Base class for pruning optimizers."""
    
    SUPPORTED_METHODS = ['magnitude', 'structured', 'unstructured', 'gradual']
    
    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, config, device)
        self.pruning_method = self.config.get('method', 'magnitude')
        self.sparsity_ratio = self.config.get('sparsity_ratio', 0.5)
        
    @abstractmethod
    def _identify_pruning_targets(self) -> List[nn.Module]:
        """Identify layers/modules to prune."""
        pass
    
    @abstractmethod
    def _apply_pruning_mask(self, targets: List[nn.Module]) -> None:
        """Apply pruning masks to target modules."""
        pass


class DistillationOptimizer(BaseOptimizer):
    """Base class for knowledge distillation optimizers."""
    
    def __init__(
        self,
        student_model: Any,
        teacher_model: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        # For distillation, the model being optimized is the student
        super().__init__(student_model, config, device)
        self.teacher_model = teacher_model
        self.temperature = self.config.get('temperature', 4.0)
        self.alpha = self.config.get('alpha', 0.7)
        
    @abstractmethod
    def _compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        pass


class OptimizationPipeline:
    """
    Pipeline for applying multiple optimization techniques.
    
    This class allows chaining multiple optimization techniques
    in a specified order.
    """
    
    def __init__(self, model: Any):
        self.original_model = model
        self.current_model = model
        self.optimizers = []
        self.pipeline_history = []
        
    def add_optimizer(
        self,
        optimizer_class: type,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'OptimizationPipeline':
        """
        Add an optimizer to the pipeline.
        
        Args:
            optimizer_class: Class of the optimizer to add
            config: Configuration for the optimizer
            **kwargs: Additional arguments for optimizer initialization
            
        Returns:
            Self for method chaining
        """
        optimizer = optimizer_class(
            model=self.current_model,
            config=config,
            **kwargs
        )
        self.optimizers.append(optimizer)
        return self
    
    def run_pipeline(self, **kwargs) -> Any:
        """
        Run the complete optimization pipeline.
        
        Returns:
            Final optimized model
        """
        for i, optimizer in enumerate(self.optimizers):
            logger.info(f"Running optimization step {i+1}/{len(self.optimizers)}: {type(optimizer).__name__}")
            
            # Run optimization
            optimized_model = optimizer.optimize(**kwargs)
            
            # Update current model for next step
            self.current_model = optimized_model
            
            # Record optimization info
            step_info = {
                'step': i + 1,
                'optimizer': type(optimizer).__name__,
                'optimization_info': optimizer.get_optimization_info()
            }
            self.pipeline_history.append(step_info)
            
            logger.info(f"Completed optimization step {i+1}")
        
        return self.current_model
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the optimization pipeline."""
        return {
            'total_steps': len(self.optimizers),
            'optimizers_used': [type(opt).__name__ for opt in self.optimizers],
            'pipeline_history': self.pipeline_history
        }
    
    def save_pipeline(self, path: Union[str, Path]) -> None:
        """Save the entire optimization pipeline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'final_model': self.current_model,
            'pipeline_summary': self.get_pipeline_summary(),
            'optimizers_config': [opt.config for opt in self.optimizers]
        }
        
        torch.save(save_dict, path)
        logger.info(f"Optimization pipeline saved to: {path}")


class OptimizationRegistry:
    """Registry for optimization techniques."""
    
    _optimizers = {}
    
    @classmethod
    def register(cls, name: str, optimizer_class: type) -> None:
        """Register an optimizer class."""
        cls._optimizers[name] = optimizer_class
    
    @classmethod
    def get_optimizer(cls, name: str) -> type:
        """Get an optimizer class by name."""
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(cls._optimizers.keys())}")
        return cls._optimizers[name]
    
    @classmethod
    def list_optimizers(cls) -> List[str]:
        """List all registered optimizers."""
        return list(cls._optimizers.keys())
    
    @classmethod
    def create_optimizer(
        cls,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseOptimizer:
        """Create an optimizer instance by name."""
        optimizer_class = cls.get_optimizer(name)
        return optimizer_class(model=model, config=config, **kwargs)