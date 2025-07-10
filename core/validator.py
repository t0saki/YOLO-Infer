"""
YOLO11 Validation Module

This module provides comprehensive validation capabilities for YOLO11 models,
including accuracy metrics, performance benchmarks, and model comparison.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import logging
import time
import numpy as np
from datetime import datetime

from ultralytics import YOLO
from .model import YOLO11Model

logger = logging.getLogger(__name__)


class YOLO11Validator:
    """
    Comprehensive validator for YOLO11 models with support for:
    - Standard validation metrics (mAP, Precision, Recall)
    - Performance benchmarking (inference speed, throughput)
    - Model comparison and analysis
    - Cross-validation
    """
    
    def __init__(
        self,
        model: Union[YOLO11Model, str, Path],
        device: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the validator.
        
        Args:
            model: YOLO11Model instance or path to model weights
            device: Device to run validation on
            output_dir: Directory to save validation outputs
        """
        self.device = device or self._get_default_device()
        self.output_dir = Path(output_dir) if output_dir else Path("experiments") / f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if isinstance(model, (str, Path)):
            self.model = YOLO11Model(model_path=model, device=self.device)
        elif isinstance(model, YOLO11Model):
            self.model = model
        else:
            raise ValueError("Model must be YOLO11Model instance or path to weights")
        
        # Validation state
        self.validation_history = []
        self.benchmark_results = {}
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"YOLO11Validator initialized with device: {self.device}")
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
        """Setup validation-specific logging."""
        log_file = self.output_dir / "validation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def validate(
        self,
        data: Union[str, Path],
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.001,
        iou: float = 0.6,
        save_json: bool = True,
        save_hybrid: bool = False,
        augment: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the YOLO11 model.
        
        Args:
            data: Path to validation dataset configuration
            imgsz: Image size for validation
            batch: Batch size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save_json: Save results in JSON format
            save_hybrid: Save hybrid version of labels
            augment: Apply test time augmentation
            verbose: Verbose output
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting YOLO11 validation...")
        logger.info(f"Validation parameters: imgsz={imgsz}, batch={batch}, conf={conf}, iou={iou}")
        
        # Prepare validation arguments
        val_args = {
            'data': data,
            'imgsz': imgsz,
            'batch': batch,
            'conf': conf,
            'iou': iou,
            'save_json': save_json,
            'save_hybrid': save_hybrid,
            'augment': augment,
            'verbose': verbose,
            'device': self.device,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            **kwargs
        }
        
        try:
            # Run validation
            start_time = time.time()
            results = self.model.val(**val_args)
            end_time = time.time()
            
            # Process and record results
            validation_time = end_time - start_time
            processed_results = self._process_validation_results(results, validation_time)
            
            # Save validation summary
            self._save_validation_summary(processed_results)
            
            logger.info(f"Validation completed in {validation_time:.2f} seconds")
            return processed_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def benchmark_speed(
        self,
        test_data: Union[str, Path, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10,
        batch_sizes: List[int] = [1, 8, 16, 32],
        image_sizes: List[int] = [320, 640, 1280]
    ) -> Dict[str, Any]:
        """
        Benchmark inference speed of the model.
        
        Args:
            test_data: Test data for benchmarking
            num_runs: Number of inference runs for each configuration
            warmup_runs: Number of warmup runs
            batch_sizes: List of batch sizes to test
            image_sizes: List of image sizes to test
        
        Returns:
            Benchmarking results
        """
        logger.info("Starting speed benchmarking...")
        
        benchmark_results = {
            'device': self.device,
            'model_info': self.model.get_model_info(),
            'configurations': [],
            'summary': {}
        }
        
        for img_size in image_sizes:
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking: batch_size={batch_size}, img_size={img_size}")
                
                # Create test input
                if isinstance(test_data, torch.Tensor):
                    test_input = test_data
                else:
                    # Create dummy input for speed testing
                    test_input = torch.randn(batch_size, 3, img_size, img_size, device=self.device)
                
                # Run benchmark
                config_results = self.model.benchmark(
                    data_source=test_input,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs
                )
                
                config_results.update({
                    'batch_size': batch_size,
                    'image_size': img_size,
                    'images_per_second': batch_size * config_results['fps']
                })
                
                benchmark_results['configurations'].append(config_results)
        
        # Calculate summary statistics
        benchmark_results['summary'] = self._calculate_benchmark_summary(benchmark_results['configurations'])
        
        # Save benchmark results
        self._save_benchmark_results(benchmark_results)
        
        logger.info("Speed benchmarking completed")
        return benchmark_results
    
    def compare_models(
        self,
        models: List[Union[YOLO11Model, str, Path]],
        data: Union[str, Path],
        metrics: List[str] = ['mAP50', 'mAP50-95', 'precision', 'recall', 'inference_time'],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple YOLO11 models.
        
        Args:
            models: List of models to compare
            data: Validation dataset
            metrics: Metrics to compare
            **kwargs: Additional validation arguments
        
        Returns:
            Model comparison results
        """
        logger.info(f"Comparing {len(models)} models...")
        
        comparison_results = {
            'models': [],
            'metrics': metrics,
            'comparison_table': {},
            'rankings': {}
        }
        
        # Validate each model
        for i, model in enumerate(models):
            logger.info(f"Validating model {i+1}/{len(models)}")
            
            # Create validator for this model
            if not isinstance(model, YOLO11Model):
                model = YOLO11Model(model_path=model, device=self.device)
            
            validator = YOLO11Validator(model, device=self.device)
            
            # Run validation
            val_results = validator.validate(data=data, **kwargs)
            
            # Extract relevant metrics
            model_metrics = self._extract_comparison_metrics(val_results, metrics)
            model_info = {
                'model_name': f"Model_{i+1}",
                'model_info': model.get_model_info(),
                'metrics': model_metrics
            }
            
            comparison_results['models'].append(model_info)
        
        # Create comparison table
        comparison_results['comparison_table'] = self._create_comparison_table(comparison_results['models'], metrics)
        
        # Calculate rankings
        comparison_results['rankings'] = self._calculate_model_rankings(comparison_results['models'], metrics)
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
        
        logger.info("Model comparison completed")
        return comparison_results
    
    def cross_validate(
        self,
        data: Union[str, Path],
        k_folds: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            data: Dataset configuration
            k_folds: Number of folds for cross-validation
            **kwargs: Additional validation arguments
        
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {k_folds}-fold cross-validation...")
        
        cv_results = {
            'k_folds': k_folds,
            'fold_results': [],
            'summary_statistics': {}
        }
        
        for fold in range(k_folds):
            logger.info(f"Running fold {fold+1}/{k_folds}")
            
            # Run validation for this fold
            fold_results = self.validate(data=data, **kwargs)
            fold_results['fold'] = fold + 1
            
            cv_results['fold_results'].append(fold_results)
        
        # Calculate summary statistics across folds
        cv_results['summary_statistics'] = self._calculate_cv_statistics(cv_results['fold_results'])
        
        # Save cross-validation results
        self._save_cv_results(cv_results)
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def _process_validation_results(self, results: Any, validation_time: float) -> Dict[str, Any]:
        """Process raw validation results into structured format."""
        processed = {
            'timestamp': datetime.now().isoformat(),
            'validation_time_seconds': validation_time,
            'device': self.device,
            'model_info': self.model.get_model_info()
        }
        
        # Extract metrics from results
        if hasattr(results, 'box'):
            box_metrics = {}
            if hasattr(results.box, 'map'):
                box_metrics['mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                box_metrics['mAP50'] = float(results.box.map50)
            if hasattr(results.box, 'map75'):
                box_metrics['mAP75'] = float(results.box.map75)
            if hasattr(results.box, 'mp'):
                box_metrics['precision'] = float(results.box.mp)
            if hasattr(results.box, 'mr'):
                box_metrics['recall'] = float(results.box.mr)
            
            processed['box_metrics'] = box_metrics
        
        # Extract speed metrics if available
        if hasattr(results, 'speed'):
            speed_metrics = {}
            for key, value in results.speed.items():
                speed_metrics[key] = float(value)
            processed['speed_metrics'] = speed_metrics
        
        return processed
    
    def _calculate_benchmark_summary(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark results."""
        if not configurations:
            return {}
        
        # Extract key metrics
        fps_values = [config['fps'] for config in configurations]
        latency_values = [config['avg_inference_time'] for config in configurations]
        throughput_values = [config.get('images_per_second', 0) for config in configurations]
        
        summary = {
            'best_fps': max(fps_values),
            'avg_fps': sum(fps_values) / len(fps_values),
            'best_latency': min(latency_values),
            'avg_latency': sum(latency_values) / len(latency_values),
            'best_throughput': max(throughput_values),
            'total_configurations_tested': len(configurations)
        }
        
        # Find best configuration
        best_config_idx = fps_values.index(summary['best_fps'])
        summary['best_configuration'] = configurations[best_config_idx]
        
        return summary
    
    def _extract_comparison_metrics(self, val_results: Dict, metrics: List[str]) -> Dict[str, float]:
        """Extract specified metrics from validation results."""
        extracted = {}
        
        for metric in metrics:
            if metric == 'mAP50-95' and 'box_metrics' in val_results:
                extracted[metric] = val_results['box_metrics'].get('mAP50-95', 0.0)
            elif metric == 'mAP50' and 'box_metrics' in val_results:
                extracted[metric] = val_results['box_metrics'].get('mAP50', 0.0)
            elif metric == 'mAP75' and 'box_metrics' in val_results:
                extracted[metric] = val_results['box_metrics'].get('mAP75', 0.0)
            elif metric == 'precision' and 'box_metrics' in val_results:
                extracted[metric] = val_results['box_metrics'].get('precision', 0.0)
            elif metric == 'recall' and 'box_metrics' in val_results:
                extracted[metric] = val_results['box_metrics'].get('recall', 0.0)
            elif metric == 'inference_time':
                extracted[metric] = val_results.get('validation_time_seconds', 0.0)
            else:
                extracted[metric] = 0.0
        
        return extracted
    
    def _create_comparison_table(self, models: List[Dict], metrics: List[str]) -> Dict[str, List]:
        """Create comparison table from model results."""
        table = {'Model': [model['model_name'] for model in models]}
        
        for metric in metrics:
            table[metric] = [model['metrics'].get(metric, 0.0) for model in models]
        
        return table
    
    def _calculate_model_rankings(self, models: List[Dict], metrics: List[str]) -> Dict[str, List]:
        """Calculate model rankings for each metric."""
        rankings = {}
        
        for metric in metrics:
            # Get metric values for all models
            metric_values = [(i, model['metrics'].get(metric, 0.0)) for i, model in enumerate(models)]
            
            # Sort by metric value (descending for most metrics, ascending for inference_time)
            reverse_sort = metric != 'inference_time'
            metric_values.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            # Create ranking list
            ranking = []
            for rank, (model_idx, value) in enumerate(metric_values, 1):
                ranking.append({
                    'rank': rank,
                    'model_name': models[model_idx]['model_name'],
                    'value': value
                })
            
            rankings[metric] = ranking
        
        return rankings
    
    def _calculate_cv_statistics(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Calculate cross-validation statistics."""
        if not fold_results:
            return {}
        
        stats = {}
        
        # Collect metrics from all folds
        all_metrics = {}
        for fold_result in fold_results:
            if 'box_metrics' in fold_result:
                for metric, value in fold_result['box_metrics'].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate statistics for each metric
        for metric, values in all_metrics.items():
            if values:
                stats[metric] = {
                    'mean': sum(values) / len(values),
                    'std': (sum([(v - sum(values)/len(values))**2 for v in values]) / len(values))**0.5,
                    'min': min(values),
                    'max': max(values),
                    'cv': (sum([(v - sum(values)/len(values))**2 for v in values]) / len(values))**0.5 / (sum(values)/len(values)) if sum(values) > 0 else 0
                }
        
        return stats
    
    def _save_validation_summary(self, results: Dict[str, Any]):
        """Save validation summary to file."""
        summary_file = self.output_dir / "validation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Validation time: {results['validation_time_seconds']:.2f} seconds\n")
            f.write(f"Device: {results['device']}\n\n")
            
            # Model info
            f.write("Model Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in results['model_info'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Metrics
            if 'box_metrics' in results:
                f.write("Box Detection Metrics:\n")
                f.write("-" * 25 + "\n")
                for metric, value in results['box_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            if 'speed_metrics' in results:
                f.write("Speed Metrics:\n")
                f.write("-" * 15 + "\n")
                for metric, value in results['speed_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Validation summary saved to: {summary_file}")
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        import json
        
        benchmark_file = self.output_dir / "benchmark_results.json"
        
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save human-readable summary
        summary_file = self.output_dir / "benchmark_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Device: {results['device']}\n")
            f.write(f"Total configurations tested: {results['summary']['total_configurations_tested']}\n\n")
            
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best FPS: {results['summary']['best_fps']:.2f}\n")
            f.write(f"Average FPS: {results['summary']['avg_fps']:.2f}\n")
            f.write(f"Best Latency: {results['summary']['best_latency']:.4f} seconds\n")
            f.write(f"Average Latency: {results['summary']['avg_latency']:.4f} seconds\n")
            f.write(f"Best Throughput: {results['summary']['best_throughput']:.2f} images/second\n\n")
            
            if 'best_configuration' in results['summary']:
                best_config = results['summary']['best_configuration']
                f.write("Best Configuration:\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Batch Size: {best_config['batch_size']}\n")
                f.write(f"  Image Size: {best_config['image_size']}\n")
                f.write(f"  FPS: {best_config['fps']:.2f}\n")
                f.write(f"  Latency: {best_config['avg_inference_time']:.4f} seconds\n")
        
        logger.info(f"Benchmark results saved to: {benchmark_file}")
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save model comparison results to file."""
        import json
        
        comparison_file = self.output_dir / "model_comparison.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable comparison table
        table_file = self.output_dir / "comparison_table.txt"
        
        with open(table_file, 'w') as f:
            f.write("YOLO11 Model Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            # Write comparison table
            table = results['comparison_table']
            if table:
                # Write header
                headers = list(table.keys())
                f.write("\t".join(headers) + "\n")
                f.write("-" * (len("\t".join(headers)) + 10) + "\n")
                
                # Write data rows
                num_rows = len(table[headers[0]])
                for i in range(num_rows):
                    row = []
                    for header in headers:
                        value = table[header][i]
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    f.write("\t".join(row) + "\n")
                
                f.write("\n")
            
            # Write rankings
            f.write("Model Rankings by Metric:\n")
            f.write("-" * 30 + "\n")
            for metric, ranking in results['rankings'].items():
                f.write(f"\n{metric}:\n")
                for entry in ranking:
                    f.write(f"  {entry['rank']}. {entry['model_name']}: {entry['value']:.4f}\n")
        
        logger.info(f"Model comparison results saved to: {comparison_file}")
    
    def _save_cv_results(self, results: Dict[str, Any]):
        """Save cross-validation results to file."""
        import json
        
        cv_file = self.output_dir / "cross_validation.json"
        
        with open(cv_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable summary
        summary_file = self.output_dir / "cv_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Cross-Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of folds: {results['k_folds']}\n\n")
            
            # Write summary statistics
            f.write("Cross-Validation Statistics:\n")
            f.write("-" * 30 + "\n")
            for metric, stats in results['summary_statistics'].items():
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std: {stats['std']:.4f}\n")
                f.write(f"  Min: {stats['min']:.4f}\n")
                f.write(f"  Max: {stats['max']:.4f}\n")
                f.write(f"  CV: {stats['cv']:.4f}\n")
        
        logger.info(f"Cross-validation results saved to: {cv_file}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get benchmark results."""
        return self.benchmark_results
    
    def __repr__(self) -> str:
        return (f"YOLO11Validator(model={self.model.task}, device={self.device}, "
                f"output_dir={self.output_dir})")


def create_validator(
    model_type: str = 'detect',
    model_size: str = 'n',
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> YOLO11Validator:
    """
    Factory function to create YOLO11 validators.
    
    Args:
        model_type: Type of model ('detect', 'segment', 'classify', 'pose', 'obb')
        model_size: Size of model ('n', 's', 'm', 'l', 'x')
        model_path: Path to model weights (if None, uses pretrained)
        device: Device to use for validation
        output_dir: Output directory for validation results
    
    Returns:
        YOLO11Validator instance
    """
    # Create model
    if model_path:
        model = YOLO11Model(model_path=model_path, device=device)
    else:
        model = YOLO11Model(task=model_type, size=model_size, device=device)
    
    # Create validator
    validator = YOLO11Validator(
        model=model,
        device=device,
        output_dir=output_dir
    )
    
    return validator