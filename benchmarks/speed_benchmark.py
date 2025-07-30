"""
YOLO11 Speed Benchmark

This script provides comprehensive speed benchmarking for YOLO11 models
across different configurations and optimization levels.
"""

import argparse
import torch
import time
import statistics
from pathlib import Path
import logging
import json
import sys
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import YOLO11Model, YOLO11Factory
from core.validator import YOLO11Validator
from optimization.quantization.quantizers import create_quantizer
from utils.helpers import get_device_info, format_time, Timer, ResourceMonitor

logger = logging.getLogger(__name__)


class SpeedBenchmark:
    """
    Comprehensive speed benchmarking for YOLO11 models.
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ):
        """
        Initialize speed benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # Get system info
        self.system_info = get_device_info()
        
        logger.info(f"Speed benchmark initialized")
        logger.info(f"System info: {self.system_info['platform']}")
        logger.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    def benchmark_model_sizes(
        self,
        task: str = 'detect',
        sizes: List[str] = ['n', 's', 'm', 'l', 'x'],
        image_sizes: List[int] = [320, 640, 1280],
        batch_sizes: List[int] = [1, 4, 8, 16]
    ) -> Dict[str, Any]:
        """
        Benchmark different model sizes.
        
        Args:
            task: Model task type
            sizes: Model sizes to benchmark
            image_sizes: Image sizes to test
            batch_sizes: Batch sizes to test
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking model sizes: {sizes}")
        
        results = {
            'task': task,
            'system_info': self.system_info,
            'configurations': [],
            'summary': {}
        }
        
        for size in sizes:
            logger.info(f"Benchmarking model size: {size}")
            
            # Create model
            model = YOLO11Model(task=task, size=size)
            
            for img_size in image_sizes:
                for batch_size in batch_sizes:
                    logger.info(f"Testing: size={size}, img_size={img_size}, batch_size={batch_size}")
                    
                    # Create test input
                    test_input = torch.randn(batch_size, 3, img_size, img_size)
                    if torch.cuda.is_available():
                        test_input = test_input.cuda()
                    
                    # Run benchmark
                    metrics = self._benchmark_inference(model, test_input)
                    
                    config_result = {
                        'model_size': size,
                        'image_size': img_size,
                        'batch_size': batch_size,
                        **metrics
                    }
                    
                    results['configurations'].append(config_result)
        
        # Calculate summary
        results['summary'] = self._calculate_summary(results['configurations'])
        
        # Save results
        self._save_results(results, 'model_sizes_benchmark.json')
        
        return results
    
    def benchmark_quantization(
        self,
        model_size: str = 'n',
        task: str = 'detect',
        quantization_methods: List[str] = ['dynamic', 'ptq'],
        image_size: int = 640,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark quantization methods.
        
        Args:
            model_size: Model size to use
            task: Model task type
            quantization_methods: Quantization methods to test
            image_size: Image size
            batch_size: Batch size
        
        Returns:
            Quantization benchmark results
        """
        logger.info(f"Benchmarking quantization methods: {quantization_methods}")
        
        results = {
            'model_size': model_size,
            'task': task,
            'image_size': image_size,
            'batch_size': batch_size,
            'system_info': self.system_info,
            'methods': {}
        }
        
        # Benchmark original model
        logger.info("Benchmarking original model (FP32)")
        original_model = YOLO11Model(task=task, size=model_size)
        
        test_input = torch.randn(batch_size, 3, image_size, image_size)
        if torch.cuda.is_available():
            test_input = test_input.cuda()
        
        original_metrics = self._benchmark_inference(original_model, test_input)
        results['methods']['original'] = original_metrics
        
        # Benchmark quantization methods
        for method in quantization_methods:
            logger.info(f"Benchmarking quantization method: {method}")
            
            try:
                # Create quantized model
                quantizer = create_quantizer(method, original_model)
                
                # For dynamic quantization, no calibration needed
                if method == 'dynamic':
                    quantized_model = quantizer.optimize()
                elif method == 'ptq':
                    # Create dummy calibration data
                    calibration_data = [test_input for _ in range(10)]
                    quantizer.set_calibration_data(calibration_data)
                    quantized_model = quantizer.optimize()
                else:
                    logger.warning(f"Quantization method {method} not implemented for benchmarking")
                    continue
                
                # Benchmark quantized model
                quantized_metrics = self._benchmark_inference(quantized_model, test_input)
                
                # Calculate speedup and compression
                speedup = original_metrics['avg_inference_time'] / quantized_metrics['avg_inference_time']
                
                quantized_metrics.update({
                    'speedup': speedup,
                    'optimization_info': quantizer.get_optimization_info()
                })
                
                results['methods'][method] = quantized_metrics
                
            except Exception as e:
                logger.error(f"Failed to benchmark quantization method {method}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                results['methods'][method] = {'error': str(e)}
        
        # Save results
        self._save_results(results, 'quantization_benchmark.json')
        
        return results
    
    def benchmark_throughput(
        self,
        model_size: str = 'n',
        task: str = 'detect',
        duration_seconds: int = 60,
        image_size: int = 640,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark sustained throughput.
        
        Args:
            model_size: Model size
            task: Model task type
            duration_seconds: Benchmark duration in seconds
            image_size: Image size
            batch_size: Batch size
        
        Returns:
            Throughput benchmark results
        """
        logger.info(f"Benchmarking throughput for {duration_seconds} seconds")
        
        # Create model
        model = YOLO11Model(task=task, size=model_size)
        
        # Create test input
        test_input = torch.randn(batch_size, 3, image_size, image_size)
        if torch.cuda.is_available():
            test_input = test_input.cuda()
        
        # Start resource monitoring
        monitor = ResourceMonitor(interval=1.0)
        monitor.start_monitoring()
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(self.warmup_runs):
            _ = model.predict(test_input, verbose=False)
        
        # Benchmark
        logger.info("Starting throughput benchmark...")
        start_time = time.time()
        inference_count = 0
        inference_times = []
        
        while time.time() - start_time < duration_seconds:
            inf_start = time.time()
            _ = model.predict(test_input, verbose=False)
            inf_end = time.time()
            
            inference_times.append(inf_end - inf_start)
            inference_count += 1
            
            if inference_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = inference_count / elapsed
                logger.info(f"Progress: {elapsed:.1f}s, Inferences: {inference_count}, FPS: {current_fps:.2f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Calculate metrics
        avg_inference_time = statistics.mean(inference_times)
        fps = inference_count / total_time
        images_per_second = fps * batch_size
        
        resource_stats = monitor.get_average_usage()
        
        results = {
            'model_size': model_size,
            'task': task,
            'image_size': image_size,
            'batch_size': batch_size,
            'duration_seconds': total_time,
            'total_inferences': inference_count,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'images_per_second': images_per_second,
            'resource_usage': resource_stats,
            'system_info': self.system_info
        }
        
        # Save resource monitoring history
        monitor.save_history(self.output_dir / 'resource_history.json')
        
        # Save results
        self._save_results(results, 'throughput_benchmark.json')
        
        logger.info(f"Throughput benchmark completed: {fps:.2f} FPS")
        
        return results
    
    def _benchmark_inference(
        self,
        model: YOLO11Model,
        test_input: torch.Tensor
    ) -> Dict[str, float]:
        """
        Benchmark inference for a specific model and input.
        
        Args:
            model: Model to benchmark
            test_input: Test input tensor
        
        Returns:
            Inference metrics
        """
        # Warmup
        model.model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model.predict(test_input, verbose=False)
        
        # Benchmark
        inference_times = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                start_time = time.time()
                _ = model.predict(test_input, verbose=False)
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'std_inference_time': std_time,
            'fps': 1.0 / avg_time,
            'throughput': test_input.shape[0] / avg_time  # images per second
        }
    
    def _calculate_summary(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not configurations:
            return {}
        
        fps_values = [config['fps'] for config in configurations]
        throughput_values = [config['throughput'] for config in configurations]
        
        return {
            'best_fps': max(fps_values),
            'worst_fps': min(fps_values),
            'avg_fps': statistics.mean(fps_values),
            'best_throughput': max(throughput_values),
            'worst_throughput': min(throughput_values),
            'avg_throughput': statistics.mean(throughput_values),
            'total_configurations': len(configurations)
        }
    
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("YOLO11 Speed Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            # System information
            f.write("System Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.system_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Load and summarize results
            result_files = list(self.output_dir.glob("*.json"))
            for result_file in result_files:
                if result_file.name == "resource_history.json":
                    continue
                
                try:
                    with open(result_file, 'r') as rf:
                        data = json.load(rf)
                    
                    f.write(f"Results from {result_file.name}:\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'summary' in data:
                        for key, value in data['summary'].items():
                            f.write(f"{key}: {value}\n")
                    
                    f.write("\n")
                    
                except Exception as e:
                    logger.warning(f"Could not process {result_file}: {e}")
        
        logger.info(f"Benchmark report generated: {report_path}")
        return str(report_path)


def main():
    """Main function for speed benchmark."""
    parser = argparse.ArgumentParser(description='YOLO11 Speed Benchmark')
    parser.add_argument('--benchmark-type', '-t', type=str, default='model_sizes',
                       choices=['model_sizes', 'quantization', 'throughput', 'all'],
                       help='Type of benchmark to run')
    parser.add_argument('--task', type=str, default='detect',
                       choices=['detect', 'segment', 'classify', 'pose', 'obb'],
                       help='Model task type')
    parser.add_argument('--sizes', nargs='+', default=['n', 's', 'm'],
                       help='Model sizes to benchmark')
    parser.add_argument('--output-dir', '-o', type=str, default='benchmark_results',
                       help='Output directory for benchmark results')
    parser.add_argument('--warmup-runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--image-sizes', nargs='+', type=int, default=[320, 640, 1280],
                       help='Image sizes to test')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--quantization-methods', nargs='+', default=['dynamic', 'ptq'],
                       help='Quantization methods to benchmark')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for throughput benchmark (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize benchmark
    benchmark = SpeedBenchmark(
        output_dir=args.output_dir,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )
    
    try:
        # Run benchmarks based on type
        if args.benchmark_type == 'model_sizes' or args.benchmark_type == 'all':
            logger.info("Running model sizes benchmark...")
            benchmark.benchmark_model_sizes(
                task=args.task,
                sizes=args.sizes,
                image_sizes=args.image_sizes,
                batch_sizes=args.batch_sizes
            )
        
        if args.benchmark_type == 'quantization' or args.benchmark_type == 'all':
            logger.info("Running quantization benchmark...")
            benchmark.benchmark_quantization(
                model_size=args.sizes[0] if args.sizes else 'n',
                task=args.task,
                quantization_methods=args.quantization_methods,
                image_size=args.image_sizes[0] if args.image_sizes else 640,
                batch_size=args.batch_sizes[0] if args.batch_sizes else 1
            )
        
        if args.benchmark_type == 'throughput' or args.benchmark_type == 'all':
            logger.info("Running throughput benchmark...")
            benchmark.benchmark_throughput(
                model_size=args.sizes[0] if args.sizes else 'n',
                task=args.task,
                duration_seconds=args.duration,
                image_size=args.image_sizes[0] if args.image_sizes else 640,
                batch_size=args.batch_sizes[0] if args.batch_sizes else 1
            )
        
        # Generate comprehensive report
        report_path = benchmark.generate_report()
        logger.info(f"Benchmark completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())