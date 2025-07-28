#!/usr/bin/env python3
"""
YOLO11 Project Main Entry Point

This is the main command-line interface for the YOLO11 project,
providing unified access to all functionality including training,
validation, inference, optimization, and benchmarking.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.model import YOLO11Model, YOLO11Factory
from core.trainer import YOLO11Trainer, create_trainer
from core.validator import YOLO11Validator, create_validator
from optimization.quantization.quantizers import create_quantizer
from benchmarks.speed_benchmark import SpeedBenchmark
from demos.detection_demo import DetectionDemo
from utils.helpers import load_config, setup_logging, get_device_info


class YOLO11CLI:
    """
    Command-line interface for YOLO11 project.
    """
    
    def __init__(self):
        """Initialize CLI."""
        self.config = {}
        self.logger = logging.getLogger(__name__)
    
    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Setup command-line argument parser."""
        parser = argparse.ArgumentParser(
            description='YOLO11 Project - Unified Deep Learning Platform',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run detection demo on image
  python main.py demo --task detect --input image.jpg
  
  # Train a model
  python main.py train --data coco8.yaml --epochs 100
  
  # Validate a model
  python main.py val --model yolo11n.pt --data coco8.yaml
  
  # Quantize a model
  python main.py optimize --model yolo11n.pt --method dynamic
  
  # Run speed benchmark
  python main.py benchmark --type model_sizes
  
  # Show system information
  python main.py info
            """
        )
        
        # Global arguments
        parser.add_argument('--config', '-c', type=str, default='configs/default.yaml',
                          help='Configuration file path')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging')
        parser.add_argument('--device', '-d', type=str, default=None,
                          help='Device to use (cpu, cuda, mps, etc.)')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Demo command
        demo_parser = subparsers.add_parser('demo', help='Run inference demo')
        demo_parser.add_argument('--task', type=str, default='detect',
                               choices=['detect', 'segment', 'classify', 'pose', 'obb'],
                               help='Task type')
        demo_parser.add_argument('--model', '-m', type=str, default=None,
                               help='Model path or size')
        demo_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input source (image, video, webcam)')
        demo_parser.add_argument('--output', '-o', type=str, default=None,
                               help='Output path')
        demo_parser.add_argument('--conf', type=float, default=0.5,
                               help='Confidence threshold')
        demo_parser.add_argument('--iou', type=float, default=0.45,
                               help='IoU threshold')
        demo_parser.add_argument('--no-show', action='store_true',
                               help='Do not display results')
        demo_parser.add_argument('--no-save', action='store_true',
                               help='Do not save results')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train a model')
        train_parser.add_argument('--model', '-m', type=str, default=None,
                                help='Model path or size')
        train_parser.add_argument('--data', type=str, required=True,
                                help='Dataset configuration file')
        train_parser.add_argument('--epochs', type=int, default=100,
                                help='Number of epochs')
        train_parser.add_argument('--batch', type=int, default=16,
                                help='Batch size')
        train_parser.add_argument('--imgsz', type=int, default=640,
                                help='Image size')
        train_parser.add_argument('--lr', type=float, default=0.01,
                                help='Learning rate')
        train_parser.add_argument('--output-dir', type=str, default=None,
                                help='Output directory')
        train_parser.add_argument('--resume', action='store_true',
                                help='Resume training from checkpoint')
        train_parser.add_argument('--fine-tune', action='store_true',
                                help='Fine-tune a pretrained model')
        
        # Validation command
        val_parser = subparsers.add_parser('val', help='Validate a model')
        val_parser.add_argument('--model', '-m', type=str, required=True,
                              help='Model path')
        val_parser.add_argument('--data', type=str, required=True,
                              help='Dataset configuration file')
        val_parser.add_argument('--batch', type=int, default=16,
                              help='Batch size')
        val_parser.add_argument('--imgsz', type=int, default=640,
                              help='Image size')
        val_parser.add_argument('--conf', type=float, default=0.001,
                              help='Confidence threshold')
        val_parser.add_argument('--iou', type=float, default=0.6,
                              help='IoU threshold')
        val_parser.add_argument('--output-dir', type=str, default=None,
                              help='Output directory')
        
        # Optimization command
        opt_parser = subparsers.add_parser('optimize', help='Optimize a model')
        opt_parser.add_argument('--model', '-m', type=str, required=True,
                              help='Model path')
        opt_parser.add_argument('--method', type=str, default='dynamic',
                              choices=['dynamic', 'ptq', 'qat'],
                              help='Optimization method')
        opt_parser.add_argument('--calibration-data', type=str, default=None,
                              help='Calibration dataset for PTQ')
        opt_parser.add_argument('--output', '-o', type=str, default=None,
                              help='Output path for optimized model')
        opt_parser.add_argument('--config-file', type=str, default=None,
                              help='Optimization configuration file')
        
        # Benchmark command
        bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
        bench_parser.add_argument('--type', type=str, default='model_sizes',
                                choices=['model_sizes', 'quantization', 'throughput', 'all'],
                                help='Benchmark type')
        bench_parser.add_argument('--model-sizes', nargs='+', default=['n', 's', 'm'],
                                help='Model sizes to benchmark')
        bench_parser.add_argument('--output-dir', type=str, default='benchmark_results',
                                help='Output directory')
        bench_parser.add_argument('--duration', type=int, default=60,
                                help='Throughput benchmark duration')
        
        # Info command
        subparsers.add_parser('info', help='Show system information')
        
        return parser
    
    def load_configuration(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if Path(config_path).exists():
                config = load_config(config_path)
                self.logger.info(f"Configuration loaded from: {config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
    
    def run_demo(self, args: argparse.Namespace) -> int:
        """Run inference demo."""
        try:
            # Initialize demo
            demo = DetectionDemo(
                model_path=args.model,
                model_size=args.model if args.model and len(args.model) == 1 else 'n',
                device=args.device,
                conf_threshold=args.conf,
                iou_threshold=args.iou
            )
            
            # Run demo based on input type
            if args.input.lower() == 'webcam':
                demo.detect_webcam(
                    camera_id=0,
                    output_path=args.output,
                    save=not args.no_save
                )
            elif Path(args.input).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                summary = demo.detect_video(
                    video_path=args.input,
                    output_path=args.output,
                    show=not args.no_show,
                    save=not args.no_save
                )
                self.logger.info(f"Video processing summary: {summary}")
            else:
                results = demo.detect_image(
                    image_path=args.input,
                    output_path=args.output,
                    show=not args.no_show,
                    save=not args.no_save
                )
                self.logger.info(f"Detection results: {results}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            return 1
    
    def run_training(self, args: argparse.Namespace) -> int:
        """Run model training."""
        try:
            # Create trainer
            if args.model and Path(args.model).exists():
                # Load existing model
                trainer = YOLO11Trainer(
                    model=args.model,
                    device=args.device,
                    output_dir=args.output_dir
                )
            else:
                # Create new model
                model_size = args.model if args.model and len(args.model) == 1 else 'n'
                trainer = create_trainer(
                    model_size=model_size,
                    device=args.device,
                    output_dir=args.output_dir
                )
            
            # Run training
            if args.resume:
                # Resume from checkpoint
                results = trainer.resume_training(
                    checkpoint_path=args.model,
                    data=args.data
                )
            elif args.fine_tune:
                # Fine-tune model
                results = trainer.fine_tune(
                    data=args.data,
                    epochs=args.epochs,
                    lr=args.lr
                )
            else:
                # Standard training
                results = trainer.train(
                    data=args.data,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    lr=args.lr
                )
            
            self.logger.info("Training completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return 1
    
    def run_validation(self, args: argparse.Namespace) -> int:
        """Run model validation."""
        try:
            # Create validator
            validator = YOLO11Validator(
                model=args.model,
                device=args.device,
                output_dir=args.output_dir
            )
            
            # Run validation
            results = validator.validate(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                conf=args.conf,
                iou=args.iou
            )
            
            self.logger.info("Validation completed successfully")
            self.logger.info(f"Validation results: {results}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return 1
    
    def run_optimization(self, args: argparse.Namespace) -> int:
        """Run model optimization."""
        try:
            # Load original model
            model = YOLO11Model(model_path=args.model, device=args.device)
            
            # Load optimization config if provided
            opt_config = {}
            if args.config_file and Path(args.config_file).exists():
                opt_config = load_config(args.config_file)
            
            # Create quantizer
            quantizer = create_quantizer(
                quantization_type=args.method,
                model=model,
                config=opt_config
            )
            
            # Run optimization
            if args.method == 'ptq' and args.calibration_data:
                # For PTQ, load calibration data
                # This is simplified - in practice you'd load actual data
                calibration_loader = [torch.randn(1, 3, 640, 640) for _ in range(100)]
                quantizer.set_calibration_data(calibration_loader)
            
            optimized_model = quantizer.optimize()
            
            # Save optimized model
            output_path = args.output or f"{Path(args.model).stem}_{args.method}.pt"
            quantizer.save_optimized_model(output_path)
            
            self.logger.info(f"Model optimization completed: {output_path}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return 1
    
    def run_benchmark(self, args: argparse.Namespace) -> int:
        """Run performance benchmark."""
        try:
            # Initialize benchmark
            benchmark = SpeedBenchmark(
                output_dir=args.output_dir,
                warmup_runs=10,
                benchmark_runs=100
            )
            
            # Run benchmarks
            if args.type == 'model_sizes' or args.type == 'all':
                benchmark.benchmark_model_sizes(sizes=args.model_sizes)
            
            if args.type == 'quantization' or args.type == 'all':
                benchmark.benchmark_quantization(model_size=args.model_sizes[0])
            
            if args.type == 'throughput' or args.type == 'all':
                benchmark.benchmark_throughput(
                    model_size=args.model_sizes[0],
                    duration_seconds=args.duration
                )
            
            # Generate report
            report_path = benchmark.generate_report()
            self.logger.info(f"Benchmark completed: {report_path}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return 1
    
    def show_system_info(self) -> int:
        """Show system information."""
        try:
            info = get_device_info()
            
            print("\nYOLO11 Project System Information")
            print("=" * 50)
            
            print(f"\nPlatform: {info.get('platform', 'Unknown')}")
            print(f"Processor: {info.get('processor', 'Unknown')}")
            print(f"Python Version: {info.get('python_version', 'Unknown')}")
            print(f"PyTorch Version: {info.get('torch_version', 'Unknown')}")
            
            print(f"\nCPU Cores: {info.get('cpu_count', 'Unknown')}")
            print(f"Memory Total: {info.get('memory_total_gb', 'Unknown')} GB")
            print(f"Memory Available: {info.get('memory_available_gb', 'Unknown')} GB")
            
            print(f"\nCUDA Available: {info.get('cuda_available', False)}")
            if info.get('cuda_available', False):
                print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
                print(f"GPU Count: {info.get('cuda_device_count', 0)}")
                
                if 'gpus' in info and info['gpus']:
                    print("\nGPU Information:")
                    for gpu in info['gpus']:
                        print(f"  GPU {gpu['id']}: {gpu['name']}")
                        print(f"    Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB")
                        print(f"    Temperature: {gpu['temperature']}°C")
                        print(f"    Load: {gpu['load']*100:.1f}%")
            
            print(f"\nMPS Available: {info.get('mps_available', False)}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return 1
    
    def run(self, args: list = None) -> int:
        """
        Main entry point for CLI.
        
        Args:
            args: Command line arguments (for testing)
        
        Returns:
            Exit code
        """
        # Parse arguments
        parser = self.setup_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging
        log_level = 'DEBUG' if parsed_args.verbose else 'INFO'
        setup_logging(log_level=log_level)
        
        # Load configuration
        self.config = self.load_configuration(parsed_args.config)
        
        # Show help if no command specified
        if not parsed_args.command:
            parser.print_help()
            return 0
        
        # Log system information if verbose
        if parsed_args.verbose:
            self.logger.info("System information:")
            system_info = get_device_info()
            for key, value in system_info.items():
                if key != 'gpus':  # Skip detailed GPU info for logs
                    self.logger.info(f"  {key}: {value}")
        
        # Route to appropriate command
        try:
            if parsed_args.command == 'demo':
                return self.run_demo(parsed_args)
            elif parsed_args.command == 'train':
                return self.run_training(parsed_args)
            elif parsed_args.command == 'val':
                return self.run_validation(parsed_args)
            elif parsed_args.command == 'optimize':
                return self.run_optimization(parsed_args)
            elif parsed_args.command == 'benchmark':
                return self.run_benchmark(parsed_args)
            elif parsed_args.command == 'info':
                return self.show_system_info()
            else:
                self.logger.error(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main function."""
    cli = YOLO11CLI()
    return cli.run()


if __name__ == '__main__':
    exit(main())