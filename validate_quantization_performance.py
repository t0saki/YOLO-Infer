#!/usr/bin/env python3
"""
综合验证不同量化方式的表现数据

该脚本提供多种验证方法：
1. 使用基准测试工具进行性能对比
2. 使用量化器内置的评估方法
3. 自定义验证流程
4. 生成详细的对比报告
"""

import sys
import torch
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.model import YOLO11Model
from optimization.quantization.quantizers import create_quantizer, QuantizationUtils
from benchmarks.speed_benchmark import SpeedBenchmark
from utils.helpers import setup_logging, get_device_info

logger = logging.getLogger(__name__)

class QuantizationValidator:
    """量化方法验证器"""
    
    def __init__(
        self,
        model_path: str = None,
        model_size: str = 'n',
        task: str = 'detect',
        device: str = None,
        output_dir: str = 'quantization_validation_results'
    ):
        """
        初始化验证器
        
        Args:
            model_path: 模型路径（可选）
            model_size: 模型大小
            task: 任务类型
            device: 设备
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.model_size = model_size
        self.task = task
        self.device = device or self._get_default_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 系统信息
        self.system_info = get_device_info()
        
        # 验证结果
        self.validation_results = {}
        
    def _get_default_device(self) -> str:
        """获取默认设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def create_test_data(
        self,
        image_size: int = 640,
        batch_size: int = 1,
        num_samples: int = 100
    ) -> List[torch.Tensor]:
        """创建测试数据"""
        test_data = []
        for _ in range(num_samples):
            # 创建随机测试图像
            test_input = torch.randn(batch_size, 3, image_size, image_size)
            if self.device and self.device != 'cpu':
                test_input = test_input.to(self.device)
            test_data.append(test_input)
        return test_data
    
    def validate_single_quantization_method(
        self,
        quantization_method: str,
        test_data: List[torch.Tensor],
        calibration_data: Optional[List[torch.Tensor]] = None,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        验证单个量化方法
        
        Args:
            quantization_method: 量化方法 ('dynamic', 'ptq', 'qat')
            test_data: 测试数据
            calibration_data: 校准数据（PTQ需要）
            config: 量化配置
            
        Returns:
            验证结果字典
        """
        logger.info(f"验证量化方法: {quantization_method}")
        
        try:
            # 创建原始模型
            if self.model_path:
                original_model = YOLO11Model(model_path=self.model_path, device=self.device)
            else:
                original_model = YOLO11Model(task=self.task, size=self.model_size, device=self.device)
            
            # 评估原始模型
            original_metrics = self._evaluate_model_performance(original_model, test_data)
            
            # 创建量化器
            quantizer = create_quantizer(
                quantization_type=quantization_method,
                model=original_model,
                config=config or {},
                device=self.device
            )
            
            # 执行量化
            start_time = time.time()
            
            if quantization_method == 'ptq':
                if calibration_data is None:
                    calibration_data = test_data[:20]  # 使用前20个样本作为校准数据
                quantizer.set_calibration_data(calibration_data)
                quantized_model = quantizer.optimize()
            elif quantization_method == 'qat':
                # QAT需要训练数据，此处使用测试数据模拟
                train_data = test_data[:50]  # 使用前50个样本作为训练数据
                quantized_model = quantizer.optimize(train_loader=train_data)
            else:  # dynamic
                quantized_model = quantizer.optimize()
            
            quantization_time = time.time() - start_time
            
            # 评估量化后模型
            quantized_metrics = self._evaluate_model_performance(quantized_model, test_data)
            
            # 获取优化信息
            optimization_info = quantizer.get_optimization_info()
            
            # 计算性能对比
            performance_comparison = self._calculate_performance_comparison(
                original_metrics, quantized_metrics
            )
            
            # 模型大小对比
            size_comparison = QuantizationUtils.compare_model_sizes(
                original_model, quantized_model
            )
            
            result = {
                'method': quantization_method,
                'status': 'success',
                'quantization_time': quantization_time,
                'original_metrics': original_metrics,
                'quantized_metrics': quantized_metrics,
                'performance_comparison': performance_comparison,
                'size_comparison': size_comparison,
                'optimization_info': optimization_info,
                'config': config or {}
            }
            
            logger.info(f"{quantization_method} 量化验证完成")
            return result
            
        except Exception as e:
            logger.error(f"{quantization_method} 量化验证失败: {e}")
            return {
                'method': quantization_method,
                'status': 'failed',
                'error': str(e)
            }
    
    def _evaluate_model_performance(
        self,
        model: Any,
        test_data: List[torch.Tensor],
        warmup_runs: int = 10,
        benchmark_runs: int = 50
    ) -> Dict[str, float]:
        """评估模型性能"""
        # 预热
        model_eval = model.model if hasattr(model, 'model') else model
        model_eval.eval()
        
        with torch.no_grad():
            for i, test_input in enumerate(test_data[:warmup_runs]):
                try:
                    _ = model_eval(test_input)
                except Exception as e:
                    logger.warning(f"预热运行 {i} 失败: {e}")
        
        # 基准测试
        inference_times = []
        with torch.no_grad():
            for i, test_input in enumerate(test_data[:benchmark_runs]):
                try:
                    start_time = time.time()
                    _ = model_eval(test_input)
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                except Exception as e:
                    logger.warning(f"基准测试运行 {i} 失败: {e}")
        
        if not inference_times:
            return {'error': 'No successful inference runs'}
        
        # 计算性能指标
        metrics = {
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'fps': 1.0 / np.mean(inference_times),
            'successful_runs': len(inference_times),
            'total_runs': benchmark_runs
        }
        
        # 获取模型大小
        if hasattr(model, 'model'):
            model_size = self._calculate_model_size(model.model)
        else:
            model_size = self._calculate_model_size(model)
        metrics['model_size_mb'] = model_size
        
        return metrics
    
    def _calculate_model_size(self, model: Any) -> float:
        """计算模型大小（MB）"""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _calculate_performance_comparison(
        self,
        original_metrics: Dict[str, float],
        quantized_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """计算性能对比"""
        comparison = {}
        
        for key in original_metrics:
            if key in quantized_metrics and isinstance(original_metrics[key], (int, float)):
                original_val = original_metrics[key]
                quantized_val = quantized_metrics[key]
                
                if key == 'avg_inference_time':
                    # 推理时间：越小越好，计算加速比
                    speedup = original_val / quantized_val if quantized_val > 0 else 0
                    comparison[f'{key}_speedup'] = speedup
                    comparison[f'{key}_improvement_percent'] = ((original_val - quantized_val) / original_val) * 100 if original_val > 0 else 0
                elif key == 'model_size_mb':
                    # 模型大小：越小越好，计算压缩比
                    compression_ratio = original_val / quantized_val if quantized_val > 0 else 0
                    comparison[f'{key}_compression_ratio'] = compression_ratio
                    comparison[f'{key}_reduction_percent'] = ((original_val - quantized_val) / original_val) * 100 if original_val > 0 else 0
                elif key == 'fps':
                    # FPS：越大越好
                    improvement = quantized_val / original_val if original_val > 0 else 0
                    comparison[f'{key}_improvement_ratio'] = improvement
                    comparison[f'{key}_improvement_percent'] = ((quantized_val - original_val) / original_val) * 100 if original_val > 0 else 0
                
                comparison[f'{key}_absolute_change'] = quantized_val - original_val
        
        return comparison
    
    def validate_all_methods(
        self,
        methods: List[str] = ['dynamic', 'ptq'],
        test_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        验证所有量化方法
        
        Args:
            methods: 要验证的量化方法列表
            test_config: 测试配置
            
        Returns:
            完整的验证结果
        """
        config = test_config or {}
        image_size = config.get('image_size', 640)
        batch_size = config.get('batch_size', 1)
        num_samples = config.get('num_samples', 100)
        
        logger.info("开始验证所有量化方法...")
        logger.info(f"测试配置: image_size={image_size}, batch_size={batch_size}, samples={num_samples}")
        
        # 创建测试数据
        logger.info("创建测试数据...")
        test_data = self.create_test_data(image_size, batch_size, num_samples)
        calibration_data = test_data[:20]  # 前20个样本用作校准数据
        
        # 验证结果
        results = {
            'system_info': self.system_info,
            'test_config': {
                'image_size': image_size,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'device': self.device,
                'model_info': {
                    'model_path': self.model_path,
                    'model_size': self.model_size,
                    'task': self.task
                }
            },
            'methods': {},
            'summary': {}
        }
        
        # 逐个验证量化方法
        for method in methods:
            logger.info(f"正在验证 {method} 量化方法...")
            
            method_config = config.get(f'{method}_config', {})
            method_result = self.validate_single_quantization_method(
                quantization_method=method,
                test_data=test_data,
                calibration_data=calibration_data,
                config=method_config
            )
            
            results['methods'][method] = method_result
        
        # 生成总结
        results['summary'] = self._generate_summary(results['methods'])
        
        # 保存结果
        self._save_results(results)
        
        logger.info("所有量化方法验证完成")
        return results
    
    def _generate_summary(self, methods_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证总结"""
        summary = {
            'successful_methods': [],
            'failed_methods': [],
            'best_performance': {},
            'comparison_table': []
        }
        
        successful_results = {}
        
        for method, result in methods_results.items():
            if result.get('status') == 'success':
                summary['successful_methods'].append(method)
                successful_results[method] = result
            else:
                summary['failed_methods'].append(method)
        
        if successful_results:
            # 找出最佳性能方法
            for metric in ['avg_inference_time', 'model_size_mb', 'fps']:
                best_method = None
                best_value = None
                
                for method, result in successful_results.items():
                    if 'quantized_metrics' in result and metric in result['quantized_metrics']:
                        value = result['quantized_metrics'][metric]
                        
                        # 对于推理时间和模型大小，越小越好；对于FPS，越大越好
                        if metric in ['avg_inference_time', 'model_size_mb']:
                            if best_value is None or value < best_value:
                                best_value = value
                                best_method = method
                        elif metric == 'fps':
                            if best_value is None or value > best_value:
                                best_value = value
                                best_method = method
                
                if best_method:
                    summary['best_performance'][metric] = {
                        'method': best_method,
                        'value': best_value
                    }
            
            # 创建对比表
            comparison_table = []
            for method, result in successful_results.items():
                if 'quantized_metrics' in result:
                    metrics = result['quantized_metrics']
                    comparison_data = {
                        'method': method,
                        'avg_inference_time': metrics.get('avg_inference_time', 0),
                        'fps': metrics.get('fps', 0),
                        'model_size_mb': metrics.get('model_size_mb', 0),
                        'successful_runs': metrics.get('successful_runs', 0)
                    }
                    
                    # 添加性能改进信息
                    if 'performance_comparison' in result:
                        perf_comp = result['performance_comparison']
                        comparison_data.update({
                            'speedup': perf_comp.get('avg_inference_time_speedup', 1.0),
                            'compression_ratio': perf_comp.get('model_size_mb_compression_ratio', 1.0)
                        })
                    
                    comparison_table.append(comparison_data)
            
            summary['comparison_table'] = comparison_table
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存验证结果"""
        # 保存JSON格式的详细结果
        json_path = self.output_dir / 'quantization_validation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可读的报告
        report_path = self.output_dir / 'quantization_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_readable_report(f, results)
        
        logger.info(f"验证结果已保存到: {self.output_dir}")
        logger.info(f"详细结果: {json_path}")
        logger.info(f"可读报告: {report_path}")
    
    def _write_readable_report(self, file, results: Dict[str, Any]) -> None:
        """写入可读的报告"""
        file.write("量化方法验证报告\n")
        file.write("=" * 50 + "\n\n")
        
        # 系统信息
        file.write("系统信息:\n")
        for key, value in results['system_info'].items():
            if key != 'gpus':
                file.write(f"  {key}: {value}\n")
        file.write("\n")
        
        # 测试配置
        file.write("测试配置:\n")
        for key, value in results['test_config'].items():
            if isinstance(value, dict):
                file.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"    {sub_key}: {sub_value}\n")
            else:
                file.write(f"  {key}: {value}\n")
        file.write("\n")
        
        # 验证结果
        file.write("验证结果:\n")
        file.write("-" * 30 + "\n")
        
        for method, result in results['methods'].items():
            file.write(f"\n{method.upper()} 量化方法:\n")
            
            if result.get('status') == 'success':
                # 原始模型性能
                if 'original_metrics' in result:
                    orig = result['original_metrics']
                    file.write(f"  原始模型:\n")
                    file.write(f"    推理时间: {orig.get('avg_inference_time', 0):.4f}s\n")
                    file.write(f"    FPS: {orig.get('fps', 0):.2f}\n")
                    file.write(f"    模型大小: {orig.get('model_size_mb', 0):.2f}MB\n")
                
                # 量化后模型性能
                if 'quantized_metrics' in result:
                    quant = result['quantized_metrics']
                    file.write(f"  量化后模型:\n")
                    file.write(f"    推理时间: {quant.get('avg_inference_time', 0):.4f}s\n")
                    file.write(f"    FPS: {quant.get('fps', 0):.2f}\n")
                    file.write(f"    模型大小: {quant.get('model_size_mb', 0):.2f}MB\n")
                
                # 性能改进
                if 'performance_comparison' in result:
                    comp = result['performance_comparison']
                    file.write(f"  性能改进:\n")
                    if 'avg_inference_time_speedup' in comp:
                        file.write(f"    推理加速: {comp['avg_inference_time_speedup']:.2f}x\n")
                    if 'model_size_mb_compression_ratio' in comp:
                        file.write(f"    模型压缩: {comp['model_size_mb_compression_ratio']:.2f}x\n")
                
            else:
                file.write(f"  状态: 失败\n")
                file.write(f"  错误: {result.get('error', 'Unknown error')}\n")
        
        # 总结
        if 'summary' in results:
            summary = results['summary']
            file.write(f"\n总结:\n")
            file.write("-" * 30 + "\n")
            file.write(f"成功的方法: {', '.join(summary['successful_methods'])}\n")
            if summary['failed_methods']:
                file.write(f"失败的方法: {', '.join(summary['failed_methods'])}\n")
            
            if summary['best_performance']:
                file.write(f"\n最佳性能:\n")
                for metric, info in summary['best_performance'].items():
                    file.write(f"  {metric}: {info['method']} ({info['value']:.4f})\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='量化方法验证工具')
    parser.add_argument('--model-path', type=str, help='模型路径')
    parser.add_argument('--model-size', type=str, default='n', help='模型大小')
    parser.add_argument('--task', type=str, default='detect', help='任务类型')
    parser.add_argument('--device', type=str, help='设备')
    parser.add_argument('--methods', nargs='+', default=['dynamic', 'ptq'], 
                       help='要验证的量化方法')
    parser.add_argument('--image-size', type=int, default=640, help='图像大小')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--num-samples', type=int, default=100, help='测试样本数')
    parser.add_argument('--output-dir', type=str, default='quantization_validation_results',
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level)
    
    # 创建验证器
    validator = QuantizationValidator(
        model_path=args.model_path,
        model_size=args.model_size,
        task=args.task,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 执行验证
    test_config = {
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_samples': args.num_samples
    }
    
    results = validator.validate_all_methods(
        methods=args.methods,
        test_config=test_config
    )
    
    # 打印总结
    print("\n验证完成！")
    print(f"结果保存在: {args.output_dir}")
    
    if results.get('summary'):
        summary = results['summary']
        print(f"\n成功验证的方法: {', '.join(summary['successful_methods'])}")
        if summary['failed_methods']:
            print(f"失败的方法: {', '.join(summary['failed_methods'])}")
        
        if summary['best_performance']:
            print(f"\n最佳性能:")
            for metric, info in summary['best_performance'].items():
                print(f"  {metric}: {info['method']} ({info['value']:.4f})")

if __name__ == '__main__':
    main()