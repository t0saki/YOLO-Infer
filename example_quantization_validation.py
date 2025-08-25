#!/usr/bin/env python3
"""
量化验证示例脚本
展示如何使用项目现有工具进行量化验证
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from core.model import YOLO11Model
from optimization.quantization.quantizers import create_quantizer, QuantizationUtils
from benchmarks.speed_benchmark import SpeedBenchmark

def example_quantization_validation():
    """量化验证示例"""
    print("开始量化验证示例...")
    
    # 1. 创建原始模型
    print("\n1. 创建原始模型")
    original_model = YOLO11Model(task='detect', size='n', device='cpu')
    print(f"原始模型创建完成: {original_model}")
    
    # 2. 创建测试数据
    print("\n2. 创建测试数据")
    test_input = torch.randn(1, 3, 640, 640)
    print(f"测试数据形状: {test_input.shape}")
    
    # 3. 验证动态量化
    print("\n3. 验证动态量化")
    dynamic_quantizer = create_quantizer('dynamic', original_model)
    dynamic_model = dynamic_quantizer.optimize()
    
    # 评估动态量化模型
    dynamic_info = dynamic_quantizer.get_optimization_info()
    print(f"动态量化信息: {dynamic_info['metrics']}")
    
    # 4. 验证PTQ量化
    print("\n4. 验证PTQ量化")
    ptq_quantizer = create_quantizer('ptq', original_model)
    
    # 创建校准数据
    calibration_data = [torch.randn(1, 3, 640, 640) for _ in range(10)]
    ptq_quantizer.set_calibration_data(calibration_data)
    
    try:
        ptq_model = ptq_quantizer.optimize()
        ptq_info = ptq_quantizer.get_optimization_info()
        print(f"PTQ量化信息: {ptq_info['metrics']}")
    except Exception as e:
        print(f"PTQ量化失败: {e}")
    
    # 5. 模型大小对比
    print("\n5. 模型大小对比")
    try:
        size_comparison = QuantizationUtils.compare_model_sizes(original_model, dynamic_model)
        print(f"模型大小对比:")
        print(f"  原始模型: {size_comparison['original_size_mb']:.2f} MB")
        print(f"  量化模型: {size_comparison['quantized_size_mb']:.2f} MB")
        print(f"  压缩比: {size_comparison['compression_ratio']:.2f}x")
        print(f"  大小减少: {size_comparison['size_reduction_percent']:.1f}%")
    except Exception as e:
        print(f"模型大小对比失败: {e}")
    
    # 6. 推理速度对比
    print("\n6. 推理速度对比")
    try:
        # 原始模型速度
        original_speed = QuantizationUtils.benchmark_inference_speed(original_model, test_input, num_runs=50)
        print(f"原始模型推理速度:")
        print(f"  平均推理时间: {original_speed['avg_inference_time']:.4f}s")
        print(f"  FPS: {original_speed['fps']:.2f}")
        
        # 量化模型速度
        dynamic_speed = QuantizationUtils.benchmark_inference_speed(dynamic_model, test_input, num_runs=50)
        print(f"动态量化模型推理速度:")
        print(f"  平均推理时间: {dynamic_speed['avg_inference_time']:.4f}s")
        print(f"  FPS: {dynamic_speed['fps']:.2f}")
        
        # 计算加速比
        speedup = original_speed['avg_inference_time'] / dynamic_speed['avg_inference_time']
        print(f"  加速比: {speedup:.2f}x")
        
    except Exception as e:
        print(f"推理速度对比失败: {e}")
    
    print("\n量化验证示例完成!")

def example_using_speed_benchmark():
    """使用SpeedBenchmark进行量化验证"""
    print("\n使用SpeedBenchmark进行量化验证...")
    
    try:
        # 创建基准测试器
        benchmark = SpeedBenchmark(
            output_dir='benchmark_results',
            warmup_runs=5,
            benchmark_runs=20,
            device='cpu'
        )
        
        # 运行量化基准测试
        results = benchmark.benchmark_quantization(
            model_size='n',
            task='detect',
            quantization_methods=['dynamic', 'ptq'],
            image_size=640,
            batch_size=1
        )
        
        print("量化基准测试完成!")
        print(f"结果保存在: benchmark_results/")
        
        # 打印简要结果
        for method, result in results['methods'].items():
            if 'error' not in result:
                print(f"\n{method}:")
                print(f"  推理时间: {result.get('avg_inference_time', 0):.4f}s")
                print(f"  FPS: {result.get('fps', 0):.2f}")
                if 'speedup' in result:
                    print(f"  加速比: {result['speedup']:.2f}x")
            else:
                print(f"{method}: 测试失败 - {result['error']}")
                
    except Exception as e:
        print(f"SpeedBenchmark测试失败: {e}")

if __name__ == '__main__':
    # 运行验证示例
    example_quantization_validation()
    
    # 使用SpeedBenchmark
    example_using_speed_benchmark()