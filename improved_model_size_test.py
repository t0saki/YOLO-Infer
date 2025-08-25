import torch
import sys
sys.path.append('.')

from core.model import YOLO11Model
from optimization.quantization.quantizers import create_quantizer

def improved_model_size_test():
    print("=== Improved Model Size Test ===")
    
    # Create a small YOLO model
    model = YOLO11Model(task='detect', size='n', device='cpu')
    print(f"Original model type: {type(model)}")
    
    if hasattr(model, 'model'):
        pytorch_model = model.model
        print(f"PyTorch model type: {type(pytorch_model)}")
        
        # Calculate original model size using different methods
        print("\nOriginal model size calculations:")
        
        # Method 1: torch.save to buffer (as in our implementation)
        try:
            import io
            buffer = io.BytesIO()
            torch.save(pytorch_model, buffer)
            size_bytes = buffer.tell()
            size_mb_v1 = size_bytes / (1024 * 1024)
            print(f"  Method 1 (torch.save): {size_mb_v1:.4f} MB")
        except Exception as e:
            print(f"  Method 1 failed: {e}")
            size_mb_v1 = 0
            
        # Method 2: Parameter-based calculation (as in our implementation)
        param_size = 0
        buffer_size = 0
        for param in pytorch_model.parameters():
            try:
                param_size += param.nelement() * param.element_size()
            except Exception:
                pass
                
        for buffer_obj in pytorch_model.buffers():
            try:
                buffer_size += buffer_obj.nelement() * buffer_obj.element_size()
            except Exception:
                pass
                
        size_mb_v2 = (param_size + buffer_size) / (1024 * 1024)
        print(f"  Method 2 (param-based): {size_mb_v2:.4f} MB")
        
        # Check if model is quantized (PyTorch 2.x style)
        is_quantized = False
        quantized_modules = []
        for name, module in pytorch_model.named_modules():
            if hasattr(module, '_packed_params'):
                is_quantized = True
                quantized_modules.append(name)
            elif hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                if 'qint' in str(module.weight.dtype):
                    is_quantized = True
                    quantized_modules.append(name)
                    
        print(f"  Is quantized: {is_quantized}")
        if quantized_modules:
            print(f"  Quantized modules found: {len(quantized_modules)}")
            for name in quantized_modules[:5]:  # Show first 5
                print(f"    {name}")
            if len(quantized_modules) > 5:
                print(f"    ... and {len(quantized_modules) - 5} more")
        
        # Apply dynamic quantization
        print("\nApplying dynamic quantization...")
        torch.backends.quantized.engine = 'qnnpack'
        quantizer = create_quantizer('dynamic', model, device='cpu')
        quantized_model = quantizer.optimize()
        
        print("Quantization completed.")
        
        # Calculate quantized model size using different methods
        print("\nQuantized model size calculations:")
        
        if hasattr(quantized_model, 'model'):
            quantized_pytorch_model = quantized_model.model
            
            # Method 1: torch.save to buffer
            try:
                buffer = io.BytesIO()
                torch.save(quantized_pytorch_model, buffer)
                size_bytes = buffer.tell()
                quant_size_mb_v1 = size_bytes / (1024 * 1024)
                print(f"  Method 1 (torch.save): {quant_size_mb_v1:.4f} MB")
            except Exception as e:
                print(f"  Method 1 failed: {e}")
                quant_size_mb_v1 = 0
                
            # Method 2: Parameter-based calculation with quantization check
            param_size = 0
            buffer_size = 0
            
            # Check if quantized model is quantized
            is_quantized_after = False
            quantized_modules_after = []
            for name, module in quantized_pytorch_model.named_modules():
                if hasattr(module, '_packed_params'):
                    is_quantized_after = True
                    quantized_modules_after.append(name)
                elif hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                    if 'qint' in str(module.weight.dtype):
                        is_quantized_after = True
                        quantized_modules_after.append(name)
                        
            print(f"  Is quantized after: {is_quantized_after}")
            if quantized_modules_after:
                print(f"  Quantized modules found after: {len(quantized_modules_after)}")
                for name in quantized_modules_after[:5]:  # Show first 5
                    print(f"    {name}")
                if len(quantized_modules_after) > 5:
                    print(f"    ... and {len(quantized_modules_after) - 5} more")
            
            # Use our implementation's fallback method
            if is_quantized_after:
                # For quantized models, use more conservative estimates
                for module in quantized_pytorch_model.modules():
                    # Process parameters
                    for param in module.parameters(recurse=False):  # Only direct parameters
                        if param is not None:
                            try:
                                if hasattr(param, 'dtype') and ('qint' in str(param.dtype) or 'quint' in str(param.dtype)):
                                    # Quantized parameter - use 1 byte per element for int8/uint8
                                    param_size += param.nelement() * 1
                                else:
                                    # Regular parameter
                                    param_size += param.nelement() * param.element_size()
                            except Exception:
                                # Fallback
                                try:
                                    param_size += param.nelement() * param.element_size()
                                except:
                                    pass
                    # Process buffers
                    for buffer_obj in module.buffers(recurse=False):  # Only direct buffers
                        if buffer_obj is not None:
                            try:
                                buffer_size += buffer_obj.nelement() * buffer_obj.element_size()
                            except Exception:
                                pass
            else:
                # For non-quantized models, use standard calculation
                for param in quantized_pytorch_model.parameters():
                    try:
                        param_size += param.nelement() * param.element_size()
                    except Exception:
                        pass
                        
                for buffer_obj in quantized_pytorch_model.buffers():
                    try:
                        buffer_size += buffer_obj.nelement() * buffer_obj.element_size()
                    except Exception:
                        pass
                        
            quant_size_mb_v2 = (param_size + buffer_size) / (1024 * 1024)
            print(f"  Method 2 (param-based with quant check): {quant_size_mb_v2:.4f} MB")
            
            # Compare sizes
            if size_mb_v2 > 0 and quant_size_mb_v2 > 0:
                reduction = size_mb_v2 - quant_size_mb_v2
                reduction_percent = (reduction / size_mb_v2) * 100
                print(f"\nSize reduction: {reduction:.4f} MB ({reduction_percent:.2f}%)")
            else:
                print("\nCould not calculate size reduction")

if __name__ == "__main__":
    improved_model_size_test()