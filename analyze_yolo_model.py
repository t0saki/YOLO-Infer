import torch
import sys
sys.path.append('.')

from core.model import YOLO11Model

# Create a small YOLO model
model = YOLO11Model(task='detect', size='n', device='cpu')

print("=== YOLO Model Analysis ===")
print(f"Model type: {type(model)}")
if hasattr(model, 'model'):
    pytorch_model = model.model
    print(f"PyTorch model type: {type(pytorch_model)}")
    
    # Check if the model has a specific structure that might prevent quantization
    print("\nModel attributes:")
    for attr in dir(pytorch_model):
        if not attr.startswith('_'):
            try:
                attr_value = getattr(pytorch_model, attr)
                if not callable(attr_value):
                    print(f"  {attr}: {type(attr_value)}")
            except:
                print(f"  {attr}: <error accessing>")
    
    # Check model's modules for any special attributes
    print("\nModel module types:")
    for name, module in pytorch_model.named_modules():
        if len(name) > 0 and len(name) < 50:  # Only print shorter names for readability
            # Check if module has special attributes that might prevent quantization
            special_attrs = []
            for attr in ['_is_full_backward_hook', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks']:
                if hasattr(module, attr):
                    special_attrs.append(attr)
            
            if special_attrs:
                print(f"  {name}: {type(module)} - Special attrs: {special_attrs}")
            else:
                print(f"  {name}: {type(module)}")
                
    # Try to check if model is scriptable (sometimes this can reveal issues)
    print("\nChecking if model is scriptable:")
    try:
        scripted_model = torch.jit.script(pytorch_model)
        print("  Model is scriptable")
    except Exception as e:
        print(f"  Model is not scriptable: {e}")
        
    # Try to check if model can be copied (sometimes this can reveal issues with pickling)
    print("\nChecking if model can be copied:")
    try:
        import copy
        copied_model = copy.deepcopy(pytorch_model)
        print("  Model can be copied")
    except Exception as e:
        print(f"  Model cannot be copied: {e}")