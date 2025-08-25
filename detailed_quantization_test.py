import torch
import torch.nn as nn

# Create a simple model that mimics the structure we want to quantize
class SimpleQuantTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers like in YOLO
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Batch norm layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        # Activation
        self.relu = nn.ReLU()
        # Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final classifier
        self.classifier = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def detailed_quantization_test():
    print("=== Detailed Quantization Test ===")
    
    # Create original model
    original_model = SimpleQuantTestModel()
    print(f"Original model type: {type(original_model)}")
    
    # Print original model structure
    print("\nOriginal model layers:")
    for name, module in original_model.named_modules():
        if len(name) > 0 and len(name) < 30:
            attrs = []
            if hasattr(module, 'weight') and module.weight is not None:
                attrs.append(f"weight_dtype={module.weight.dtype}")
            if hasattr(module, '_packed_params'):
                attrs.append("_packed_params=True")
            if hasattr(module, 'bias') and module.bias is not None:
                attrs.append("has_bias=True")
                
            print(f"  {name}: {type(module).__name__} {attrs}")
    
    # Apply dynamic quantization using PyTorch directly
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.quantize_dynamic(
        original_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    print("\nQuantized model layers:")
    for name, module in quantized_model.named_modules():
        if len(name) > 0 and len(name) < 30:
            attrs = []
            if hasattr(module, 'weight') and module.weight is not None:
                attrs.append(f"weight_dtype={module.weight.dtype}")
            if hasattr(module, '_packed_params'):
                attrs.append("_packed_params=True")
            if hasattr(module, 'bias') and module.bias is not None:
                attrs.append("has_bias=True")
                
            print(f"  {name}: {type(module).__name__} {attrs}")
    
    # Check for _packed_params which indicates quantized layers in PyTorch 2.x
    print("\nChecking for quantized layers (PyTorch 2.x style):")
    quantized_layers = []
    for name, module in quantized_model.named_modules():
        if hasattr(module, '_packed_params'):
            quantized_layers.append(name)
            print(f"  Found quantized layer: {name}")
            
    print(f"Total quantized layers found: {len(quantized_layers)}")
    
    # Test if we can access packed parameters
    if quantized_layers:
        for name in quantized_layers[:3]:  # Check first 3
            module = dict(quantized_model.named_modules())[name]
            try:
                packed_params = module._packed_params
                print(f"  {name}._packed_params: {type(packed_params)}")
            except Exception as e:
                print(f"  {name}._packed_params access failed: {e}")

if __name__ == "__main__":
    detailed_quantization_test()