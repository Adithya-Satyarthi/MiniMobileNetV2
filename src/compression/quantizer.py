"""
Custom Post-Training Quantization (PTQ) for MobileNetV2
Implements PTQ from scratch without using PyTorch quantization API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def quantize_tensor(tensor, bits=8, symmetric=True):
    """Quantize tensor using uniform quantization"""
    if bits >= 32:
        return tensor, torch.tensor(1.0, device=tensor.device), torch.tensor(0.0, device=tensor.device)
    
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        max_val = torch.max(torch.abs(tensor)).clamp(min=1e-8)
        scale = max_val / qmax
        zero_point = torch.zeros_like(scale)  # Same device as tensor
    else:
        qmin = 0
        qmax = (2 ** bits) - 1
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = ((max_val - min_val) / qmax).clamp(min=1e-8)
        zero_point = qmin - min_val / scale
    
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    
    return dequantized, scale, zero_point


class QuantizedConv2d(nn.Module):
    """Conv2d with weight and activation quantization for PTQ"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, weight_bits=8, activation_bits=8):
        super().__init__()
        
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.groups = groups
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        self.quantized = False
    
    def quantize_weights(self):
        """Quantize weights per-channel"""
        if self.weight_bits >= 32:
            return
        
        with torch.no_grad():
            for i in range(self.weight.shape[0]):
                self.weight[i], _, _ = quantize_tensor(self.weight[i], self.weight_bits, symmetric=True)
            self.quantized = True
    
    def forward(self, x):
        if self.quantized and self.activation_bits < 32:
            x, _, _ = quantize_tensor(x, self.activation_bits, symmetric=False)
        
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, groups=self.groups)
    
    @classmethod
    def from_float(cls, conv, weight_bits=8, activation_bits=8):
        """Create quantized conv from float conv"""
        quant = cls(
            conv.in_channels, conv.out_channels,
            conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size,
            stride=conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
            padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
            groups=conv.groups,
            bias=conv.bias is not None,
            weight_bits=weight_bits,
            activation_bits=activation_bits
        )
        quant.weight.data = conv.weight.data.clone()
        if conv.bias is not None:
            quant.bias.data = conv.bias.data.clone()
        return quant


class QuantizedLinear(nn.Module):
    """Linear with weight and activation quantization for PTQ"""
    
    def __init__(self, in_features, out_features, bias=True, weight_bits=8, activation_bits=8):
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        self.quantized = False
    
    def quantize_weights(self):
        """Quantize weights"""
        if self.weight_bits >= 32:
            return
        
        with torch.no_grad():
            self.weight.data, _, _ = quantize_tensor(self.weight.data, self.weight_bits, symmetric=True)
            self.quantized = True
    
    def forward(self, x):
        if self.quantized and self.activation_bits < 32:
            x, _, _ = quantize_tensor(x, self.activation_bits, symmetric=False)
        return F.linear(x, self.weight, self.bias)
    
    @classmethod
    def from_float(cls, linear, weight_bits=8, activation_bits=8):
        """Create quantized linear from float linear"""
        quant = cls(linear.in_features, linear.out_features,
                   bias=linear.bias is not None,
                   weight_bits=weight_bits, activation_bits=activation_bits)
        quant.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            quant.bias.data = linear.bias.data.clone()
        return quant


def get_layer_bits(path, config):
    """Determine bit-width config based on layer path in MobileNetV2"""
    if 'features.0' in path:
        return config.get('first_conv', {'weight_bits': 8, 'activation_bits': 8})
    elif 'features.18' in path:
        return config.get('final_conv', {'weight_bits': 8, 'activation_bits': 8})
    elif 'classifier' in path:
        return config.get('classifier', {'weight_bits': 8, 'activation_bits': 8})
    return config.get('inverted_residual', {'weight_bits': 8, 'activation_bits': 8})


def replace_with_quantized(model, config, path=''):
    """
    Recursively replace Conv2d/Linear with quantized versions.
    Handles MobileNetV2's nested structure properly.
    """
    for name, module in model.named_children():
        full_path = f"{path}.{name}" if path else name
        
        if isinstance(module, nn.Conv2d):
            bits = get_layer_bits(full_path, config)
            quant = QuantizedConv2d.from_float(
                module, 
                weight_bits=bits['weight_bits'], 
                activation_bits=bits['activation_bits']
            )
            setattr(model, name, quant)
            
        elif isinstance(module, nn.Linear):
            bits = get_layer_bits(full_path, config)
            quant = QuantizedLinear.from_float(
                module, 
                weight_bits=bits['weight_bits'], 
                activation_bits=bits['activation_bits']
            )
            setattr(model, name, quant)
            
        else:
            replace_with_quantized(module, config, full_path)
    
    return model


class PTQQuantizer:
    """Post-Training Quantization"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def calibrate(self, data_loader):
        """Run calibration for min-max statistics"""
        print("\nCalibrating...")
        self.model.eval()
        device = next(self.model.parameters()).device
        max_batches = self.config['quantization']['ptq']['calibration_batches']
        
        with torch.no_grad():
            for idx, (data, _) in enumerate(data_loader):
                if idx >= max_batches:
                    break
                _ = self.model(data.to(device))
                if (idx + 1) % 20 == 0:
                    print(f"  {idx + 1}/{max_batches} batches")
        
        print("Calibration complete")
    
    def quantize(self):
        """Apply PTQ quantization"""
        print("\nQuantizing model...")
        
        bits_config = self.config['quantization']['bits']
        
        quantized = copy.deepcopy(self.model)
        replace_with_quantized(quantized, bits_config)
        
        count = 0
        for module in quantized.modules():
            if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                module.quantize_weights()
                count += 1
        
        print(f"Quantized {count} layers")
        return quantized

