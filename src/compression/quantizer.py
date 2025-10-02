import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, Optional

class SymmetricUniformQuantizer:
    """
    Custom symmetric uniform quantization implementation.
    No external APIs - pure PyTorch implementation as required.
    """
    
    def __init__(self, n_bits: int, signed: bool = True):
        """
        Args:
            n_bits: Number of quantization bits (2-8)
            signed: Whether to use signed quantization
        """
        self.n_bits = n_bits
        self.signed = signed
        
        if signed:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1
            
        self.scale = None
    
    def calibrate(self, tensor: torch.Tensor) -> None:
        """
        Calibrate quantization scale from tensor statistics.
        Uses symmetric quantization: scale = max(|min|, |max|) / qmax
        """
        with torch.no_grad():
            # Symmetric quantization scale calculation
            abs_max = torch.max(torch.abs(tensor))
            if abs_max == 0:
                self.scale = 1.0
            else:
                if self.signed:
                    self.scale = abs_max / (2 ** (self.n_bits - 1) - 1)
                else:
                    self.scale = abs_max / (2 ** self.n_bits - 1)
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor to n_bits using symmetric quantization.
        Formula: q = clamp(round(x / scale), qmin, qmax)
        """
        if self.scale is None:
            raise ValueError("Must calibrate quantizer before quantization")
        
        with torch.no_grad():
            # Symmetric quantization: q = clamp(round(x / scale), qmin, qmax)
            quantized = torch.round(tensor / self.scale)
            quantized = torch.clamp(quantized, self.qmin, self.qmax)
            return quantized
    
    def dequantize(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize back to floating point.
        Formula: x = q * scale
        """
        if self.scale is None:
            raise ValueError("Must calibrate quantizer before dequantization")
        
        return quantized_tensor * self.scale
    
    def get_metadata(self) -> Dict:
        """Return quantization metadata for storage overhead calculation"""
        return {
            'n_bits': self.n_bits,
            'signed': self.signed,
            'scale': float(self.scale) if self.scale is not None else None,
            'qmin': self.qmin,
            'qmax': self.qmax
        }

class ActivationQuantizer(nn.Module):
    """
    Quantization module for activations during forward pass.
    Handles dynamic quantization of activations.
    """
    
    def __init__(self, n_bits: int = 4, signed: bool = True):
        super().__init__()
        self.n_bits = n_bits
        self.signed = signed
        self.quantizer = SymmetricUniformQuantizer(n_bits, signed)
        
        # Statistics for calibration
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.register_buffer('calibrated', torch.tensor(False))
        self.calibration_steps = 0
        self.max_calibration_steps = 100
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.calibrated:
            # Calibration phase: collect statistics
            with torch.no_grad():
                batch_min = torch.min(x)
                batch_max = torch.max(x)
                
                self.running_min = torch.min(self.running_min, batch_min)
                self.running_max = torch.max(self.running_max, batch_max)
                
                self.calibration_steps += 1
                
                if self.calibration_steps >= self.max_calibration_steps:
                    # Finalize calibration
                    abs_max = torch.max(torch.abs(self.running_min), 
                                      torch.abs(self.running_max))
                    
                    if abs_max == 0:
                        self.quantizer.scale = 1.0
                    else:
                        if self.signed:
                            self.quantizer.scale = abs_max / (2 ** (self.n_bits - 1) - 1)
                        else:
                            self.quantizer.scale = abs_max / (2 ** self.n_bits - 1)
                    
                    self.calibrated = True
            
            return x  # During calibration, return original values
        
        elif self.calibrated:
            # Quantization phase: apply quantization
            with torch.no_grad():
                quantized = self.quantizer.quantize(x)
                dequantized = self.quantizer.dequantize(quantized)
            return dequantized
        else:
            # Inference without calibration
            return x

class WeightQuantizer:
    """
    Static quantization for model weights.
    Quantizes weights once and stores quantized values.
    """
    
    def __init__(self, n_bits: int = 4, signed: bool = True):
        self.n_bits = n_bits
        self.signed = signed
        self.quantizer = SymmetricUniformQuantizer(n_bits, signed)
        
    def quantize_layer(self, layer: nn.Module) -> Dict:
        """
        Quantize a single layer's weights and return quantization info.
        """
        if not hasattr(layer, 'weight') or layer.weight is None:
            return {}
        
        original_weight = layer.weight.data.clone()
        
        # Calibrate and quantize
        self.quantizer.calibrate(original_weight)
        quantized_weight = self.quantizer.quantize(original_weight)
        
        # Replace original weights with dequantized values
        layer.weight.data = self.quantizer.dequantize(quantized_weight)
        
        return {
            'original_shape': original_weight.shape,
            'quantized_weight': quantized_weight,
            'scale': self.quantizer.scale,
            'metadata': self.quantizer.get_metadata()
        }

class MobileNetV2Quantizer:
    """
    Complete quantization pipeline for MobileNetV2.
    Handles both weight and activation quantization with configurable bit-widths.
    """
    
    def __init__(self, weight_bits: int = 4, activation_bits: int = 4):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_quantizer = WeightQuantizer(weight_bits, signed=True)
        
        # Storage for quantization metadata (for overhead calculation)
        self.quantization_info = {
            'weights': {},
            'activations': {},
            'total_params': 0,
            'quantized_layers': []
        }
    
    def apply_weight_quantization(self, model: nn.Module) -> None:
        """
        Apply weight quantization to all quantizable layers in MobileNetV2.
        """
        layer_count = 0
        
        for name, module in model.named_modules():
            # Quantize Conv2d and Linear layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    print(f"Quantizing weights in layer: {name}")
                    
                    quant_info = self.weight_quantizer.quantize_layer(module)
                    if quant_info:
                        self.quantization_info['weights'][name] = quant_info
                        self.quantization_info['quantized_layers'].append(name)
                        layer_count += 1
        
        print(f"Quantized weights in {layer_count} layers")
    
    def apply_activation_quantization(self, model: nn.Module) -> None:
        """
        Insert activation quantization modules after ReLU layers.
        """
        def add_activation_quantizers(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, (nn.ReLU, nn.ReLU6)):
                    # Replace ReLU with ReLU + ActivationQuantizer
                    setattr(module, child_name, nn.Sequential(
                        child,
                        ActivationQuantizer(self.activation_bits, signed=False)
                    ))
                    print(f"Added activation quantizer after: {full_name}")
                else:
                    add_activation_quantizers(child, full_name)
        
        add_activation_quantizers(model)
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply complete quantization pipeline to MobileNetV2.
        """
        print(f"Applying W{self.weight_bits}A{self.activation_bits} quantization to MobileNetV2")
        
        # Count original parameters
        self.quantization_info['total_params'] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Apply weight quantization
        self.apply_weight_quantization(model)
        
        # Apply activation quantization
        self.apply_activation_quantization(model)
        
        return model
    
    def calculate_storage_overhead(self) -> Dict:
        """
        Calculate storage overhead from quantization metadata.
        Required for Question 2c.
        """
        total_scales = len(self.quantization_info['weights'])  # One scale per layer
        total_metadata_params = len(self.quantization_info['quantized_layers'])
        
        # Each scale: 4 bytes (float32)
        # Each metadata entry: ~16 bytes (n_bits, signed, qmin, qmax)
        scale_overhead_bytes = total_scales * 4
        metadata_overhead_bytes = total_metadata_params * 16
        
        return {
            'scale_factors_bytes': scale_overhead_bytes,
            'metadata_bytes': metadata_overhead_bytes,
            'total_overhead_bytes': scale_overhead_bytes + metadata_overhead_bytes,
            'total_overhead_mb': (scale_overhead_bytes + metadata_overhead_bytes) / (1024 * 1024),
            'num_quantized_layers': len(self.quantization_info['quantized_layers'])
        }
    
    def get_compression_stats(self, original_model_size_mb: float) -> Dict:
        """
        Calculate comprehensive compression statistics.
        Required for Questions 3 and 4.
        """
        overhead = self.calculate_storage_overhead()
        
        # Calculate theoretical compressed size
        # Original: 32-bit floats, Quantized: n-bit integers
        weight_compression_ratio = 32.0 / self.weight_bits
        activation_compression_ratio = 32.0 / self.activation_bits
        
        # Approximate compressed model size
        compressed_size_mb = original_model_size_mb / weight_compression_ratio
        final_size_mb = compressed_size_mb + overhead['total_overhead_mb']
        
        return {
            'original_size_mb': original_model_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'final_size_mb': final_size_mb,
            'weight_compression_ratio': weight_compression_ratio,
            'activation_compression_ratio': activation_compression_ratio,
            'overall_compression_ratio': original_model_size_mb / final_size_mb,
            'storage_overhead': overhead,
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits
        }
