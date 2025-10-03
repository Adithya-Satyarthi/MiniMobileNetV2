"""
Utility functions for model compression
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_nonzero_parameters(model):
    """Count only non-zero parameters (useful for pruned models)"""
    total_nonzero = 0
    trainable_nonzero = 0
    
    for param in model.parameters():
        nonzero_count = torch.count_nonzero(param).item()
        total_nonzero += nonzero_count
        
        if param.requires_grad:
            trainable_nonzero += nonzero_count
    
    return total_nonzero, trainable_nonzero


def calculate_sparsity(model):
    """Calculate sparsity ratio (percentage of zero weights)"""
    total_params, _ = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    
    if total_params == 0:
        return 0.0
    
    return 100.0 * (1.0 - nonzero_params / total_params)


def calculate_model_size(model, bits=32, count_nonzero_only=True):
    """
    Calculate model size in MB based on parameter count
    
    Args:
        model: PyTorch model
        bits: Bit-width for weights (32 for FP32, 8 for INT8, 4 for INT4)
        count_nonzero_only: If True, only count non-zero parameters (for pruned models)
    
    Returns:
        size_mb: Model size in megabytes
    """
    if count_nonzero_only:
        total_params, _ = count_nonzero_parameters(model)
    else:
        total_params, _ = count_parameters(model)
    
    size_mb = (total_params * bits) / (8 * 1024 * 1024)
    return size_mb


def calculate_quantized_model_size(model, bits_config, count_nonzero_only=True):
    """
    Calculate size for quantized model with mixed precision
    
    Args:
        model: PyTorch model
        bits_config: Dictionary with bit-widths per layer type
        count_nonzero_only: If True, only count non-zero parameters (for pruned models)
    
    Returns:
        size_mb: Estimated model size in megabytes
    """
    total_bits = 0
    
    for name, param in model.named_parameters():
        # Count only non-zero parameters if specified
        if count_nonzero_only:
            num_params = torch.count_nonzero(param).item()
        else:
            num_params = param.numel()
        
        # Determine bit-width based on layer name
        if 'features.0' in name:
            bits = bits_config['first_conv']['weight_bits']
        elif 'features.18' in name:
            bits = bits_config['final_conv']['weight_bits']
        elif 'classifier' in name:
            bits = bits_config['classifier']['weight_bits']
        elif 'features' in name:
            bits = bits_config['inverted_residual']['weight_bits']
        else:
            bits = 32  # Default FP32
        
        total_bits += num_params * bits
    
    return total_bits / (8 * 1024 * 1024)



def print_model_summary(model, title="Model Summary"):
    """Print model summary with parameter counts"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    sparsity = calculate_sparsity(model)
    
    size_fp32 = calculate_model_size(model, bits=32, count_nonzero_only=False)
    size_fp32_sparse = calculate_model_size(model, bits=32, count_nonzero_only=True)
    size_int8 = calculate_model_size(model, bits=8, count_nonzero_only=True)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Non-zero parameters: {nonzero_params:,}")
    
    if sparsity > 0.1:
        print(f"Sparsity: {sparsity:.2f}%")
    
    print(f"\nModel size (parameter-based):")
    print(f"  FP32 (dense):  {size_fp32:.2f} MB")
    
    if sparsity > 0.1:
        print(f"  FP32 (sparse): {size_fp32_sparse:.2f} MB")
        print(f"  INT8 (sparse): {size_int8:.2f} MB")
    else:
        print(f"  INT8:          {calculate_model_size(model, bits=8):.2f} MB")
    
    print("=" * 80)


def get_layer_info(model):
    """Get information about model layers"""
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_info.append({
                'name': name,
                'type': 'Conv2d',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'groups': module.groups
            })
    
    return layer_info




def save_model_checkpoint(model, optimizer, epoch, val_acc, save_path, config=None, strip_masks=False):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        epoch: Current epoch
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
        config: Configuration dict (optional)
        strip_masks: If True, removes pruning_mask buffers before saving
    """
    state_dict = model.state_dict()
    
    # Strip pruning masks if requested
    if strip_masks:
        state_dict = {k: v for k, v in state_dict.items() if 'pruning_mask' not in k}
        print("Stripped pruning masks from checkpoint")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'val_acc': val_acc,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model_checkpoint(checkpoint_path, model, strict=True):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        strict: If False, allows loading with missing/unexpected keys
    
    Returns:
        model: Model with loaded weights
        epoch: Training epoch
        val_acc: Validation accuracy
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Filter out pruning masks if loading into unpruned model
        state_dict = checkpoint['model_state_dict']
        
        # Check if state_dict has pruning masks
        has_masks = any('pruning_mask' in key for key in state_dict.keys())
        
        if has_masks and strict:
            print("Warning: Checkpoint contains pruning masks. Loading with strict=False")
            strict = False
        
        model.load_state_dict(state_dict, strict=strict)
        epoch = checkpoint.get('epoch', 0)
        val_acc = checkpoint.get('val_acc', 0)
    else:
        model.load_state_dict(checkpoint, strict=strict)
        epoch = 0
        val_acc = 0
    
    return model, epoch, val_acc
