"""
Utility functions for model compression
Includes sparse storage format with 3-bit index encoding
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



def calculate_sparse_storage_overhead(num_nonzero_params, index_bits=3):
    """
    Calculate storage overhead for sparse format with index encoding

    In sparse storage, we store:
    - The non-zero parameter values
    - Index offsets (distance to next non-zero parameter)

    Args:
        num_nonzero_params: Number of non-zero parameters
        index_bits: Bits per index offset (default: 3)

    Returns:
        overhead_mb: Storage overhead in megabytes
        overhead_bits: Storage overhead in bits
    """
    # Overhead: index_bits per non-zero parameter
    overhead_bits = index_bits * num_nonzero_params
    overhead_mb = overhead_bits / (8 * 1024 * 1024)

    return overhead_mb, overhead_bits



def calculate_model_size(model, bits=32, count_nonzero_only=True, 
                        use_sparse_format=False, index_bits=3):
    """
    Calculate model size in MB based on parameter count

    Args:
        model: PyTorch model
        bits: Bit-width for weights (32 for FP32, 8 for INT8, 4 for INT4)
        count_nonzero_only: If True, only count non-zero parameters (for pruned models)
        use_sparse_format: If True, add sparse storage overhead (index encoding)
        index_bits: Bits per index in sparse format (default: 3)

    Returns:
        size_mb: Model size in megabytes
    """
    if count_nonzero_only:
        total_params, _ = count_nonzero_parameters(model)
    else:
        total_params, _ = count_parameters(model)

    # Base size: parameter values
    size_mb = (total_params * bits) / (8 * 1024 * 1024)

    # Add sparse storage overhead if requested
    if use_sparse_format and count_nonzero_only:
        overhead_mb, _ = calculate_sparse_storage_overhead(total_params, index_bits)
        size_mb += overhead_mb

    return size_mb



def calculate_quantization_metadata(model, bits_config, use_per_channel=True):
    """
    Calculate metadata overhead for quantized model with SYMMETRIC quantization

    Metadata consists of:
    - Scale factors: FP32 (4 bytes per tensor/channel)
    - Zero-points: NOT STORED (always 0 for symmetric quantization)

    Args:
        model: PyTorch model
        bits_config: Dictionary with bit-widths per layer type
        use_per_channel: If True, use per-channel quantization (more metadata)

    Returns:
        metadata_bytes: Total metadata size in bytes
        metadata_breakdown: Dictionary with per-layer metadata details
    """
    total_metadata_bytes = 0
    metadata_breakdown = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_metadata = 0

            if use_per_channel:
                # Per-channel quantization: one scale per output channel
                if isinstance(module, nn.Conv2d):
                    num_channels = module.out_channels
                else:  # Linear
                    num_channels = module.out_features

                # Scale (FP32) per channel (symmetric quantization)
                layer_metadata = num_channels * 4  # 4 bytes per channel
            else:
                # Per-tensor quantization: one scale per layer
                layer_metadata = 4  # 4 bytes total

            # Activation scale per layer (symmetric quantization)
            activation_metadata = 4  # 4 bytes per layer

            total_layer_metadata = layer_metadata + activation_metadata
            total_metadata_bytes += total_layer_metadata

            metadata_breakdown[name] = {
                'type': type(module).__name__,
                'weight_metadata_bytes': layer_metadata,
                'activation_metadata_bytes': activation_metadata,
                'total_metadata_bytes': total_layer_metadata,
                'num_channels': num_channels if use_per_channel else 1,
                'quantization_type': 'symmetric (scale only)'
            }

    return total_metadata_bytes, metadata_breakdown



def calculate_quantized_model_size(model, bits_config, count_nonzero_only=True, 
                                   use_per_channel=True, include_metadata=True,
                                   use_sparse_format=True, index_bits=3):
    """
    Calculate size for quantized model with mixed precision

    Args:
        model: PyTorch model
        bits_config: Dictionary with bit-widths per layer type
            Example: {
                'first_conv': {'weight_bits': 8, 'activation_bits': 8},
                'inverted_residual': {'weight_bits': 8, 'activation_bits': 8},
                'final_conv': {'weight_bits': 8, 'activation_bits': 8},
                'classifier': {'weight_bits': 8, 'activation_bits': 8}
            }
        count_nonzero_only: If True, only count non-zero parameters (for pruned models)
        use_per_channel: If True, use per-channel quantization metadata
        include_metadata: If True, include quantization metadata in size calculation
        use_sparse_format: If True, add sparse storage overhead for pruned models
        index_bits: Bits per index in sparse format (default: 3)

    Returns:
        total_size_mb: Total model size in megabytes
        size_breakdown: Dictionary with detailed breakdown
    """
    total_weight_bits = 0
    total_nonzero_params = 0
    layer_count = 0

    for name, param in model.named_parameters():
        # Skip non-weight parameters (biases are typically kept in higher precision)
        if 'weight' not in name:
            continue

        # Count only non-zero parameters if specified
        if count_nonzero_only:
            num_params = torch.count_nonzero(param).item()
        else:
            num_params = param.numel()

        total_nonzero_params += num_params

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

        total_weight_bits += num_params * bits
        layer_count += 1

    # Convert bits to MB
    weight_size_mb = total_weight_bits / (8 * 1024 * 1024)

    # Calculate quantization metadata overhead
    metadata_bytes = 0
    metadata_breakdown = {}

    if include_metadata:
        metadata_bytes, metadata_breakdown = calculate_quantization_metadata(
            model, bits_config, use_per_channel
        )

    metadata_size_mb = metadata_bytes / (1024 * 1024)

    # Calculate sparse storage overhead if requested
    sparse_overhead_mb = 0
    sparse_overhead_bits = 0

    if use_sparse_format and count_nonzero_only:
        sparse_overhead_mb, sparse_overhead_bits = calculate_sparse_storage_overhead(
            total_nonzero_params, index_bits
        )

    # Total size
    total_size_mb = weight_size_mb + metadata_size_mb + sparse_overhead_mb

    size_breakdown = {
        'weight_size_mb': weight_size_mb,
        'metadata_size_mb': metadata_size_mb,
        'sparse_overhead_mb': sparse_overhead_mb,
        'sparse_overhead_bits': sparse_overhead_bits,
        'total_size_mb': total_size_mb,
        'metadata_bytes': metadata_bytes,
        'metadata_percentage': (metadata_size_mb / total_size_mb * 100) if total_size_mb > 0 else 0,
        'sparse_overhead_percentage': (sparse_overhead_mb / total_size_mb * 100) if total_size_mb > 0 else 0,
        'num_quantized_layers': layer_count,
        'num_nonzero_params': total_nonzero_params,
        'use_per_channel': use_per_channel,
        'use_sparse_format': use_sparse_format,
        'index_bits': index_bits if use_sparse_format else 0,
        'metadata_breakdown': metadata_breakdown
    }

    return total_size_mb, size_breakdown



def print_quantization_summary(size_breakdown):
    """
    Print detailed quantization size breakdown

    Args:
        size_breakdown: Dictionary returned by calculate_quantized_model_size
    """
    print("\n" + "=" * 80)
    print("Quantized Model Size Breakdown")
    print("=" * 80)

    print(f"\nWeight Storage:")
    print(f"  Size (weights only):    {size_breakdown['weight_size_mb']:.4f} MB")
    print(f"  Non-zero parameters:    {size_breakdown.get('num_nonzero_params', 'N/A'):,}")

    print(f"\nQuantization Metadata:")
    print(f"  Quantization type:      {size_breakdown.get('quantization_type', 'symmetric')}")
    print(f"  Quantization scheme:    {'Per-channel' if size_breakdown['use_per_channel'] else 'Per-tensor'}")
    print(f"  Metadata size:          {size_breakdown['metadata_size_mb']:.4f} MB ({size_breakdown['metadata_bytes']:,} bytes)")
    print(f"  Metadata percentage:    {size_breakdown['metadata_percentage']:.2f}% of total")

    # Sparse format information
    if size_breakdown.get('use_sparse_format', False):
        print(f"\nSparse Storage Format:")
        print(f"  Index encoding:         {size_breakdown['index_bits']}-bit per non-zero parameter")
        print(f"  Sparse overhead:        {size_breakdown['sparse_overhead_mb']:.4f} MB ({size_breakdown['sparse_overhead_bits']:,} bits)")
        print(f"  Sparse percentage:      {size_breakdown['sparse_overhead_percentage']:.2f}% of total")

    print(f"\nTotal Model Size:")
    print(f"  Weights:                {size_breakdown['weight_size_mb']:.4f} MB")
    print(f"  Quantization metadata:  {size_breakdown['metadata_size_mb']:.4f} MB")
    if size_breakdown.get('use_sparse_format', False):
        print(f"  Sparse index overhead:  {size_breakdown['sparse_overhead_mb']:.4f} MB")
    print(f"  {'─' * 30}")
    print(f"  Total:                  {size_breakdown['total_size_mb']:.4f} MB")

    print(f"\nQuantized layers: {size_breakdown['num_quantized_layers']}")

    # Show top 5 layers with most metadata
    if size_breakdown['metadata_breakdown']:
        sorted_layers = sorted(
            size_breakdown['metadata_breakdown'].items(),
            key=lambda x: x[1]['total_metadata_bytes'],
            reverse=True
        )

        print(f"\nTop 5 Layers by Metadata Size:")
        for i, (layer_name, info) in enumerate(sorted_layers[:5]):
            print(f"  {i+1}. {layer_name}")
            print(f"     Type: {info['type']}, Channels: {info['num_channels']}, "
                  f"Metadata: {info['total_metadata_bytes']} bytes")

    print("=" * 80)



def print_model_summary(model, title="Model Summary", use_sparse_format=False, index_bits=3):
    """Print model summary with parameter counts"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    total_params, trainable_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    sparsity = calculate_sparsity(model)

    size_fp32 = calculate_model_size(model, bits=32, count_nonzero_only=False, use_sparse_format=False)
    size_fp32_sparse = calculate_model_size(model, bits=32, count_nonzero_only=True, use_sparse_format=use_sparse_format, index_bits=index_bits)
    size_int8 = calculate_model_size(model, bits=8, count_nonzero_only=True, use_sparse_format=use_sparse_format, index_bits=index_bits)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Non-zero parameters: {nonzero_params:,}")

    if sparsity > 0.1:
        print(f"Sparsity: {sparsity:.2f}%")

    print(f"\nModel size (parameter-based):")
    print(f"  FP32 (dense):  {size_fp32:.2f} MB")

    if sparsity > 0.1:
        sparse_label = f" (sparse{' + index' if use_sparse_format else ''})"
        print(f"  FP32{sparse_label}: {size_fp32_sparse:.2f} MB")
        print(f"  INT8{sparse_label}: {size_int8:.2f} MB")

        if use_sparse_format:
            sparse_overhead_mb, _ = calculate_sparse_storage_overhead(nonzero_params, index_bits)
            print(f"\n  Sparse index overhead ({index_bits}-bit): {sparse_overhead_mb:.4f} MB")
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
