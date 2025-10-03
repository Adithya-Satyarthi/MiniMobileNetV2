"""
Compression Analysis Script
Calculates compression ratio as: baseline_total_params / (compressed_nonzero_params + metadata + overhead)
"""

import os
import argparse
import torch
import torch.nn as nn
import yaml
from src.model import MobileNetV2_CIFAR10
from src.compression.utils import (
    load_model_checkpoint,
    count_parameters,
    count_nonzero_parameters,
    calculate_sparsity,
    calculate_quantization_metadata,
    calculate_sparse_storage_overhead,
)
from src.data_loader import get_cifar10_dataloaders


class ActivationProfiler:
    """Profile activation memory usage during forward pass."""
    def __init__(self, measure_sparsity=True):
        self.activation_sizes = []
        self.activation_nonzero_counts = []
        self.activation_shapes = []
        self.hooks = []
        self.measure_sparsity = measure_sparsity

    def register_hooks(self, model):
        """Register forward hooks on all modules"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                total_elements = output.numel()
                self.activation_sizes.append(total_elements)
                self.activation_shapes.append(tuple(output.shape))

                if self.measure_sparsity:
                    nonzero_elements = torch.count_nonzero(output).item()
                    self.activation_nonzero_counts.append(nonzero_elements)
                else:
                    self.activation_nonzero_counts.append(total_elements)

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_total_activations(self):
        return sum(self.activation_sizes)

    def get_nonzero_activations(self):
        return sum(self.activation_nonzero_counts)

    def get_activation_sparsity(self):
        total = self.get_total_activations()
        nonzero = self.get_nonzero_activations()
        if total == 0:
            return 0.0
        return 100.0 * (1.0 - nonzero / total)

    def reset(self):
        self.activation_sizes = []
        self.activation_nonzero_counts = []
        self.activation_shapes = []


def measure_activation_memory(model, batch_size=1, input_size=(3, 32, 32), 
                              device='cpu', use_real_data=False, data_loader=None,
                              measure_sparsity=True):
    """Measure activation memory during forward pass."""
    model = model.to(device)
    model.eval()

    profiler = ActivationProfiler(measure_sparsity=measure_sparsity)
    profiler.register_hooks(model)

    if use_real_data and data_loader is not None:
        data_iter = iter(data_loader)
        images, _ = next(data_iter)
        images = images[:batch_size].to(device)
    else:
        images = torch.randn(batch_size, *input_size).to(device)

    with torch.no_grad():
        _ = model(images)

    total_activations = profiler.get_total_activations()
    nonzero_activations = profiler.get_nonzero_activations()
    activation_sparsity = profiler.get_activation_sparsity()

    activation_memory_fp32_dense = (total_activations * 4) / (1024 * 1024)
    activation_memory_sparse = (nonzero_activations * 4) / (1024 * 1024)

    profiler.remove_hooks()

    return (total_activations, nonzero_activations, activation_sparsity,
            activation_memory_fp32_dense, activation_memory_sparse)


def get_quantization_config_from_model_path(model_path):
    """Infer quantization config from model path."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'quantization_config.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['quantization']['bits']

    return {
        'first_conv': {'weight_bits': 8, 'activation_bits': 8},
        'inverted_residual': {'weight_bits': 8, 'activation_bits': 8},
        'final_conv': {'weight_bits': 8, 'activation_bits': 8},
        'classifier': {'weight_bits': 8, 'activation_bits': 8}
    }


def analyze_compression(baseline_path, compressed_path, output_dir=None, 
                       use_sparse_format=True, index_bits=3,
                       measure_activation_sparsity=True):
    """
    Compression analysis:
    - Baseline: Total parameters from baseline model
    - Compressed: Non-zero parameters from quantized model + metadata + overhead
    """
    print("-" * 80)
    print("COMPRESSION ANALYSIS")
    print("-" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_cifar10_dataloaders(batch_size=128, num_workers=4)

    # Load models
    print("\nLoading models...")
    baseline_model = MobileNetV2_CIFAR10()
    baseline_model, _, _ = load_model_checkpoint(baseline_path, baseline_model, strict=False)
    baseline_model = baseline_model.to(device)

    compressed_model = MobileNetV2_CIFAR10()
    compressed_model, _, _ = load_model_checkpoint(compressed_path, compressed_model, strict=False)
    compressed_model = compressed_model.to(device)

    # Get quantization config
    bits_config = get_quantization_config_from_model_path(compressed_path)
    weight_bits = bits_config['inverted_residual']['weight_bits']
    activation_bits = bits_config['inverted_residual']['activation_bits']

    print(f"Baseline model: {baseline_path}")
    print(f"Compressed model: {compressed_path}")
    print(f"Quantization: W{weight_bits}A{activation_bits}")

    # =========================================================================
    # Calculate baseline and compressed sizes
    # =========================================================================

    # Baseline: ALL parameters from baseline model
    baseline_total_params, _ = count_parameters(baseline_model)
    baseline_size_mb = (baseline_total_params * 32) / (8 * 1024 * 1024)

    # Compressed: NON-ZERO parameters from quantized model
    compressed_total_params, _ = count_parameters(compressed_model)
    compressed_nonzero_params, _ = count_nonzero_parameters(compressed_model)
    weight_sparsity = calculate_sparsity(compressed_model)

    # Pure weight data (only non-zero weights)
    compressed_weights_only_mb = (compressed_nonzero_params * weight_bits) / (8 * 1024 * 1024)

    # Quantization metadata
    metadata_bytes, _ = calculate_quantization_metadata(
        compressed_model, bits_config, use_per_channel=True
    )
    metadata_mb = metadata_bytes / (1024 * 1024)

    # Sparse storage overhead
    sparse_overhead_mb = 0
    if use_sparse_format:
        sparse_overhead_mb, _ = calculate_sparse_storage_overhead(
            compressed_nonzero_params, index_bits
        )

    # Total compressed size = weights + metadata + sparse overhead
    total_compressed_mb = compressed_weights_only_mb + metadata_mb + sparse_overhead_mb

    # =========================================================================
    # (a) OVERALL MODEL COMPRESSION RATIO
    # =========================================================================
    print("\n" + "-" * 80)
    print("(a) OVERALL MODEL COMPRESSION RATIO")
    print("-" * 80)
    print("\nBaseline: Total parameters from baseline model")
    print("Compressed: Non-zero parameters + metadata + sparse overhead")

    overall_compression_ratio = baseline_size_mb / total_compressed_mb

    print(f"\nBaseline Model (FP32):")
    print(f"  Total parameters:     {baseline_total_params:,}")
    print(f"  Model size:           {baseline_size_mb:.4f} MB")
    print(f"  Bits per param:       32")

    print(f"\nCompressed Model (W{weight_bits}A{activation_bits}, {weight_sparsity:.1f}% sparse):")
    print(f"  Total parameters:     {compressed_total_params:,}")
    print(f"  Non-zero parameters:  {compressed_nonzero_params:,}")
    print(f"  Quantized weights:    {compressed_weights_only_mb:.4f} MB")
    print(f"  Quantization metadata:{metadata_mb:.4f} MB")
    print(f"  Sparse overhead:      {sparse_overhead_mb:.4f} MB")
    print(f"  Total model size:     {total_compressed_mb:.4f} MB")

    print(f"\n{'─' * 80}")
    print(f"Overall Compression Ratio: {overall_compression_ratio:.2f}×")
    print(f"  From {baseline_size_mb:.4f} MB to {total_compressed_mb:.4f} MB")
    print(f"{'─' * 80}")

    # =========================================================================
    # (b) WEIGHT COMPRESSION RATIO (pure weights only)
    # =========================================================================
    print("\n" + "-" * 80)
    print("(b) WEIGHT COMPRESSION RATIO")
    print("-" * 80)
    print("\nPure weight data only (no metadata, no sparse overhead)")

    weight_only_compression_ratio = baseline_size_mb / compressed_weights_only_mb

    print(f"\nBaseline Weights (FP32):")
    print(f"  All parameters:       {baseline_total_params:,}")
    print(f"  Size:                 {baseline_size_mb:.4f} MB")
    print(f"  Bits per param:       32")

    print(f"\nCompressed Weights (INT{weight_bits}, sparse):")
    print(f"  Non-zero parameters:  {compressed_nonzero_params:,}")
    print(f"  Pure weight data:     {compressed_weights_only_mb:.4f} MB")
    print(f"  Bits per param:       {weight_bits}")
    print(f"  Effective reduction:  {baseline_total_params}/{compressed_nonzero_params} params × {32}/{weight_bits} bits")

    print(f"\n{'─' * 80}")
    print(f"Weight Compression Ratio: {weight_only_compression_ratio:.2f}×")
    print(f"  From {baseline_size_mb:.4f} MB to {compressed_weights_only_mb:.4f} MB")
    print(f"  (Theoretical: {(baseline_total_params/compressed_nonzero_params) * (32/weight_bits):.2f}× from sparsity + quantization)")
    print(f"{'─' * 80}")

    # =========================================================================
    # (c) ACTIVATION COMPRESSION RATIO
    # =========================================================================
    print("\n" + "-" * 80)
    print("(c) ACTIVATION COMPRESSION RATIO (Runtime Analysis)")
    print("-" * 80)
    print("\nRuntime memory, not stored in model file")

    # Baseline activations
    (baseline_total_acts, baseline_nonzero_acts, baseline_act_sparsity,
     baseline_activation_mb, _) = measure_activation_memory(
        baseline_model, batch_size=1, device=device, 
        use_real_data=True, data_loader=test_loader,
        measure_sparsity=measure_activation_sparsity
    )

    # Compressed activations
    (compressed_total_acts, compressed_nonzero_acts, compressed_act_sparsity,
     _, _) = measure_activation_memory(
        compressed_model, batch_size=1, device=device,
        use_real_data=True, data_loader=test_loader,
        measure_sparsity=measure_activation_sparsity
    )

    # Quantized activation size
    compressed_activation_mb_dense = (compressed_total_acts * activation_bits) / (8 * 1024 * 1024)
    compressed_activation_mb_sparse = (compressed_nonzero_acts * activation_bits) / (8 * 1024 * 1024)

    if measure_activation_sparsity and compressed_act_sparsity > 0.1:
        act_sparse_overhead, _ = calculate_sparse_storage_overhead(compressed_nonzero_acts, index_bits)
        compressed_activation_mb_sparse += act_sparse_overhead
        compressed_activation_final = compressed_activation_mb_sparse
    else:
        compressed_activation_final = compressed_activation_mb_dense
        compressed_act_sparsity = 0.0

    activation_compression_ratio = baseline_activation_mb / compressed_activation_final

    print(f"\nMeasurement: Forward pass profiling with hooks on real data")

    print(f"\nBaseline Activations (FP32):")
    print(f"  Total elements:       {baseline_total_acts:,}")
    print(f"  Non-zero elements:    {baseline_nonzero_acts:,}")
    if measure_activation_sparsity and baseline_act_sparsity > 0.1:
        print(f"  Natural sparsity:     {baseline_act_sparsity:.2f}% (from ReLU)")
    print(f"  Memory (per sample):  {baseline_activation_mb:.4f} MB")

    print(f"\nCompressed Activations (INT{activation_bits}):")
    print(f"  Total elements:       {compressed_total_acts:,}")
    print(f"  Non-zero elements:    {compressed_nonzero_acts:,}")
    if measure_activation_sparsity and compressed_act_sparsity > 0.1:
        print(f"  Sparsity:             {compressed_act_sparsity:.2f}%")
        print(f"  Memory dense:         {compressed_activation_mb_dense:.4f} MB")
        print(f"  Memory sparse:        {compressed_activation_mb_sparse:.4f} MB")
    print(f"  Memory (per sample):  {compressed_activation_final:.4f} MB")

    print(f"\n{'─' * 80}")
    print(f"Activation Compression Ratio: {activation_compression_ratio:.2f}×")
    print(f"  From {baseline_activation_mb:.4f} MB to {compressed_activation_final:.4f} MB (per sample)")
    if measure_activation_sparsity and compressed_act_sparsity > 0.1:
        print(f"  Factors: {32/activation_bits:.2f}× (quantization) × sparsity benefit")
    print(f"\nFor batch size 128:")
    print(f"  Baseline: {baseline_activation_mb * 128:.2f} MB")
    print(f"  Compressed: {compressed_activation_final * 128:.2f} MB")
    print(f"{'─' * 80}")

    # =========================================================================
    # (d) FINAL MODEL SIZE
    # =========================================================================
    print("\n" + "-" * 80)
    print("(d) FINAL MODEL SIZE (Detailed Breakdown)")
    print("-" * 80)
    print("\nStorage size = Weights + Metadata + Sparse Overhead")

    print(f"\nModel Storage Components:")
    print(f"  1. Quantized weights:      {compressed_weights_only_mb:.4f} MB ({compressed_weights_only_mb/total_compressed_mb*100:.1f}%)")
    print(f"  2. Quantization metadata:  {metadata_mb:.4f} MB ({metadata_mb/total_compressed_mb*100:.1f}%)")
    print(f"  3. Sparse index overhead:  {sparse_overhead_mb:.4f} MB ({sparse_overhead_mb/total_compressed_mb*100:.1f}%)")
    print(f"  {'─' * 60}")
    print(f"  TOTAL MODEL SIZE:          {total_compressed_mb:.4f} MB")

    print(f"\nRuntime Memory (additional, not stored in model):")
    print(f"  Activations (per sample):  {compressed_activation_final:.4f} MB")
    print(f"  Activations (batch=128):   {compressed_activation_final * 128:.2f} MB")

    print(f"\nTotal Runtime Memory (model + activations, batch=128):")
    print(f"  {total_compressed_mb:.4f} MB (model) + {compressed_activation_final * 128:.2f} MB (activations)")
    print(f"  = {total_compressed_mb + compressed_activation_final * 128:.2f} MB")

    print(f"\n{'─' * 80}")
    print(f"Final Model Size: {total_compressed_mb:.4f} MB")
    print(f"Compression Ratio: {overall_compression_ratio:.2f}×")
    print(f"{'─' * 80}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "-" * 80)
    print("SUMMARY TABLE")
    print("-" * 80)

    print(f"\n{'Metric':<40} {'Baseline':<15} {'Compressed':<15} {'Ratio':<10}")
    print("─" * 85)
    print(f"{'(a) Model Size (weights+meta+sparse)':<40} {baseline_size_mb:>10.4f} MB {total_compressed_mb:>10.4f} MB {overall_compression_ratio:>8.2f}×")
    print(f"{'(b) Pure Weight Data':<40} {baseline_size_mb:>10.4f} MB {compressed_weights_only_mb:>10.4f} MB {weight_only_compression_ratio:>8.2f}×")
    print(f"{'(c) Activations (per sample)':<40} {baseline_activation_mb:>10.4f} MB {compressed_activation_final:>10.4f} MB {activation_compression_ratio:>8.2f}×")
    print(f"{'(d) Total Runtime (model+acts, batch=128)':<40} {baseline_size_mb + baseline_activation_mb*128:>10.2f} MB {total_compressed_mb + compressed_activation_final*128:>10.2f} MB {(baseline_size_mb + baseline_activation_mb*128)/(total_compressed_mb + compressed_activation_final*128):>8.2f}×")

    # =========================================================================
    # Save Report
    # =========================================================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'compression_analysis.txt')

        with open(report_path, 'w') as f:
            f.write("-" * 80 + "\n")
            f.write("COMPRESSION ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Baseline Model: {baseline_path}\n")
            f.write(f"Compressed Model: {compressed_path}\n")
            f.write(f"Quantization: W{weight_bits}A{activation_bits}\n")
            f.write(f"Weight Sparsity: {weight_sparsity:.2f}%\n")
            if measure_activation_sparsity and compressed_act_sparsity > 0.1:
                f.write(f"Activation Sparsity: {compressed_act_sparsity:.2f}%\n")
            f.write("\n")

            f.write("Compression Method:\n")
            f.write("  Baseline: Total parameters from baseline model\n")
            f.write("  Compressed: Non-zero parameters + metadata + sparse overhead\n\n")

            f.write("(a) Overall Model Compression Ratio\n")
            f.write("─" * 80 + "\n")
            f.write(f"Baseline: {baseline_size_mb:.4f} MB ({baseline_total_params:,} total params)\n")
            f.write(f"Compressed: {total_compressed_mb:.4f} MB ({compressed_nonzero_params:,} nonzero params)\n")
            f.write(f"Ratio: {overall_compression_ratio:.2f}×\n\n")

            f.write("(b) Pure Weight Compression Ratio\n")
            f.write("─" * 80 + "\n")
            f.write(f"Baseline: {baseline_size_mb:.4f} MB ({baseline_total_params:,} params × 32 bits)\n")
            f.write(f"Compressed: {compressed_weights_only_mb:.4f} MB ({compressed_nonzero_params:,} params × {weight_bits} bits)\n")
            f.write(f"Ratio: {weight_only_compression_ratio:.2f}×\n\n")

            f.write("(c) Activation Compression Ratio (Runtime Analysis)\n")
            f.write("─" * 80 + "\n")
            f.write("Runtime memory, not stored in model file\n")
            f.write(f"Baseline: {baseline_activation_mb:.4f} MB (FP32, per sample)\n")
            f.write(f"Compressed: {compressed_activation_final:.4f} MB (INT{activation_bits}")
            if measure_activation_sparsity and compressed_act_sparsity > 0.1:
                f.write(f", {compressed_act_sparsity:.1f}% sparse")
            f.write(", per sample)\n")
            f.write(f"Ratio: {activation_compression_ratio:.2f}×\n\n")

            f.write("(d) Final Model Size\n")
            f.write("─" * 80 + "\n")
            f.write(f"Total Model Storage: {total_compressed_mb:.4f} MB\n")
            f.write(f"  - Quantized weights: {compressed_weights_only_mb:.4f} MB\n")
            f.write(f"  - Quantization metadata: {metadata_mb:.4f} MB\n")
            f.write(f"  - Sparse index overhead: {sparse_overhead_mb:.4f} MB\n")
            f.write(f"\nRuntime Memory (additional):\n")
            f.write(f"  - Activations (per sample): {compressed_activation_final:.4f} MB\n")
            f.write(f"  - Activations (batch=128): {compressed_activation_final * 128:.2f} MB\n")

        print(f"\nReport saved to: {report_path}")

    return {
        'overall_compression_ratio': overall_compression_ratio,
        'weight_only_compression_ratio': weight_only_compression_ratio,
        'activation_compression_ratio': activation_compression_ratio,
        'model_size_mb': total_compressed_mb,
        'weight_only_mb': compressed_weights_only_mb,
        'metadata_mb': metadata_mb,
        'sparse_overhead_mb': sparse_overhead_mb,
        'activation_mb_per_sample': compressed_activation_final,
        'baseline_size_mb': baseline_size_mb,
        'baseline_total_params': baseline_total_params,
        'compressed_nonzero_params': compressed_nonzero_params,
        'weight_sparsity': weight_sparsity,
        'activation_sparsity': compressed_act_sparsity,
        'weight_bits': weight_bits,
        'activation_bits': activation_bits
    }


def main():
    parser = argparse.ArgumentParser(description='Compression Analysis')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline FP32 model')
    parser.add_argument('--compressed', type=str, required=True,
                       help='Path to compressed (pruned + quantized) model')
    parser.add_argument('--output-dir', type=str, default='results/compression_analysis',
                       help='Output directory for analysis report')
    parser.add_argument('--no-sparse-format', action='store_true',
                       help='Exclude sparse storage overhead from calculations')
    parser.add_argument('--index-bits', type=int, default=3,
                       help='Bits per index for sparse storage (default: 3)')
    parser.add_argument('--no-activation-sparsity', action='store_true',
                       help='Do not measure activation sparsity')

    args = parser.parse_args()

    results = analyze_compression(
        baseline_path=args.baseline,
        compressed_path=args.compressed,
        output_dir=args.output_dir,
        use_sparse_format=not args.no_sparse_format,
        index_bits=args.index_bits,
        measure_activation_sparsity=not args.no_activation_sparsity
    )

    print("\n" + "-" * 80)
    print("Analysis complete!")
    print("-" * 80)


if __name__ == '__main__':
    main()
