"""
Sweep different compression configurations (sparsity + quantization)
and log results to Wandb for comparison using Parallel Coordinates chart.

This script has been optimized to:
1. Prune only ONCE per sparsity level.
2. Reuse existing pruned models if found, skipping the pruning step.
3. Correctly calculate quantization metadata and sparse storage overhead.
"""


import os
import yaml
import subprocess
import shutil
import wandb
from datetime import datetime
import torch
from src.model import MobileNetV2_CIFAR10
from src.compression.utils import (
    load_model_checkpoint,
    count_parameters,
    count_nonzero_parameters,
    calculate_quantization_metadata,
    calculate_sparse_storage_overhead,
)


# Sweep configuration
SPARSITY_VALUES = [0.50, 0.70, 0.90]
BIT_WIDTHS = [6, 8]  # For both weights and activations


BASELINE_MODEL = "results/baseline/best_model.pth"
PRUNE_CONFIG_TEMPLATE = "configs/pruning.yaml"
QUANT_CONFIG_TEMPLATE = "configs/quantization.yaml"


# Output directory for sweep results
SWEEP_DIR = "results/compression_sweep"
os.makedirs(SWEEP_DIR, exist_ok=True)



def update_pruning_config(template_path, sparsity, output_dir):
    """Update pruning config with target sparsity and output path."""
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)


    config['pruning']['target_sparsity'] = sparsity
    config['paths']['output'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_config_path = os.path.join(output_dir, "pruning_config.yaml")
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f)


    return config, output_config_path



def update_quantization_config(template_path, weight_bits, act_bits, input_model, output_dir):
    """Update quantization config with bit widths and paths."""
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)


    # Set uniform bit widths across all layer types
    for layer_type in config['quantization']['bits']:
        config['quantization']['bits'][layer_type]['weight_bits'] = weight_bits
        config['quantization']['bits'][layer_type]['activation_bits'] = act_bits


    config['paths']['input_model'] = input_model
    config['paths']['output'] = output_dir
    os.makedirs(output_dir, exist_ok=True)


    output_config_path = os.path.join(output_dir, "quantization_config.yaml")
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f)


    return config, output_config_path



def run_command(cmd):
    """Run a command with subprocess and check for errors."""
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False
    return True



def evaluate_model_accuracy(model_path):
    """Evaluate model accuracy on the test set."""
    from src.data_loader import get_cifar10_dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    try:
        model = MobileNetV2_CIFAR10()
        model, _, _ = load_model_checkpoint(model_path, model, strict=False)
        model.to(device)
        model.eval()


        _, _, test_loader = get_cifar10_dataloaders(batch_size=128, num_workers=4)
        correct = total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    except Exception as e:
        print(f"Error evaluating model at {model_path}: {e}")
        return None



def get_model_stats(pruned_model_path, weight_bits, act_bits, use_per_channel=True, 
                   use_sparse_format=True, index_bits=3):
    """
    Calculate complete model statistics with proper metadata calculation.

    Includes:
    - Original FP32 size
    - Quantized weights size
    - Quantization metadata (scale factors only, symmetric quantization)
    - Sparse storage overhead (3-bit index encoding)
    """
    try:
        model = MobileNetV2_CIFAR10()
        model, _, _ = load_model_checkpoint(pruned_model_path, model, strict=False)


        # Count parameters
        total_params, _ = count_parameters(model)
        nonzero_params, _ = count_nonzero_parameters(model)
        sparsity = 100.0 * (1 - nonzero_params / total_params)


        # Original size (FP32, dense)
        original_size_bits = total_params * 32
        original_size_mb = original_size_bits / (8 * 1024 * 1024)


        # Quantized weights size (only non-zero parameters)
        quantized_weights_bits = nonzero_params * weight_bits
        quantized_weights_mb = quantized_weights_bits / (8 * 1024 * 1024)


        # Build bits_config for metadata calculation
        bits_config = {
            'first_conv': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'inverted_residual': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'final_conv': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'classifier': {'weight_bits': weight_bits, 'activation_bits': act_bits}
        }


        # Calculate quantization metadata (symmetric: scale factors only, no zero-points)
        metadata_bytes, metadata_breakdown = calculate_quantization_metadata(
            model, bits_config, use_per_channel=use_per_channel
        )
        metadata_mb = metadata_bytes / (1024 * 1024)


        # Calculate sparse storage overhead (3-bit indices per non-zero parameter)
        sparse_overhead_mb = 0
        sparse_overhead_bits = 0
        if use_sparse_format:
            sparse_overhead_mb, sparse_overhead_bits = calculate_sparse_storage_overhead(
                nonzero_params, index_bits=index_bits
            )


        # Total quantized size = weights + quantization metadata + sparse overhead
        quantized_size_mb = quantized_weights_mb + metadata_mb + sparse_overhead_mb


        return {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity,
            'original_size_mb': original_size_mb,
            'quantized_weights_mb': quantized_weights_mb,
            'metadata_mb': metadata_mb,
            'metadata_bytes': metadata_bytes,
            'sparse_overhead_mb': sparse_overhead_mb,
            'sparse_overhead_bits': sparse_overhead_bits,
            'quantized_size_mb': quantized_size_mb,
            'compression_ratio': original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0,
            'use_per_channel': use_per_channel,
            'use_sparse_format': use_sparse_format,
            'index_bits': index_bits
        }
    except Exception as e:
        print(f"Error getting model stats: {e}")
        import traceback
        traceback.print_exc()
        return None



def main():
    wandb.init(
        project="mobilenetv2-compression-sweep",
        name=f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


    print("-" * 80)
    print("Evaluating Baseline Model")
    baseline_accuracy = evaluate_model_accuracy(BASELINE_MODEL)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%\n")


    all_results = []


    for sparsity in SPARSITY_VALUES:
        print("\n" + "-" * 80)
        print(f"# Processing Sparsity Level: {sparsity*100:.0f}%")
        print("-" * 80)


        prune_dir = os.path.join(SWEEP_DIR, f"pruned_s{int(sparsity*100)}")
        pruned_model_path = os.path.join(prune_dir, 'pruned_model_final.pth')


        # --- 1. Pruning (or reuse existing model) ---
        if os.path.exists(pruned_model_path):
            print(f" Found existing pruned model at: {pruned_model_path}")
            print("   Reusing it and skipping the pruning step.")
        else:
            print(f"Could not find pruned model for {sparsity*100:.0f}% sparsity. Running pruning...")
            _, prune_config_path = update_pruning_config(PRUNE_CONFIG_TEMPLATE, sparsity, prune_dir)
            if not run_command(["python", "main.py", "--mode", "prune", "--prune-config", prune_config_path]):
                print(f"Pruning failed for sparsity {sparsity}. Skipping this level.")
                continue


        if not os.path.exists(pruned_model_path):
            print(f"Pruned model still not found after running. Critical error. Skipping.")
            continue

        pruned_accuracy = evaluate_model_accuracy(pruned_model_path)
        print(f"Pruned Model Accuracy at {sparsity*100:.0f}% sparsity: {pruned_accuracy:.2f}%")


        # --- 2. Quantization ---
        for weight_bits in BIT_WIDTHS:
            for act_bits in BIT_WIDTHS:
                config_name = f"s{int(sparsity*100)}_w{weight_bits}_a{act_bits}"
                print("\n" + "-" * 80)
                print(f"Running Quantization: {config_name}")
                print("-" * 80)


                run_dir = os.path.join(SWEEP_DIR, config_name)

                quant_output_dir = os.path.join(run_dir, 'quant_output')
                _, quant_config_path = update_quantization_config(
                    QUANT_CONFIG_TEMPLATE, weight_bits, act_bits, pruned_model_path, quant_output_dir
                )


                if not run_command(["python", "main.py", "--mode", "quantize", "--quant-config", quant_config_path]):
                    print(f"Quantization failed for {config_name}. Skipping.")
                    continue


                quant_source = os.path.join(quant_output_dir, 'ptq', 'quantized_model.pth')
                quant_dest = os.path.join(run_dir, 'quantized_model.pth')


                if os.path.exists(quant_source):
                    shutil.copy(quant_source, quant_dest)
                else:
                    print(f"Quantized model not found at {quant_source}. Skipping.")
                    continue

                # --- 3. Evaluation with correct metadata calculation ---
                quant_accuracy = evaluate_model_accuracy(quant_dest)

                # Calculate stats with proper metadata
                stats = get_model_stats(
                    pruned_model_path, 
                    weight_bits, 
                    act_bits,
                    use_per_channel=True,      # Use per-channel quantization metadata
                    use_sparse_format=True,    # Include sparse storage overhead
                    index_bits=3               # 3-bit index encoding
                )


                if stats and quant_accuracy is not None:
                    result = {
                        'config_name': config_name,
                        'sparsity': sparsity * 100,
                        'weight_bits': weight_bits,
                        'activation_bits': act_bits,
                        'baseline_accuracy': baseline_accuracy,
                        'pruned_accuracy': pruned_accuracy,
                        'final_accuracy': quant_accuracy,
                        'accuracy_drop': baseline_accuracy - quant_accuracy,
                        'total_params': stats['total_params'],
                        'nonzero_params': stats['nonzero_params'],
                        'original_size_mb': stats['original_size_mb'],
                        'quantized_weights_mb': stats['quantized_weights_mb'],
                        'metadata_mb': stats['metadata_mb'],
                        'sparse_overhead_mb': stats['sparse_overhead_mb'],
                        'final_size_mb': stats['quantized_size_mb'],
                        'compression_ratio': stats['compression_ratio']
                    }
                    all_results.append(result)
                    wandb.log(result)


                    print(f"\nResults for {config_name}:")
                    print(f"  Accuracy: {baseline_accuracy:.2f}% → {quant_accuracy:.2f}% (Drop: {result['accuracy_drop']:.2f}%)")
                    print(f"  Size Breakdown:")
                    print(f"    Original (FP32):        {stats['original_size_mb']:.4f} MB")
                    print(f"    Quantized weights:      {stats['quantized_weights_mb']:.4f} MB")
                    print(f"    Quantization metadata:  {stats['metadata_mb']:.4f} MB ({stats['metadata_bytes']:,} bytes)")
                    print(f"    Sparse index overhead:  {stats['sparse_overhead_mb']:.4f} MB ({stats['sparse_overhead_bits']:,} bits)")
                    print(f"    Total compressed:       {stats['quantized_size_mb']:.4f} MB")
                    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}×")


    # --- 4. Summary ---
    if all_results:
        print("\n" + "-" * 80)
        print("SWEEP SUMMARY")
        print("-" * 80)

        best_accuracy = max(all_results, key=lambda x: x['final_accuracy'])
        best_compression = max(all_results, key=lambda x: x['compression_ratio'])

        # Balanced score: 60% accuracy weight, 40% compression weight
        for r in all_results:
            r['score'] = (r['final_accuracy'] / baseline_accuracy) * 0.6 + (r['compression_ratio'] / 50) * 0.4
        best_balanced = max(all_results, key=lambda x: x['score'])


        print("\n Best Accuracy Config:")
        print(f"  Config: {best_accuracy['config_name']}")
        print(f"  Accuracy: {best_accuracy['final_accuracy']:.2f}% (drop: {best_accuracy['accuracy_drop']:.2f}%)")
        print(f"  Compression: {best_accuracy['compression_ratio']:.2f}×")
        print(f"  Size: {best_accuracy['original_size_mb']:.2f} MB → {best_accuracy['final_size_mb']:.4f} MB")

        print("\n Best Compression Config:")
        print(f"  Config: {best_compression['config_name']}")
        print(f"  Compression: {best_compression['compression_ratio']:.2f}×")
        print(f"  Accuracy: {best_compression['final_accuracy']:.2f}% (drop: {best_compression['accuracy_drop']:.2f}%)")
        print(f"  Size: {best_compression['original_size_mb']:.2f} MB → {best_compression['final_size_mb']:.4f} MB")

        print("\n  Best Balanced Config:")
        print(f"  Config: {best_balanced['config_name']}")
        print(f"  Accuracy: {best_balanced['final_accuracy']:.2f}% (drop: {best_balanced['accuracy_drop']:.2f}%)")
        print(f"  Compression: {best_balanced['compression_ratio']:.2f}×")
        print(f"  Size: {best_balanced['original_size_mb']:.2f} MB → {best_balanced['final_size_mb']:.4f} MB")
        print(f"  Balance score: {best_balanced['score']:.4f}")


        # Detailed breakdown for best balanced config
        print("\n" + "-" * 80)
        print(f"Best Balanced Config Detailed Breakdown ({best_balanced['config_name']}):")
        print("-" * 80)
        print(f"  Quantized weights:      {best_balanced['quantized_weights_mb']:.4f} MB")
        print(f"  Quantization metadata:  {best_balanced['metadata_mb']:.4f} MB")
        print(f"  Sparse index overhead:  {best_balanced['sparse_overhead_mb']:.4f} MB")
        print(f"  ────────────────────────────────────")
        print(f"  Total:                  {best_balanced['final_size_mb']:.4f} MB")


        # Save summary to file
        summary_path = os.path.join(SWEEP_DIR, 'sweep_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump({
                'baseline_accuracy': float(baseline_accuracy),
                'best_accuracy': best_accuracy,
                'best_compression': best_compression,
                'best_balanced': best_balanced,
                'all_results': all_results
            }, f)
        print(f"\nSummary saved to: {summary_path}")


        # Log to wandb
        wandb.summary['best_accuracy_config'] = best_accuracy
        wandb.summary['best_compression_config'] = best_compression
        wandb.summary['best_balanced_config'] = best_balanced


    wandb.finish()



if __name__ == '__main__':
    main()
