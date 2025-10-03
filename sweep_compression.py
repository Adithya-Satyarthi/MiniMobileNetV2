"""
Sweep different compression configurations (sparsity + quantization)
FIXED: Copies quantization config file so analyze_compression.py can detect correct bits
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
BIT_WIDTHS = [6, 8]

BASELINE_MODEL = "results/baseline/best_model.pth"
PRUNE_CONFIG_TEMPLATE = "configs/pruning.yaml"
QUANT_CONFIG_TEMPLATE = "configs/quantization.yaml"

SWEEP_DIR = "results/compression_sweep"
os.makedirs(SWEEP_DIR, exist_ok=True)

WANDB_PROJECT = "mobilenetv2-compression-sweep"
SWEEP_ID = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def find_pruned_model(sparsity, sweep_dir):
    """Find pruned model with multiple naming pattern attempts."""
    possible_dirs = [
        f"pruned_s{int(sparsity*100)}",
        f"prune_s{int(sparsity*100)}",
        f"s{int(sparsity*100)}",
    ]

    for dir_name in possible_dirs:
        prune_dir = os.path.join(sweep_dir, dir_name)
        pruned_model_path = os.path.join(prune_dir, 'pruned_model_final.pth')

        if os.path.exists(pruned_model_path):
            return prune_dir, pruned_model_path

    return None, None


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


def get_model_stats(model_path, weight_bits, act_bits, baseline_size_mb,
                   use_per_channel=True, use_sparse_format=True, index_bits=3):
    """Calculate model compression statistics."""
    try:
        model = MobileNetV2_CIFAR10()
        model, _, _ = load_model_checkpoint(model_path, model, strict=False)

        total_params, _ = count_parameters(model)
        nonzero_params, _ = count_nonzero_parameters(model)
        sparsity = 100.0 * (1 - nonzero_params / total_params)

        original_size_mb = baseline_size_mb
        quantized_weights_mb = (nonzero_params * weight_bits) / (8 * 1024 * 1024)

        bits_config = {
            'first_conv': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'inverted_residual': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'final_conv': {'weight_bits': weight_bits, 'activation_bits': act_bits},
            'classifier': {'weight_bits': weight_bits, 'activation_bits': act_bits}
        }

        metadata_bytes, metadata_breakdown = calculate_quantization_metadata(
            model, bits_config, use_per_channel=use_per_channel
        )
        metadata_mb = metadata_bytes / (1024 * 1024)

        sparse_overhead_mb = 0
        sparse_overhead_bits = 0
        if use_sparse_format:
            sparse_overhead_mb, sparse_overhead_bits = calculate_sparse_storage_overhead(
                nonzero_params, index_bits=index_bits
            )

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


def run_single_config(config_name, sparsity, weight_bits, act_bits, 
                     pruned_model_path, baseline_accuracy, baseline_size_mb,
                     pruned_accuracy, sweep_id):
    """Run a single compression configuration and log to wandb."""
    run = wandb.init(
        project=WANDB_PROJECT,
        name=config_name,
        group=sweep_id,
        config={
            'sparsity': sparsity * 100,
            'weight_bits': weight_bits,
            'activation_bits': act_bits,
            'baseline_accuracy': baseline_accuracy,
        },
        reinit=True
    )

    print("\n" + "=" * 80)
    print(f"Running Configuration: {config_name}")
    print("=" * 80)

    run_dir = os.path.join(SWEEP_DIR, config_name)
    quant_output_dir = os.path.join(run_dir, 'quant_output')

    _, quant_config_path = update_quantization_config(
        QUANT_CONFIG_TEMPLATE, weight_bits, act_bits, pruned_model_path, quant_output_dir
    )

    if not run_command(["python", "main.py", "--mode", "quantize", "--quant-config", quant_config_path]):
        print(f"Quantization failed for {config_name}. Skipping.")
        wandb.finish()
        return None

    quant_source = os.path.join(quant_output_dir, 'ptq', 'quantized_model.pth')
    quant_dest = os.path.join(run_dir, 'quantized_model.pth')

    # ═════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Also copy the quantization config file!
    # ═════════════════════════════════════════════════════════════════════════
    config_source = os.path.join(quant_output_dir, 'quantization_config.yaml')
    config_dest = os.path.join(run_dir, 'quantization_config.yaml')

    if os.path.exists(quant_source):
        shutil.copy(quant_source, quant_dest)
        # Copy config file so analyze_compression.py can find it
        if os.path.exists(config_source):
            shutil.copy(config_source, config_dest)
            print(f"✓ Copied quantization config to {config_dest}")
        else:
            print(f"⚠ Warning: Config not found at {config_source}")
    else:
        print(f"Quantized model not found at {quant_source}. Skipping.")
        wandb.finish()
        return None

    # Evaluate quantized model
    quant_accuracy = evaluate_model_accuracy(quant_dest)

    # Calculate stats
    stats = get_model_stats(
        quant_dest,
        weight_bits, 
        act_bits,
        baseline_size_mb,
        use_per_channel=True,
        use_sparse_format=True,
        index_bits=3
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

        wandb.log(result)
        for key, value in result.items():
            wandb.summary[key] = value

        print(f"\nResults for {config_name}:")
        print(f"  Accuracy: {baseline_accuracy:.2f}% → {quant_accuracy:.2f}% (Drop: {result['accuracy_drop']:.2f}%)")
        print(f"  Size: {stats['original_size_mb']:.4f} MB → {stats['quantized_size_mb']:.4f} MB")
        print(f"  Compression: {stats['compression_ratio']:.2f}×")
        print(f"  Sparsity: {stats['sparsity']:.2f}% ({stats['nonzero_params']:,} non-zero)")
        print(f"  Config: W{weight_bits}A{act_bits}")

        wandb.finish()
        return result

    wandb.finish()
    return None


def main():
    print("=" * 80)
    print("COMPRESSION SWEEP WITH WANDB")
    print(f"Sweep ID: {SWEEP_ID}")
    print("=" * 80)

    print("\nCalculating baseline statistics from baseline model...")
    baseline_model = MobileNetV2_CIFAR10()
    baseline_model, _, _ = load_model_checkpoint(BASELINE_MODEL, baseline_model, strict=False)
    baseline_total_params, _ = count_parameters(baseline_model)
    baseline_size_mb = (baseline_total_params * 32) / (8 * 1024 * 1024)

    print(f"Baseline Model:")
    print(f"  Path: {BASELINE_MODEL}")
    print(f"  Total parameters: {baseline_total_params:,}")
    print(f"  Size: {baseline_size_mb:.4f} MB (FP32)")

    print("\nCompression Method:")
    print("  Baseline: Total parameters from baseline model")
    print("  Compressed: Non-zero parameters from quantized model + metadata + overhead")

    print("\nEvaluating Baseline Model...")
    baseline_accuracy = evaluate_model_accuracy(BASELINE_MODEL)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%\n")

    all_results = []

    for sparsity in SPARSITY_VALUES:
        print("\n" + "#" * 80)
        print(f"# Processing Sparsity Level: {sparsity*100:.0f}%")
        print("#" * 80)

        prune_dir, pruned_model_path = find_pruned_model(sparsity, SWEEP_DIR)

        if pruned_model_path and os.path.exists(pruned_model_path):
            print(f" Found existing pruned model at: {pruned_model_path}")
        else:
            print(f"Running pruning for {sparsity*100:.0f}% sparsity...")
            prune_dir = os.path.join(SWEEP_DIR, f"pruned_s{int(sparsity*100)}")
            pruned_model_path = os.path.join(prune_dir, 'pruned_model_final.pth')

            _, prune_config_path = update_pruning_config(
                PRUNE_CONFIG_TEMPLATE, sparsity, prune_dir
            )

            if not run_command(["python", "main.py", "--mode", "prune", "--prune-config", prune_config_path]):
                print(f"Pruning failed for sparsity {sparsity}. Skipping.")
                continue

            if not os.path.exists(pruned_model_path):
                prune_dir, pruned_model_path = find_pruned_model(sparsity, SWEEP_DIR)
                if not pruned_model_path:
                    print(f"Could not locate pruned model. Skipping.")
                    continue

        pruned_accuracy = evaluate_model_accuracy(pruned_model_path)
        print(f"Pruned Model Accuracy: {pruned_accuracy:.2f}%")

        for weight_bits in BIT_WIDTHS:
            for act_bits in BIT_WIDTHS:
                config_name = f"s{int(sparsity*100)}_w{weight_bits}_a{act_bits}"

                result = run_single_config(
                    config_name, sparsity, weight_bits, act_bits,
                    pruned_model_path, baseline_accuracy, baseline_size_mb,
                    pruned_accuracy, SWEEP_ID
                )

                if result:
                    all_results.append(result)

    if all_results:
        print("\n" + "=" * 80)
        print("SWEEP SUMMARY")
        print("=" * 80)

        best_accuracy = max(all_results, key=lambda x: x['final_accuracy'])
        best_compression = max(all_results, key=lambda x: x['compression_ratio'])

        for r in all_results:
            r['score'] = (r['final_accuracy'] / baseline_accuracy) * 0.6 + (r['compression_ratio'] / 50) * 0.4
        best_balanced = max(all_results, key=lambda x: x['score'])

        print("\n Best Accuracy Config:")
        print(f"  {best_accuracy['config_name']}: {best_accuracy['final_accuracy']:.2f}% acc, {best_accuracy['compression_ratio']:.2f}× compression")

        print("\n Best Compression Config:")
        print(f"  {best_compression['config_name']}: {best_compression['compression_ratio']:.2f}× compression, {best_compression['final_accuracy']:.2f}% acc")

        print("\n  Best Balanced Config:")
        print(f"  {best_balanced['config_name']}: {best_balanced['final_accuracy']:.2f}% acc, {best_balanced['compression_ratio']:.2f}× compression")

        summary_path = os.path.join(SWEEP_DIR, 'sweep_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump({
                'sweep_id': SWEEP_ID,
                'baseline_accuracy': float(baseline_accuracy),
                'baseline_size_mb': float(baseline_size_mb),
                'baseline_params': int(baseline_total_params),
                'compression_method': 'baseline_total_params / (compressed_nonzero_params + metadata + overhead)',
                'best_accuracy': best_accuracy,
                'best_compression': best_compression,
                'best_balanced': best_balanced,
                'all_results': all_results
            }, f)
        print(f"\nSummary saved to: {summary_path}")

        print("\n" + "=" * 80)
        print(" Config files copied - analyze_compression.py will now match!")
        print("=" * 80)


if __name__ == '__main__':
    main()
