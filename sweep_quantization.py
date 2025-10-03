"""
WandB Sweep for Quantization Hyperparameter Search
Explores different bit-width configurations and visualizes with Parallel Coordinates
"""

import torch
import yaml
import wandb
import argparse
import os
from pathlib import Path

from src.data_loader import get_cifar10_dataloaders
from src.model import MobileNetV2_CIFAR10
from src.compression.quantizer import PTQQuantizer
from src.compression.utils import (
    load_model_checkpoint,
    count_parameters,
    count_nonzero_parameters,
    calculate_quantized_model_size
)


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


def quantize_and_evaluate(config=None):
    """
    Single quantization experiment with given bit-width configuration
    Logs results to WandB for parallel coordinates visualization
    """
    with wandb.init(config=config):
        config = wandb.config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pruned model (or baseline if not using pruned)
        model = MobileNetV2_CIFAR10()
        input_model_path = config.input_model
        model, _, _ = load_model_checkpoint(input_model_path, model, strict=False)
        model.to(device)
        
        # Get baseline stats
        baseline_params, _ = count_parameters(model)
        nonzero_params, _ = count_nonzero_parameters(model)
        
        # Create quantization config from sweep parameters
        quant_config = {
            'quantization': {
                'bits': {
                    'first_conv': {
                        'weight_bits': config.first_conv_w,
                        'activation_bits': config.first_conv_a
                    },
                    'inverted_residual': {
                        'weight_bits': config.inverted_residual_w,
                        'activation_bits': config.inverted_residual_a
                    },
                    'final_conv': {
                        'weight_bits': config.final_conv_w,
                        'activation_bits': config.final_conv_a
                    },
                    'classifier': {
                        'weight_bits': config.classifier_w,
                        'activation_bits': config.classifier_a
                    }
                },
                'ptq': {
                    'calibration_batches': 100,
                    'batch_size': 128,
                    'num_workers': 4
                }
            },
            'seed': 42
        }
        
        # Get data loaders
        train_loader, _, test_loader = get_cifar10_dataloaders(
            batch_size=128,
            num_workers=4
        )
        
        # Evaluate baseline
        baseline_acc = evaluate_model(model, test_loader, device)
        
        # Apply PTQ
        quantizer = PTQQuantizer(model, quant_config)
        quantizer.calibrate(train_loader)
        quantized_model = quantizer.quantize()
        quantized_model.to(device)
        
        # Evaluate quantized model
        quantized_acc = evaluate_model(quantized_model, test_loader, device)
        
        # Calculate compression metrics
        quantized_size = calculate_quantized_model_size(
            model, 
            quant_config['quantization']['bits'],
            count_nonzero_only=True
        )
        baseline_size = calculate_quantized_model_size(
            model,
            {
                'first_conv': {'weight_bits': 32, 'activation_bits': 32},
                'inverted_residual': {'weight_bits': 32, 'activation_bits': 32},
                'final_conv': {'weight_bits': 32, 'activation_bits': 32},
                'classifier': {'weight_bits': 32, 'activation_bits': 32}
            },
            count_nonzero_only=True
        )
        
        compression_ratio = baseline_size / quantized_size
        accuracy_drop = baseline_acc - quantized_acc
        
        # Log metrics to WandB
        wandb.log({
            # Bit-widths (for parallel coordinates)
            'first_conv_w': config.first_conv_w,
            'first_conv_a': config.first_conv_a,
            'inverted_residual_w': config.inverted_residual_w,
            'inverted_residual_a': config.inverted_residual_a,
            'final_conv_w': config.final_conv_w,
            'final_conv_a': config.final_conv_a,
            'classifier_w': config.classifier_w,
            'classifier_a': config.classifier_a,
            
            # Metrics (objectives for parallel coordinates)
            'accuracy': quantized_acc,
            'accuracy_drop': accuracy_drop,
            'model_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'nonzero_params': nonzero_params,
            
            # Additional info
            'baseline_accuracy': baseline_acc,
            'baseline_size_mb': baseline_size,
        })
        
        print(f"\nConfiguration:")
        print(f"  First Conv: W{config.first_conv_w}A{config.first_conv_a}")
        print(f"  Inverted Residual: W{config.inverted_residual_w}A{config.inverted_residual_a}")
        print(f"  Final Conv: W{config.final_conv_w}A{config.final_conv_a}")
        print(f"  Classifier: W{config.classifier_w}A{config.classifier_a}")
        print(f"\nResults:")
        print(f"  Accuracy: {baseline_acc:.2f}% → {quantized_acc:.2f}% ({accuracy_drop:.2f}% drop)")
        print(f"  Size: {baseline_size:.2f} MB → {quantized_size:.2f} MB ({compression_ratio:.2f}x)")
        print(f"  Non-zero params: {nonzero_params:,}")


def main():
    parser = argparse.ArgumentParser(description='WandB Sweep for Quantization')
    parser.add_argument('--input-model', type=str, 
                        default='results/pruned/pruned_model.pth',
                        help='Path to pruned model checkpoint')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='WandB sweep ID (if resuming)')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of sweep runs')
    args = parser.parse_args()
    
    # WandB Sweep Configuration
    sweep_config = {
        'name': 'quantization-bit-width-sweep',
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'input_model': {'value': args.input_model},
            
            # First Conv bit-widths
            'first_conv_w': {
                'values': [4, 6, 8]
            },
            'first_conv_a': {
                'values': [4, 6, 8]
            },
            
            # Inverted Residual bit-widths (most layers)
            'inverted_residual_w': {
                'values': [2, 3, 4, 6, 8]
            },
            'inverted_residual_a': {
                'values': [4, 6, 8]
            },
            
            # Final Conv bit-widths
            'final_conv_w': {
                'values': [4, 6, 8]
            },
            'final_conv_a': {
                'values': [4, 6, 8]
            },
            
            # Classifier bit-widths
            'classifier_w': {
                'values': [4, 6, 8]
            },
            'classifier_a': {
                'values': [8]  # Keep high for final layer
            }
        }
    }
    
    # Initialize or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep_config,
            project='mobilenetv2-quantization-sweep'
        )
        print(f"Created new sweep: {sweep_id}")
    
    # Run sweep
    wandb.agent(sweep_id, function=quantize_and_evaluate, count=args.count)


if __name__ == '__main__':
    main()
