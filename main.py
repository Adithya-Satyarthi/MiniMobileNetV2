"""
Unified entry point for MobileNetV2 CIFAR-10 Training and Compression
Supports: baseline training, pruning, and PTQ quantization
"""

import torch
import argparse
import yaml
import os
from datetime import datetime
import wandb

from src.data_loader import get_cifar10_dataloaders
from src.model import MobileNetV2_CIFAR10
from src.trainer import Trainer
from src.utils import set_seed, save_config, create_directories
from src.compression.pruner import ChannelPruner
from src.compression.quantizer import PTQQuantizer
from src.compression.utils import (
    load_model_checkpoint,
    save_model_checkpoint,
    print_model_summary,
    count_parameters,
    count_nonzero_parameters,
    calculate_model_size,
    calculate_quantized_model_size,
    calculate_sparsity
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


def train_baseline(config):
    """Train baseline MobileNetV2 model"""
    print("-" * 80)
    print("BASELINE TRAINING")
    print("-" * 80)
    
    set_seed(config['seed'])
    create_directories(config)
    
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            name=f"{config['wandb']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    model = MobileNetV2_CIFAR10(
        num_classes=config['model']['num_classes'],
        width_mult=config['model']['width_mult'],
        dropout=config['model']['dropout']
    )
    
    print_model_summary(model, "Baseline Model")
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    best_model_path = trainer.train()
    test_acc = trainer.evaluate_test()
    
    save_config(config, os.path.join(config['paths']['results'], 'baseline'))
    
    print(f"\n✓ Training completed: {test_acc:.2f}% test accuracy")
    print(f"  Model: {best_model_path}")
    
    return best_model_path, test_acc


def prune_model(config):
    """Prune a trained model"""
    print("-" * 80)
    print("PRUNING")
    print("-" * 80)
    
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(config['paths']['output'], exist_ok=True)
    os.makedirs(os.path.join(config['paths']['output'], 'baseline'), exist_ok=True)
    
    # Load and evaluate baseline
    model = MobileNetV2_CIFAR10()
    model, _, _ = load_model_checkpoint(config['paths']['baseline_model'], model)
    model.to(device)
    
    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=config['pruning']['batch_size'],
        num_workers=config['pruning']['num_workers']
    )
    baseline_acc = evaluate_model(model, test_loader, device)
    
    print(f"Baseline: {baseline_acc:.2f}% test accuracy")
    print_model_summary(model, "Original Model")
    
    # Prune
    pruner = ChannelPruner(model, config)
    pruned_model = pruner.prune_model()
    print_model_summary(pruned_model, "Pruned Model")
    
    # Statistics
    original_params, _ = count_parameters(model)
    pruned_params, _ = count_nonzero_parameters(pruned_model)
    sparsity = calculate_sparsity(pruned_model)
    
    print(f"\nPruning Results:")
    print(f"  Parameters: {original_params:,} → {pruned_params:,} (non-zero)")
    print(f"  Sparsity: {sparsity:.1f}%")
    print(f"  Effective compression: {original_params/pruned_params:.2f}x")
    
    # Fine-tune
    if config['pruning']['finetune_epochs'] > 0:
        print("\nFine-tuning pruned model...")
        
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            batch_size=config['pruning']['batch_size'],
            num_workers=config['pruning']['num_workers']
        )
        
        finetune_config = {
            'training': {
                'epochs': config['pruning']['finetune_epochs'],
                'batch_size': config['pruning']['batch_size'],
                'learning_rate': config['pruning']['finetune_lr'],
                'optimizer': 'sgd',
                'momentum': config['pruning']['finetune_momentum'],
                'weight_decay': config['pruning']['finetune_weight_decay'],
                'label_smoothing': 0.0,
                'scheduler': 'cosine',
                'num_workers': config['pruning']['num_workers']
            },
            'paths': {'results': config['paths']['output'], 'data': config['paths']['data']},
            'wandb': {'enabled': False, 'project': 'mobilenetv2-cifar10', 'run_name': 'pruned'}
        }
        
        trainer = Trainer(pruned_model, train_loader, val_loader, test_loader, finetune_config)
        best_model_path = trainer.train()
        test_acc = trainer.evaluate_test()
        
        print(f"\n✓ Pruning completed: {baseline_acc:.2f}% → {test_acc:.2f}% ({baseline_acc - test_acc:.2f}% drop)")
        print(f"  Model: {best_model_path}")
        
        return best_model_path, test_acc
    else:
        save_path = os.path.join(config['paths']['output'], 'pruned_model.pth')
        save_model_checkpoint(pruned_model, None, 0, baseline_acc, save_path, config)
        print(f"\n✓ Pruned model saved (not fine-tuned): {save_path}")
        return save_path, baseline_acc


def quantize_model(config):
    """Quantize a model using PTQ"""
    print("-" * 80)
    print("POST-TRAINING QUANTIZATION (PTQ)")
    print("-" * 80)
    
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = os.path.join(config['paths']['output'], 'ptq')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'baseline'), exist_ok=True)
    
    # Load and evaluate input model
    model = MobileNetV2_CIFAR10()
    model, _, _ = load_model_checkpoint(config['paths']['input_model'], model)
    model.to(device)
    
    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=config['quantization']['ptq'].get('batch_size', 128),
        num_workers=config['quantization']['ptq'].get('num_workers', 4)
    )
    baseline_acc = evaluate_model(model, test_loader, device)
    
    # Combined FP32 Model Summary
    total_params, _ = count_parameters(model)
    size_fp32 = calculate_model_size(model, bits=32)
    
    print("\nFP32 Model")
    print("-" * 80)
    print(f"Accuracy: {baseline_acc:.2f}%")
    print(f"Parameters: {total_params:,}")
    print(f"Size: {size_fp32:.2f} MB")
    
    # Print quantization config
    bits_config = config['quantization']['bits']
    print(f"\nQuantization Configuration:")
    print(f"  First conv:        W{bits_config['first_conv']['weight_bits']}A{bits_config['first_conv']['activation_bits']}")
    print(f"  Inverted residual: W{bits_config['inverted_residual']['weight_bits']}A{bits_config['inverted_residual']['activation_bits']}")
    print(f"  Final conv:        W{bits_config['final_conv']['weight_bits']}A{bits_config['final_conv']['activation_bits']}")
    print(f"  Classifier:        W{bits_config['classifier']['weight_bits']}A{bits_config['classifier']['activation_bits']}")
    
    # Get data loader for calibration
    train_loader, _, _ = get_cifar10_dataloaders(
        batch_size=config['quantization']['ptq'].get('batch_size', 128),
        num_workers=config['quantization']['ptq'].get('num_workers', 4)
    )
    
    # Apply PTQ
    quantizer = PTQQuantizer(model, config)
    quantizer.calibrate(train_loader)
    quantized_model = quantizer.quantize()
    quantized_model.to(device)
    
    # Evaluate
    test_acc = evaluate_model(quantized_model, test_loader, device)
    quantized_size = calculate_quantized_model_size(model, bits_config)
    
    print(f"\nQuantized Model Results:")
    print(f"  Accuracy: {baseline_acc:.2f}% → {test_acc:.2f}% ({baseline_acc - test_acc:.2f}% drop)")
    print(f"  Size: {size_fp32:.2f} MB → {quantized_size:.2f} MB ({size_fp32/quantized_size:.2f}x compression)")
    
    # Save
    save_path = os.path.join(output_dir, 'quantized_model.pth')
    save_model_checkpoint(quantized_model, None, 0, test_acc, save_path, config)
    print(f"  Saved: {save_path}")
    print("-" * 80)
    
    return save_path, test_acc



def full_pipeline(prune_config, quant_config):
    """Run full compression pipeline: Pruning → PTQ"""
    print("-" * 80)
    print("FULL PIPELINE: Pruning → PTQ")
    print("-" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Baseline stats
    baseline_path = prune_config['paths']['baseline_model']
    model = MobileNetV2_CIFAR10()
    model, _, _ = load_model_checkpoint(baseline_path, model)
    model.to(device)
    
    _, _, test_loader = get_cifar10_dataloaders(batch_size=128, num_workers=4)
    baseline_acc = evaluate_model(model, test_loader, device)
    baseline_params, _ = count_parameters(model)
    baseline_size = calculate_model_size(model, bits=32)
    
    print(f"\nBaseline: {baseline_acc:.2f}% | {baseline_params:,} params | {baseline_size:.2f} MB")
    
    # Step 1: Pruning
    print("\n" + "-" * 80)
    print("STEP 1: PRUNING")
    print("-" * 80)
    pruned_path, pruned_acc = prune_model(prune_config)
    
    model = MobileNetV2_CIFAR10()
    model, _, _ = load_model_checkpoint(pruned_path, model)
    pruned_params, _ = count_nonzero_parameters(model)
    pruned_size = calculate_model_size(model, bits=32, count_nonzero_only=True)
    
    # Step 2: PTQ Quantization
    print("\n" + "-" * 80)
    print("STEP 2: PTQ QUANTIZATION")
    print("-" * 80)
    quant_config['paths']['input_model'] = pruned_path
    final_path, final_acc = quantize_model(quant_config)
    
    try:
        model = MobileNetV2_CIFAR10()
        model, _, _ = load_model_checkpoint(final_path, model)
        final_params, _ = count_nonzero_parameters(model)
    except RuntimeError:
        final_params = pruned_params
    
    final_size = calculate_quantized_model_size(model, quant_config['quantization']['bits'])
    
    # Summary
    print("\n" + "-" * 80)
    print("COMPRESSION PIPELINE SUMMARY")
    print("-" * 80)
    print(f"\n{'Stage':<25} {'Accuracy':<12} {'Params (non-zero)':<20} {'Size (MB)':<12}")
    print("-" * 69)
    print(f"{'Baseline':<25} {baseline_acc:>10.2f}% {baseline_params:>19,} {baseline_size:>11.2f}")
    print(f"{'After Pruning':<25} {pruned_acc:>10.2f}% {pruned_params:>19,} {pruned_size:>11.2f}")
    print(f"{'After PTQ':<25} {final_acc:>10.2f}% {final_params:>19,} {final_size:>11.2f}")
    print("-" * 69)
    
    param_ratio = baseline_params / pruned_params if pruned_params > 0 else 1.0
    size_ratio = baseline_size / final_size if final_size > 0 else 1.0
    acc_drop = baseline_acc - final_acc
    
    print(f"\nFinal compression: {size_ratio:.2f}x size, {param_ratio:.2f}x params")
    print(f"Accuracy drop: {acc_drop:.2f}%")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='MobileNetV2 CIFAR-10 Training and Compression')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'prune', 'quantize', 'compress'],
                        help='Execution mode')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--prune-config', type=str, default='configs/pruning.yaml',
                        help='Path to pruning config')
    parser.add_argument('--quant-config', type=str, default='configs/quantization.yaml',
                        help='Path to quantization config')
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        with open(args.config) as f:
            train_baseline(yaml.safe_load(f))
        
    elif args.mode == 'prune':
        with open(args.prune_config) as f:
            prune_model(yaml.safe_load(f))
        
    elif args.mode == 'quantize':
        with open(args.quant_config) as f:
            quantize_model(yaml.safe_load(f))
        
    elif args.mode == 'compress':
        with open(args.prune_config) as f:
            prune_config = yaml.safe_load(f)
        with open(args.quant_config) as f:
            quant_config = yaml.safe_load(f)
        full_pipeline(prune_config, quant_config)


if __name__ == '__main__':
    main()
