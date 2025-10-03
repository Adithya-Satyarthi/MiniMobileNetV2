"""
Unified entry point for MobileNetV2 CIFAR-10 Training and Compression
Supports: baseline training, pruning, and PTQ quantization
"""

import torch
import torch.nn as nn
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
    """
    Iterative unstructured pruning with fine-tuning
    """
    print("-" * 80)
    print("ITERATIVE PRUNING")
    print("-" * 80)

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directories
    os.makedirs(config['paths']['output'], exist_ok=True)

    # Load baseline model
    print("\nLoading baseline model...")
    model = MobileNetV2_CIFAR10(
        num_classes=config['model']['num_classes'],
        width_mult=config['model']['width_mult'],
        dropout=config['model']['dropout']
    )
    model, _, _ = load_model_checkpoint(config['paths']['baseline_model'], model, strict=False)
    model.to(device)

    # Get data loaders
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['pruning']['batch_size'],
        num_workers=config['pruning']['num_workers']
    )

    # Evaluate baseline
    baseline_acc = evaluate_model(model, test_loader, device)
    baseline_params, _ = count_parameters(model)

    print(f"\n{'-'*80}")
    print(f"Baseline Model Statistics")
    print(f"{'-'*80}")
    print(f"Test Accuracy: {baseline_acc:.2f}%")
    print(f"Total Parameters: {baseline_params:,}")
    print(f"{'-'*80}\n")

    # Initialize pruner
    from src.compression.pruner import UnstructuredPruner
    pruner = UnstructuredPruner(model, config)

    # Register hooks to keep pruned weights at zero during training
    pruner.register_mask_hooks()

    # Extract pruning hyperparameters
    num_iterations = config['pruning']['num_iterations']
    finetune_epochs = config['pruning']['finetune_epochs']
    base_lr = config['pruning']['finetune_lr']
    max_accuracy_drop = config['pruning']['max_accuracy_drop']

    # Track progress
    history = []

    print(f"\n{'-'*80}")
    print(f"Starting Iterative Pruning: {num_iterations} iterations")
    print(f"Fine-tuning: {finetune_epochs} epochs per iteration")
    print(f"Max acceptable accuracy drop: {max_accuracy_drop}%")
    print(f"{'-'*80}\n")

    # Iterative pruning loop
    for iteration in range(num_iterations):
        print(f"\n{'-'*80}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'-'*80}\n")

        # Step 1: Prune weights based on magnitude
        prune_stats = pruner.prune_step(iteration)

        # Check if we should stop early
        if prune_stats['should_stop']:
            print(f"\n{'!'*80}")
            print(f"! EARLY STOPPING TRIGGERED")
            print(f"{'!'*80}")
            print(f"Target sparsity ({config['pruning']['target_sparsity']*100:.1f}%) achieved.")
            print(f"Stopping at iteration {iteration + 1}/{num_iterations}")
            print(f"{'!'*80}\n")

            # Evaluate at this point
            acc_before_finetune = evaluate_model(model, test_loader, device)
            accuracy_drop = baseline_acc - acc_before_finetune

            print(f"Accuracy after pruning: {acc_before_finetune:.2f}% (drop: {accuracy_drop:.2f}%)")

            # Still fine-tune this iteration
            if finetune_epochs > 0:
                print(f"\nFine-tuning for {finetune_epochs} epochs...")
                print("-" * 80)

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=base_lr,
                    momentum=config['pruning']['finetune_momentum'],
                    weight_decay=config['pruning']['finetune_weight_decay']
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=finetune_epochs
                )

                criterion = nn.CrossEntropyLoss()

                for epoch in range(finetune_epochs):
                    model.train()
                    train_loss = 0.0
                    correct = 0
                    total = 0

                    for batch_idx, (data, targets) in enumerate(train_loader):
                        data, targets = data.to(device), targets.to(device)

                        optimizer.zero_grad()
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        loss.backward()

                        pruner.apply_masks()
                        optimizer.step()
                        pruner.apply_masks()

                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                    train_acc = 100. * correct / total
                    avg_loss = train_loss / len(train_loader)
                    val_acc = evaluate_model(model, val_loader, device)
                    scheduler.step()

                    print(f"  Epoch {epoch+1}/{finetune_epochs}: "
                          f"Loss={avg_loss:.4f}, "
                          f"Train Acc={train_acc:.2f}%, "
                          f"Val Acc={val_acc:.2f}%")

                print("-" * 80)

            acc_after_finetune = evaluate_model(model, test_loader, device)
            accuracy_drop = baseline_acc - acc_after_finetune

            sparsity_stats = pruner.get_sparsity_stats()
            current_sparsity = sparsity_stats['global']['sparsity']
            nonzero_params = sparsity_stats['global']['nonzero']

            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1} Summary (FINAL)")
            print(f"{'='*80}")
            print(f"Sparsity: {current_sparsity:.2f}%")
            print(f"Non-zero parameters: {nonzero_params:,} / {baseline_params:,}")
            print(f"Accuracy: {acc_before_finetune:.2f}% → {acc_after_finetune:.2f}% "
                  f"(recovered: {acc_after_finetune - acc_before_finetune:+.2f}%)")
            print(f"Accuracy drop from baseline: {accuracy_drop:.2f}%")
            print(f"{'='*80}\n")

            history.append({
                'iteration': iteration + 1,
                'sparsity': current_sparsity,
                'acc_before_finetune': acc_before_finetune,
                'acc_after_finetune': acc_after_finetune,
                'accuracy_drop': accuracy_drop,
                'nonzero_params': nonzero_params,
                'early_stopped': True
            })

            # Break out of loop
            break

        # Step 2: Evaluate immediately after pruning
        acc_before_finetune = evaluate_model(model, test_loader, device)
        accuracy_drop = baseline_acc - acc_before_finetune

        print(f"Accuracy after pruning: {acc_before_finetune:.2f}% (drop: {accuracy_drop:.2f}%)")

        # Step 3: Fine-tune to recover accuracy
        if finetune_epochs > 0:
            print(f"\nFine-tuning for {finetune_epochs} epochs...")
            print("-" * 80)

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=base_lr,
                momentum=config['pruning']['finetune_momentum'],
                weight_decay=config['pruning']['finetune_weight_decay']
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=finetune_epochs
            )

            criterion = nn.CrossEntropyLoss()

            for epoch in range(finetune_epochs):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    pruner.apply_masks()
                    optimizer.step()
                    pruner.apply_masks()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                train_acc = 100. * correct / total
                avg_loss = train_loss / len(train_loader)
                val_acc = evaluate_model(model, val_loader, device)
                scheduler.step()

                print(f"  Epoch {epoch+1}/{finetune_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.2f}%, "
                      f"Val Acc={val_acc:.2f}%")

            print("-" * 80)

        # Step 4: Evaluate after fine-tuning
        acc_after_finetune = evaluate_model(model, test_loader, device)
        accuracy_drop = baseline_acc - acc_after_finetune

        sparsity_stats = pruner.get_sparsity_stats()
        current_sparsity = sparsity_stats['global']['sparsity']
        nonzero_params = sparsity_stats['global']['nonzero']

        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1} Summary")
        print(f"{'='*80}")
        print(f"Sparsity: {current_sparsity:.2f}%")
        print(f"Non-zero parameters: {nonzero_params:,} / {baseline_params:,}")
        print(f"Accuracy: {acc_before_finetune:.2f}% → {acc_after_finetune:.2f}% "
              f"(recovered: {acc_after_finetune - acc_before_finetune:+.2f}%)")
        print(f"Accuracy drop from baseline: {accuracy_drop:.2f}%")
        print(f"{'='*80}\n")

        history.append({
            'iteration': iteration + 1,
            'sparsity': current_sparsity,
            'acc_before_finetune': acc_before_finetune,
            'acc_after_finetune': acc_after_finetune,
            'accuracy_drop': accuracy_drop,
            'nonzero_params': nonzero_params,
            'early_stopped': False
        })

        # Check if accuracy drop exceeds threshold
        if accuracy_drop > max_accuracy_drop:
            print(f"\n[WARNING] Accuracy drop ({accuracy_drop:.2f}%) exceeds "
                  f"threshold ({max_accuracy_drop}%)!")
            print(f"Consider stopping or adjusting pruning parameters.\n")

            # Optional: hard stop if accuracy degrades too much
            if accuracy_drop > max_accuracy_drop :
                print(f"[CRITICAL] Accuracy drop too severe. Stopping pruning.")
                break

    # Final evaluation
    print(f"\n{'-'*80}")
    print(f"PRUNING COMPLETE")
    print(f"{'-'*80}\n")

    final_acc = evaluate_model(model, test_loader, device)
    final_stats = pruner.get_sparsity_stats()
    final_sparsity = final_stats['global']['sparsity']
    final_nonzero = final_stats['global']['nonzero']

    print(f"{'-'*80}")
    print(f"Final Results")
    print(f"{'-'*80}")
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - final_acc:.2f}%")
    print(f"")
    print(f"Original Parameters: {baseline_params:,}")
    print(f"Non-zero Parameters: {final_nonzero:,}")
    print(f"Parameters Removed: {baseline_params - final_nonzero:,}")
    print(f"")
    print(f"Final Sparsity: {final_sparsity:.2f}%")
    print(f"Compression Ratio: {baseline_params / final_nonzero:.2f}x")
    print(f"{'-'*80}\n")

    # Make pruning permanent (remove masks)
    pruner.make_pruning_permanent()

    # Save pruned model
    save_path = os.path.join(config['paths']['output'], 'pruned_model_final.pth')
    save_model_checkpoint(model, None, num_iterations, final_acc, save_path, config)
    print(f"Pruned model saved to: {save_path}\n")

    # Save pruning history
    history_path = os.path.join(config['paths']['output'], 'pruning_history.yaml')
    with open(history_path, 'w') as f:
        yaml.dump({
            'baseline_accuracy': float(baseline_acc),
            'final_accuracy': float(final_acc),
            'final_sparsity': float(final_sparsity),
            'iterations': history
        }, f)
    print(f"Pruning history saved to: {history_path}\n")

    print("-" * 80)

    return save_path, final_acc




def quantize_model(config):
    """Quantize a model using PTQ"""
    print("-" * 80)
    print("POST-TRAINING QUANTIZATION (PTQ)")
    print("-" * 80)
    
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = os.path.join(config['paths']['output'], 'ptq')
    os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(os.path.join(output_dir, 'baseline'), exist_ok=True)
    
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
