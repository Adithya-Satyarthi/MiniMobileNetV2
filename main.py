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

def main():
    parser = argparse.ArgumentParser(description='MobileNetV2 CIFAR-10 Training')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'quantized'],
                        help='Training mode')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Create directories
    create_directories(config)
    
    # Initialize wandb
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            name=f"{config['wandb']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model
    model = MobileNetV2_CIFAR10(
        num_classes=config['model']['num_classes'],
        width_mult=config['model']['width_mult'],
        dropout=config['model']['dropout']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Train model
    best_model_path = trainer.train()
    
    # Evaluate on test set
    trainer.evaluate_test()
    
    # Save configuration
    save_config(config, os.path.join(config['paths']['results'], args.mode))
    
    print(f"Training completed! Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()
