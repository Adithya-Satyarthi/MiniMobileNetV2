"""
General utility functions for the project
"""

import torch
import numpy as np
import random
import os
import yaml


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config):
    """Create necessary directories for experiments"""
    dirs_to_create = [
        config['paths']['results'],
        os.path.join(config['paths']['results'], 'baseline'),
        os.path.join(config['paths']['results'], 'pruned'),
        os.path.join(config['paths']['results'], 'quantized'),
        config['paths'].get('data', 'data')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


def save_config(config, save_dir):
    """Save configuration to YAML file"""
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")
