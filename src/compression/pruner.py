import torch
import torch.nn as nn
import numpy as np
from torchvision.models.mobilenetv2 import InvertedResidual
import copy


class ChannelPruner:
    """
    L1-norm based channel pruning for MobileNetV2
    Applies magnitude pruning by zeroing out low-importance weights
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.pruning_ratios = {
            'first_conv': config['pruning']['first_conv'],
            'inverted_residual': config['pruning']['inverted_residual'],
            'final_conv': config['pruning']['final_conv'],
            'classifier': config['pruning']['classifier']
        }
    
    def calculate_l1_importance(self, conv_layer):
        """
        Calculate L1-norm importance score for each output channel
        """
        weights = conv_layer.weight.data.abs()
        # Sum over input channels, kernel height, kernel width
        importance = weights.sum(dim=(1, 2, 3))
        return importance
    
    def prune_model(self):
        """
        Prune the entire MobileNetV2 model using magnitude-based pruning
        Returns pruned model
        """
        print("Starting magnitude-based pruning...")
        print(f"Pruning ratios: {self.pruning_ratios}")
        
        # Create a copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        total_params = 0
        pruned_params = 0
        layers_pruned = 0
        
        # Iterate through all modules
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Determine pruning ratio based on layer location
                prune_ratio = self._get_prune_ratio_for_layer(name)
                
                if prune_ratio > 0:
                    # Calculate importance for each output channel
                    importance = self.calculate_l1_importance(module)
                    
                    num_channels = importance.size(0)
                    num_prune = int(num_channels * prune_ratio)
                    
                    if num_prune > 0 and num_prune < num_channels:
                        # Get indices of channels to prune (lowest importance)
                        _, indices_to_prune = torch.topk(importance, num_prune, largest=False)
                        
                        # Zero out the pruned channels
                        module.weight.data[indices_to_prune, :, :, :] = 0
                        
                        # Also zero out corresponding bias if it exists
                        if module.bias is not None:
                            module.bias.data[indices_to_prune] = 0
                        
                        # Calculate statistics
                        layer_params = module.weight.numel()
                        layer_pruned = indices_to_prune.size(0) * module.weight[0].numel()
                        
                        total_params += layer_params
                        pruned_params += layer_pruned
                        layers_pruned += 1
                        
                        print(f"  {name}: Pruned {num_prune}/{num_channels} channels ({prune_ratio*100:.1f}%)")
        
        # Calculate overall sparsity
        if total_params > 0:
            sparsity = 100.0 * pruned_params / total_params
            print(f"\nPruning Summary:")
            print(f"  Layers pruned: {layers_pruned}")
            print(f"  Overall sparsity: {sparsity:.2f}%")
            print(f"  Pruned parameters: {pruned_params:,} / {total_params:,}")
        
        return pruned_model
    
    def _get_prune_ratio_for_layer(self, layer_name):
        """
        Determine pruning ratio based on layer name
        """
        if 'features.0' in layer_name:
            # First conv layer
            return self.pruning_ratios['first_conv']
        elif 'features.18' in layer_name:
            # Final conv layer
            return self.pruning_ratios['final_conv']
        elif 'classifier' in layer_name:
            # Classifier layer
            return self.pruning_ratios['classifier']
        elif 'features' in layer_name:
            # Inverted residual blocks
            return self.pruning_ratios['inverted_residual']
        else:
            # Default: no pruning
            return 0.0
    