"""
Unstructured weight pruning with iterative fine-tuning
Implements magnitude-based importance metric
"""

import torch
import torch.nn as nn
import numpy as np


class UnstructuredPruner:
    """
    Iterative unstructured (weight-level) magnitude-based pruning
    
    Key features:
    - Magnitude-based importance metric (L1-norm)
    - Global pruning across all eligible layers
    - Masks to keep pruned weights at zero during fine-tuning
    - Iterative pruning + fine-tuning cycles
    """
    
    def __init__(self, model, config):
        """
        Args:
            model: PyTorch model to prune
            config: Configuration dict with pruning parameters
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Pruning hyperparameters
        self.target_sparsity = config['pruning']['target_sparsity']
        self.num_iterations = config['pruning']['num_iterations']
        self.initial_sparsity = config['pruning'].get('initial_sparsity', 0.0)
        
        # Track which parameters to prune
        self.prunable_params = self._identify_prunable_parameters()
        
        # Initialize masks (1 = keep, 0 = pruned)
        self.masks = {}
        self._initialize_masks()
        
        print(f"\n{'='*80}")
        print(f"Unstructured Pruner Initialized")
        print(f"{'='*80}")
        print(f"Target sparsity: {self.target_sparsity*100:.1f}%")
        print(f"Iterations: {self.num_iterations}")
        print(f"Prunable parameters: {len(self.prunable_params)}")
        print(f"{'='*80}\n")
    
    def _identify_prunable_parameters(self):
        """
        Identify all weight parameters eligible for pruning
        Excludes biases and batch norm parameters
        
        Returns:
            List of (name, module, param_name) tuples
        """
        prunable = []
        for name, module in self.model.named_modules():
            # Prune Conv2d and Linear layer weights
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    prunable.append((name, module, 'weight'))
        return prunable
    
    def _initialize_masks(self):
        """Initialize binary masks for all prunable parameters"""
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            self.masks[f"{name}.{param_name}"] = torch.ones_like(param, dtype=torch.bool, device=self.device)
    
    def prune_step(self, iteration):
        """
        Execute one pruning step with progressive mask accumulation
        OPTIMIZED: Uses GPU operations and avoids CPU bottlenecks
        
        Standard practice in iterative pruning:
        - Masks accumulate: once a weight is pruned, it stays pruned
        - Only currently active weights are candidates for new pruning
        - Prevents "reviving" previously pruned weights
        
        Args:
            iteration: Current iteration number (0-indexed)
        
        Returns:
            dict: Statistics about pruning step, including 'should_stop' flag
        """
        print(f"\n{'='*80}")
        print(f"Pruning Step {iteration + 1}/{self.num_iterations}")
        print(f"{'='*80}")
        
        # Compute target sparsity for this iteration (global across all weights)
        t = iteration + 1
        T = self.num_iterations
        
        # Cubic sparsity schedule: s_t = s_f - (s_f - s_i) * (1 - t/T)^3
        current_target_sparsity = self.target_sparsity - (self.target_sparsity - self.initial_sparsity) * ((1 - t/T) ** 3)
        
        print(f"Target sparsity: {current_target_sparsity*100:.2f}%")
        
        # Count total parameters and currently active (non-pruned) parameters
        total_params = 0
        currently_active = 0
        
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            
            total_params += param.numel()
            currently_active += self.masks[mask_key].sum().item()
        
        # Calculate how many total weights should be pruned to reach target sparsity
        target_pruned_count = int(total_params * current_target_sparsity)
        currently_pruned = total_params - currently_active
        
        # Calculate how many ADDITIONAL weights to prune this iteration
        weights_to_prune_now = target_pruned_count - currently_pruned
        
        if weights_to_prune_now <= 0:
            print(f"Target already achieved or exceeded. No pruning needed.")
            actual_sparsity = 100.0 * currently_pruned / total_params
            print(f"Current sparsity: {actual_sparsity:.2f}%")
            print(f"{'='*80}\n")
            
            return {
                'iteration': iteration + 1,
                'threshold': 0.0,
                'target_sparsity': current_target_sparsity,
                'actual_sparsity': actual_sparsity / 100.0,
                'total_params': total_params,
                'pruned_params': currently_pruned,
                'pruned_this_iter': 0,
                'should_stop': True
            }
        
        print(f"Currently pruned: {currently_pruned:,} / {total_params:,} ({100.0*currently_pruned/total_params:.2f}%)")
        print(f"Weights to prune this iteration: {weights_to_prune_now:,}")
        
        # OPTIMIZED: Collect magnitudes efficiently on GPU
        all_magnitudes = []
        param_info = []  # Store references to parameters for later masking
        
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            current_mask = self.masks[mask_key]
            
            # Get magnitudes of active weights only (on GPU)
            active_magnitudes = param[current_mask].abs().flatten()
            
            if active_magnitudes.numel() > 0:
                all_magnitudes.append(active_magnitudes)
                # Store info for later
                param_info.append({
                    'name': name,
                    'param_name': param_name,
                    'param': param,
                    'mask_key': mask_key,
                    'mask': current_mask,
                    'num_active': active_magnitudes.numel()
                })
        
        # Concatenate all magnitudes on GPU
        all_magnitudes_tensor = torch.cat(all_magnitudes)
        
        # Find threshold using GPU operations (much faster than CPU sorting)
        # Use kthvalue to find the k-th smallest element efficiently
        k = min(weights_to_prune_now, all_magnitudes_tensor.numel())
        
        if k > 0:
            # kthvalue returns (values, indices) - we only need the value
            threshold = torch.kthvalue(all_magnitudes_tensor, k)[0].item()
            print(f"Magnitude threshold: {threshold:.6f}")
        else:
            threshold = 0.0
            print(f"Magnitude threshold: {threshold:.6f}")
        
        # Update masks based on threshold
        total_pruned_now = 0
        layer_prune_counts = {}
        
        for info in param_info:
            param = info['param']
            mask_key = info['mask_key']
            current_mask = info['mask']
            
            # Find weights that are currently active AND below threshold
            magnitudes = param.abs()
            to_prune = (magnitudes < threshold) & current_mask
            
            # Update mask: accumulate pruning (once pruned, stays pruned)
            self.masks[mask_key] = self.masks[mask_key] & ~to_prune
            
            # Apply mask to weights
            with torch.no_grad():
                param.data *= self.masks[mask_key].float()
            
            # Track statistics
            num_pruned = to_prune.sum().item()
            total_pruned_now += num_pruned
            
            if num_pruned > 0:
                layer_prune_counts[mask_key] = num_pruned
        
        # Calculate actual sparsity achieved
        total_pruned_after = currently_pruned + total_pruned_now
        actual_sparsity = 100.0 * total_pruned_after / total_params
        
        print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
        print(f"Weights pruned this iteration: {total_pruned_now:,}")
        print(f"Total weights pruned: {total_pruned_after:,} / {total_params:,}")
        
        # Show top layers affected
        if layer_prune_counts:
            sorted_layers = sorted(layer_prune_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 layers pruned this iteration:")
            for i, (layer_name, count) in enumerate(sorted_layers[:5]):
                print(f"  {i+1}. {layer_name}: {count} weights")
        
        # Check if we've met or exceeded target
        target_sparsity_percent = self.target_sparsity * 100
        should_stop = actual_sparsity >= target_sparsity_percent
        
        if should_stop:
            print(f"\n⚠️  Target sparsity ({target_sparsity_percent:.1f}%) reached or exceeded!")
            print(f"    Stopping pruning iterations early.")
        
        print(f"{'='*80}\n")
        
        return {
            'iteration': iteration + 1,
            'threshold': threshold,
            'target_sparsity': current_target_sparsity,
            'actual_sparsity': actual_sparsity / 100.0,  # Return as fraction
            'total_params': total_params,
            'pruned_params': total_pruned_after,
            'pruned_this_iter': total_pruned_now,
            'should_stop': should_stop
        }
    
    def apply_masks(self):
        """
        Apply masks to ensure pruned weights remain zero
        Call this at the beginning of each training step during fine-tuning
        """
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            with torch.no_grad():
                param.data *= self.masks[mask_key].float()
    
    def register_mask_hooks(self):
        """
        Register backward hooks to apply masks during fine-tuning
        Ensures pruned weights stay zero by zeroing their gradients
        """
        def make_hook(mask):
            def hook(grad):
                # Zero out gradients for pruned weights
                return grad * mask.float()
            return hook
        
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            if param.requires_grad:
                param.register_hook(make_hook(self.masks[mask_key]))
    
    def get_sparsity_stats(self):
        """
        Calculate detailed sparsity statistics
        
        Returns:
            dict: Sparsity statistics per layer and global
        """
        stats = {}
        total_params = 0
        total_nonzero = 0
        
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            mask = self.masks[mask_key]
            
            num_params = param.numel()
            num_nonzero = mask.sum().item()
            sparsity = 100.0 * (1 - num_nonzero / num_params)
            
            stats[mask_key] = {
                'total': num_params,
                'nonzero': num_nonzero,
                'sparsity': sparsity
            }
            
            total_params += num_params
            total_nonzero += num_nonzero
        
        stats['global'] = {
            'total': total_params,
            'nonzero': total_nonzero,
            'sparsity': 100.0 * (1 - total_nonzero / total_params)
        }
        
        return stats
    
    def make_pruning_permanent(self):
        """
        Remove mask buffers and make pruning permanent
        Call after all pruning iterations are complete
        """
        for name, module, param_name in self.prunable_params:
            param = getattr(module, param_name)
            mask_key = f"{name}.{param_name}"
            with torch.no_grad():
                param.data *= self.masks[mask_key].float()
        
        # Clear masks to save memory
        self.masks.clear()
        print("Pruning made permanent. Masks removed.")


# Legacy class for backward compatibility
class ChannelPruner:
    """
    Wrapper for compatibility - redirects to UnstructuredPruner
    """
    def __init__(self, model, config):
        print("\n[WARNING] ChannelPruner is deprecated. Using UnstructuredPruner instead.\n")
        self.pruner = UnstructuredPruner(model, config)
        self.model = model
        self.config = config
    
    def prune_model(self):
        return self.model
    
    def prune_step(self, model, step, total_steps):
        self.pruner.prune_step(step)
        return {}
