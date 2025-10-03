# MobileNetV2 CIFAR-10 Compression Project

A comprehensive implementation of neural network compression techniques (pruning and quantization) applied to MobileNetV2 on the CIFAR-10 dataset. This project achieves up to **18× model compression** while maintaining competitive accuracy through iterative magnitude-based pruning and post-training quantization (PTQ).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Baseline Training](#1-baseline-training)
  - [2. Model Pruning](#2-model-pruning)
  - [3. Model Quantization](#3-model-quantization)
  - [4. Compression Analysis](#4-compression-analysis)
  - [5. Failure Mode Analysis](#5-failure-mode-analysis)
  - [6. Compression Sweep](#6-compression-sweep)
- [Results](#results)
- [Methodology](#methodology)
---

## 🎯 Project Overview

This project implements state-of-the-art neural network compression techniques to reduce the size of MobileNetV2 models for CIFAR-10 classification. The compression pipeline combines:

- **Iterative Magnitude-Based Pruning**: Progressively removes less important weights while fine-tuning to recover accuracy
- **Post-Training Quantization (PTQ)**: Reduces bit-width of weights and activations (INT8, INT6, INT4) with symmetric quantization
- **Sparse Storage Format**: Efficient storage using 3-bit index encoding for pruned models

**Key Results**:
- Baseline Accuracy: 94.16% on CIFAR-10
- Best Compression: Up to **18× size reduction** with 1.22% accuracy drop
- Best Balanced: **8.74× compression** with 0.15% accuracy drop

***

## 📁 Project Structure

```
.
├── configs/
│   ├── baseline.yaml           # Baseline training configuration
│   ├── pruning.yaml            # Pruning hyperparameters
│   └── quantization.yaml       # Quantization settings
├── src/
│   ├── model.py                # MobileNetV2 architecture
│   ├── data_loader.py          # CIFAR-10 data loading and augmentation
│   ├── trainer.py              # Training loop
│   ├── utils.py                # Helper functions
│   └── compression/
│       ├── pruner.py           # Pruning implementation
│       ├── quantizer.py        # PTQ quantization
│       └── utils.py            # Compression utilities and size calculation
├── main.py                     # Unified entry point for all operations
├── analyze_compression.py      # Detailed compression analysis tool
├── sweep_compression.py        # Automated compression configuration sweep
├── analyze_failure_modes.py   # Confusion matrix and error analysis
├── fix_sweep_configs.py        # Fix quantization configs for sweep models
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

***

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- 4GB+ RAM

### Step 1: Clone the Repository

```
git clone https://github.com/Adithya-Satyarthi/MiniMobileNetV2.git
cd MiniMobileNetV2
```

### Step 2: Create Virtual Environment

Using **venv** (Python built-in):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using **conda**:
```
conda create -n compression python=3.10
conda activate compression
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

**Key dependencies**:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Vision datasets and transforms
- `wandb>=0.15.0` - Experiment tracking
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization (for confusion matrices)
- `scikit-learn>=1.2.0` - ML utilities (classification metrics)
- `pyyaml>=6.0` - Configuration file parsing

### Step 4: Login to Wandb (Optional but Recommended)

```
wandb login
```

Enter your API key when prompted. Get your key from: [https://wandb.ai/settings](https://wandb.ai/settings)

***

## 🚀 Usage

### 1. Baseline Training

Train a standard MobileNetV2 model on CIFAR-10 from scratch.

**Command**:
```
python main.py --mode baseline --config configs/baseline.yaml
```

**Configuration** (`configs/baseline.yaml`):
```
model:
  num_classes: 10
  width_mult: 1.0
  dropout: 0.2

training:
  epochs: 200
  batch_size: 128
  
  optimizer:
    type: 'SGD'
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001
  
  scheduler:
    type: 'CosineAnnealingLR'
    T_max: 200
    eta_min: 0.0001
```

**Expected Output**:
- Training logs printed to console
- Best model saved to: `results/baseline/best_model.pth`
- Final test accuracy: ~94%
- Training time: ~2-4 hours on GPU

**Data Augmentation**:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

***

### 2. Model Pruning

Apply iterative magnitude-based pruning to reduce model parameters.

**Command**:
```
python main.py --mode prune --prune-config configs/pruning.yaml
```

**Configuration** (`configs/pruning.yaml`):
```
pruning:
  target_sparsity: 0.70          # 70% of weights will be zero
  num_iterations: 10             # Gradual pruning over 10 steps
  finetune_epochs: 5             # Fine-tune after each pruning step
  finetune_lr: 0.001
  max_accuracy_drop: 5.0         # Stop if accuracy drops > 5%

paths:
  baseline_model: 'results/baseline/best_model.pth'
  output: 'results/pruned'
```

**Expected Output**:
- Pruned model saved to: `results/pruned/pruned_model_final.pth`
- Pruning history: `results/pruned/pruning_history.yaml`
- 70% sparsity: ~695K non-zero params (from 2.24M)
- Accuracy drop: ~0.2%
- Pruning time: ~1-2 hours

**How It Works**:
1. Start with baseline model
2. For each iteration:
   - Remove lowest-magnitude weights
   - Fine-tune remaining weights for recovery
   - Evaluate and log metrics
3. Make pruning permanent (remove masks)

***

### 3. Model Quantization

Apply post-training quantization (PTQ) with symmetric quantization.

**Command**:
```
python main.py --mode quantize --quant-config configs/quantization.yaml
```

**Configuration** (`configs/quantization.yaml`):
```
quantization:
  bits:
    first_conv:
      weight_bits: 8
      activation_bits: 8
    inverted_residual:
      weight_bits: 8
      activation_bits: 8
    final_conv:
      weight_bits: 8
      activation_bits: 8
    classifier:
      weight_bits: 8
      activation_bits: 8
  
  ptq:
    calibration_batches: 100
    batch_size: 128

paths:
  input_model: 'results/pruned/pruned_model_final.pth'
  output: 'results/quantized'
```

**Expected Output**:
- Quantized model: `results/quantized/ptq/quantized_model.pth`
- Quantization config: `results/quantized/quantization_config.yaml`
- INT8 quantization: ~4× reduction from FP32
- Combined with 70% pruning: ~8-10× total compression
- Accuracy drop from baseline: ~0.15%

**Size Breakdown** (example for 70% sparsity + INT8):
```
Quantized weights:      0.6630 MB (67.9%)
Quantization metadata:  0.0653 MB (6.7%)
Sparse index overhead:  0.2481 MB (25.4%)
────────────────────────────────
Total:                  0.9764 MB
Compression: 8.74×
```

**Note**: Symmetric quantization is used (zero-point = 0), so only scale factors are stored as metadata.

***

### 4. Compression Analysis

Detailed analysis of compression ratios for weights, activations, and overall model size.

**Command**:
```
python analyze_compression.py \
    --baseline results/baseline/best_model.pth \
    --compressed results/quantized/ptq/quantized_model.pth \
    --output-dir results/compression_analysis
```

**Optional Arguments**:
```
--no-sparse-format           # Exclude sparse storage overhead
--index-bits 3               # Bits per sparse index (default: 3)
--no-activation-sparsity     # Don't measure activation sparsity
```

**What It Analyzes**:

**(a) Overall Model Compression Ratio**
- Baseline: Total parameters × 32 bits
- Compressed: Non-zero parameters + metadata + sparse overhead
- Includes all storage requirements for deployment

**(b) Weight Compression Ratio** (pure weights only)
- Excludes metadata and sparse overhead
- Shows combined effect of pruning + quantization
- Theoretical upper bound for compression

**(c) Activation Compression Ratio** (runtime memory)
- Measured via forward pass profiling with hooks
- Accounts for bit-width reduction and ReLU sparsity
- Per-sample memory during inference

**(d) Final Model Size Breakdown**
- Component-wise analysis:
  - Quantized weights (non-zero values only)
  - Quantization metadata (scale factors per channel)
  - Sparse index overhead (3-bit indices)

**Example Output**:
```
================================================================================
(a) OVERALL MODEL COMPRESSION RATIO
================================================================================
Baseline Model (FP32):
  Total parameters:     2,236,682
  Model size:           8.5323 MB

Compressed Model (W8A8, 68.9% sparse):
  Non-zero parameters:  694,891
  Quantized weights:    0.6630 MB
  Quantization metadata:0.0653 MB
  Sparse overhead:      0.2481 MB
  Total model size:     0.9764 MB

Overall Compression Ratio: 8.74×
  From 8.5323 MB to 0.9764 MB

================================================================================
(b) WEIGHT COMPRESSION RATIO
================================================================================
Baseline Weights (FP32):
  All parameters:       2,236,682
  Size:                 8.5323 MB

Compressed Weights (INT8, sparse):
  Non-zero parameters:  694,891
  Pure weight data:     0.6630 MB

Weight Compression Ratio: 12.87×
  From 8.5323 MB to 0.6630 MB

================================================================================
(c) ACTIVATION COMPRESSION RATIO (Runtime Analysis)
================================================================================
Baseline Activations (FP32):
  Total elements:       4,783,370
  Natural sparsity:     19.01% (from ReLU)
  Memory (per sample):  18.2471 MB

Compressed Activations (INT8):
  Sparsity:             20.23%
  Memory (per sample):  5.0036 MB

Activation Compression Ratio: 3.65×

For batch size 128:
  Baseline: 2335.63 MB
  Compressed: 640.47 MB
```

**Generated Files**:
```
results/compression_analysis/
└── compression_analysis.txt    # Detailed report with all metrics
```

***

### 5. Failure Mode Analysis

Analyze model errors using confusion matrices and per-class performance metrics.

**Command**:
```
python analyze_failure_modes.py --model-path results/baseline/best_model.pth
```

**Optional Arguments**:
```
--output-dir results/baseline/failure_analysis  # Output directory
--batch-size 128                                # Evaluation batch size
--num-workers 4                                 # Data loading workers
```

**Generated Files**:
```
results/baseline/failure_analysis/
├── confusion_matrix.png              # Absolute counts
├── confusion_matrix_normalized.png   # Percentage view
├── per_class_accuracy.png            # Bar chart by class
├── confidence_distribution.png       # Correct vs incorrect predictions
├── misclassified_examples.png        # Grid of error examples
└── failure_mode_report.txt           # Detailed statistics
```

**Typical Failure Modes** (CIFAR-10):
1. **Animal Confusion** (40% of errors): Cats ↔ Dogs (~4-5% mutual)
2. **Vehicle Confusion** (25% of errors): Trucks ↔ Automobiles (~3-4%)
3. **Cross-Category** (5% of errors): Birds → Airplanes (~2-3%)

**Why These Occur**:
- Low 32×32 resolution loses fine-grained details
- Similar shapes and textures between classes
- Ambiguous viewpoints and poses

***

### 6. Compression Sweep

Automatically test multiple compression configurations (different sparsity levels and bit-widths).

**Command**:
```
python sweep_compression.py
```

**What It Does**:
- Tests **12 configurations**: 3 sparsity levels × 4 bit-width combinations
  - Sparsity: 50%, 70%, 90%
  - Bit-widths: W6A6, W6A8, W8A6, W8A8 (uniform across layers)
- Reuses existing pruned models (runs pruning once per sparsity)
- Logs all metrics to Wandb
- Generates summary with best configurations

**Expected Runtime**: 4-8 hours for full sweep (depends on hardware)

**Output**:
```
results/compression_sweep/
├── pruned_s50/                  # 50% sparsity pruned model
├── pruned_s70/                  # 70% sparsity pruned model
├── pruned_s90/                  # 90% sparsity pruned model
├── s50_w6_a6/                   # 50% + W6A6 quantized model
│   ├── quantized_model.pth
│   └── quantization_config.yaml
├── s50_w6_a8/                   # 50% + W6A8 quantized model
├── ...                          # (12 total configurations)
└── sweep_summary.yaml           # Complete results
```

**Example Summary**:
```
🏆 Best Accuracy Config:
  s50_w8_a8: 94.13% accuracy, 5.49× compression

🚀 Best Compression Config:
  s90_w6_a6: 17.99× compression, 92.94% accuracy

⚖️ Best Balanced Config:
  s70_w8_a8: 94.01% accuracy, 8.74× compression
```

**Analyzing Sweep Results**:

After the sweep completes, analyze any configuration:
```
python analyze_compression.py \
    --baseline results/baseline/best_model.pth \
    --compressed results/compression_sweep/s70_w8_a8/quantized_model.pth
```

The script automatically detects bit-widths from the `quantization_config.yaml` file.

**Wandb Parallel Coordinates**:
- View trade-offs between sparsity, bit-widths, accuracy, and compression
- Filter configurations by constraints (e.g., accuracy > 93%)
- Identify Pareto-optimal solutions

***

## 📊 Results

### Baseline Model

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 (width_mult=1.0) |
| Parameters | 2,236,682 |
| Model Size (FP32) | 8.5323 MB |
| Test Accuracy | 94.16% |
| Training Time | ~3 hours (GPU) |

### Compression Results

| Configuration | Sparsity | Quantization | Accuracy | Size | Compression | Accuracy Drop |
|--------------|----------|--------------|----------|------|-------------|---------------|
| **Baseline** | 0% | FP32 | 94.16% | 8.53 MB | 1.00× | 0.00% |
| Prune Only | 70% | FP32 | 93.93% | 2.63 MB | 3.24× | 0.23% |
| **Best Accuracy** | 50% | W8A8 | 94.13% | 1.55 MB | 5.49× | 0.03% |
| **Best Balanced** | 70% | W8A8 | 94.01% | 0.98 MB | **8.74×** | 0.15% |
| **Best Compression** | 90% | W6A6 | 92.94% | 0.47 MB | **17.99×** | 1.22% |

### Detailed Compression Analysis (s70_w8_a8)

| Metric | Baseline | Compressed | Ratio |
|--------|----------|------------|-------|
| **(a) Model Size (Storage)** | 8.53 MB | 0.98 MB | **8.74×** |
| **(b) Pure Weight Data** | 8.53 MB | 0.66 MB | **12.87×** |
| **(c) Activations (per sample)** | 18.25 MB | 5.00 MB | **3.65×** |

**Component Breakdown (s70_w8_a8)**:
- Quantized Weights: 0.66 MB (67.9%)
- Metadata: 0.07 MB (6.7%)
- Sparse Overhead: 0.25 MB (25.4%)
- **Total: 0.98 MB**

### Failure Mode Analysis

**Per-Class Accuracy** (Baseline):
- Best: Ship (96.2%)
- Worst: Cat (90.8%)
- Average: 94.16%

**Top Confusions**:
1. Cat → Dog: 4.8%
2. Dog → Cat: 4.2%
3. Truck → Automobile: 3.1%
4. Bird → Airplane: 2.7%

**Confidence Calibration**:
- Correct predictions: 0.94 avg confidence
- Incorrect predictions: 0.68 avg confidence
- High-confidence errors (>0.9): 32 cases

***

## 🔬 Methodology

### Pruning Strategy

**Unstructured Magnitude-Based Pruning**:
- Remove weights with smallest absolute values
- Iterative approach: gradual sparsity increase
- Fine-tuning after each pruning step
- Mask-based implementation during training

**Advantages**:
- Simple and effective
- Works well across architectures
- Minimal hyperparameter tuning

**Limitations**:
- No actual speedup without sparse kernels
- Overhead from mask operations during fine-tuning

### Quantization Strategy

**Post-Training Quantization (PTQ)**:
- Symmetric quantization (zero-point = 0)
- Per-channel quantization for weights
- Per-tensor quantization for activations
- Calibration using training data statistics

**Mixed Precision**:
- Can use different bit-widths per layer type
- Tested: 4-bit, 6-bit, 8-bit
- Uniform bit-width per configuration in sweep

### Sparse Storage Format

**3-bit Index Encoding**:
- For each non-zero parameter, store:
  - Quantized value (4-8 bits)
  - 3-bit index (distance to next non-zero)
- Overhead: 3 bits × num_nonzero_params
- Example: 695K params → 0.248 MB overhead

**Total Compressed Size**:
```
Size = weights + quantization_metadata + sparse_overhead
     = (nonzero × bits) + (channels × 4 bytes) + (nonzero × 3 bits)
```

### Activation Memory Measurement

**Forward Pass Profiling**:
- Attach hooks to all leaf modules
- Capture output tensor sizes during forward pass
- Count total elements and non-zero values (ReLU sparsity)
- Calculate memory requirements with quantization and sparse storage

**Compression Sources**:
- Bit-width reduction: FP32 → INT6/8 (4-5.33× theoretical)
- ReLU sparsity: ~19-20% of activations are zero
- Sparse storage overhead: 3-bit indices reduce effective compression

***

## 📝 Configuration Files

### Baseline Training (`configs/baseline.yaml`)

Controls model architecture and training hyperparameters.

**Key Parameters**:
- `width_mult`: Channel multiplier (1.0 = standard, 0.75 = smaller)
- `dropout`: Dropout rate before classifier (0.2 = 20%)
- `lr`: Initial learning rate (0.1 for SGD)
- `scheduler`: CosineAnnealingLR for smooth decay
- `epochs`: 200 epochs recommended

### Pruning (`configs/pruning.yaml`)

Controls pruning behavior and fine-tuning.

**Key Parameters**:
- `target_sparsity`: Final percentage of zero weights (0.70 = 70%)
- `num_iterations`: Gradual pruning steps (10 recommended)
- `finetune_epochs`: Recovery training per step (5-10)
- `finetune_lr`: Learning rate for fine-tuning (0.001)
- `max_accuracy_drop`: Stop if exceeds threshold (5.0%)

### Quantization (`configs/quantization.yaml`)

Controls bit-widths and quantization settings.

**Key Parameters**:
- `weight_bits`: Bits for weight values (4, 6, or 8)
- `activation_bits`: Bits for activations (4, 6, or 8)
- `calibration_batches`: Data batches for calibration (100)

**Note**: Separate bit-widths can be set for:
- `first_conv`: Initial 3×3 convolution
- `inverted_residual`: MobileNetV2 bottleneck blocks
- `final_conv`: Final 1×1 convolution
- `classifier`: Fully connected layer

***