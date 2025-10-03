# MobileNetV2 CIFAR-10 Compression Project

A comprehensive implementation of neural network compression techniques (pruning and quantization) applied to MobileNetV2 on the CIFAR-10 dataset. This project achieves up to **18× model compression** while maintaining competitive accuracy through iterative magnitude-based pruning and post-training quantization (PTQ).

**SEED Used: 42**

## 📋 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Features](https://www.google.com/search?q=%23features)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
      - [1. Baseline Training](https://www.google.com/search?q=%231-baseline-training)
      - [2. Model Pruning](https://www.google.com/search?q=%232-model-pruning)
      - [3. Model Quantization](https://www.google.com/search?q=%233-model-quantization)
      - [4. Compression Sweep](https://www.google.com/search?q=%234-compression-sweep)
      - [5. Compression Analysis](https://www.google.com/search?q=%235-compression-analysis)
      - [6. Failure Mode Analysis](https://www.google.com/search?q=%236-failure-mode-analysis)
  - [Results](https://www.google.com/search?q=%23results)
  - [Methodology](https://www.google.com/search?q=%23methodology)

-----

## 🎯 Project Overview

This project implements state-of-the-art neural network compression techniques to reduce the size of MobileNetV2 models for CIFAR-10 classification. The compression pipeline combines:

  - **Iterative Magnitude-Based Pruning**: Progressively removes less important weights while fine-tuning to recover accuracy
  - **Post-Training Quantization (PTQ)**: Reduces bit-width of weights and activations (INT8, INT6, INT4) with symmetric quantization
  - **Sparse Storage Format**: Efficient storage using 3-bit index encoding for pruned models

**Key Results**:

  - Baseline Accuracy: 94.16% on CIFAR-10
  - Best Compression: Up to **18× size reduction** with 1.22% accuracy drop
  - Best Balanced: **8.74× compression** with 0.15% accuracy drop

-----

## 📁 Project Structure

```
.
├── configs/
│   ├── baseline.yaml         # Baseline training configuration
│   ├── pruning.yaml          # Pruning hyperparameters
│   └── quantization.yaml     # Quantization settings
├── src/
│   ├── model.py              # MobileNetV2 architecture
│   ├── data_loader.py        # CIFAR-10 data loading and augmentation
│   ├── trainer.py            # Training loop
│   ├── utils.py              # Helper functions
│   └── compression/
│       ├── pruner.py         # Pruning implementation
│       ├── quantizer.py      # PTQ quantization
│       └── utils.py          # Compression utilities and size calculation
├── main.py                   # Unified entry point for all operations
├── analyze_compression.py    # Detailed compression analysis tool
├── sweep_compression.py      # Automated compression configuration sweep
├── failure_modes.py          # Confusion matrix and error analysis
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

-----

## 🛠️ Installation

### Prerequisites

  - Python 3.8+
  - CUDA-compatible GPU (recommended for faster training)
  - 4GB+ RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/Adithya-Satyarthi/MiniMobileNetV2.git
cd MiniMobileNetV2
```

### Step 2: Create Virtual Environment

Using **venv** (Python built-in):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
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

```bash
wandb login
```

Enter your API key when prompted. Get your key from: [https://wandb.ai/settings](https://wandb.ai/settings)

-----

## 🚀 Usage

### 1\. Baseline Training

Train a standard MobileNetV2 model on CIFAR-10 from scratch.

**Command**:

```bash
python main.py --mode baseline --config configs/baseline.yaml
```

**Configuration** (`configs/baseline.yaml`):

```yaml
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

**Data Augmentation**:

  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

-----

> **Note**: After completing the baseline training, you can run the compression sweep to automatically generate and evaluate all 12 compressed model configurations. This is a convenient way to explore the trade-offs without running each step manually.

### 2\. Model Pruning

Apply iterative magnitude-based pruning to reduce model parameters.

**Command**:

```bash
python main.py --mode prune --prune-config configs/pruning.yaml
```

**Configuration** (`configs/pruning.yaml`):

```yaml
pruning:
  target_sparsity: 0.70       # 70% of weights will be zero
  num_iterations: 10          # Gradual pruning over 10 steps
  finetune_epochs: 5          # Fine-tune after each pruning step
  finetune_lr: 0.001
  max_accuracy_drop: 5.0      # Stop if accuracy drops > 5%

paths:
  baseline_model: 'results/baseline/best_model.pth'
  output: 'results/pruned'
```

-----

### 3\. Model Quantization

Apply post-training quantization (PTQ) with symmetric quantization.

**Command**:

```bash
python main.py --mode quantize --quant-config configs/quantization.yaml
```

**Configuration** (`configs/quantization.yaml`):

```yaml
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

-----

### 4\. Compression Sweep

Automatically test multiple compression configurations (different sparsity levels and bit-widths).

**Command**:

```bash
python sweep_compression.py
```

**What It Does**:

  - Tests **12 configurations**: 3 sparsity levels × 4 bit-width combinations
      - Sparsity: 50%, 70%, 90%
      - Bit-widths: W6A6, W6A8, W8A6, W8A8 (uniform across layers)
  - Reuses existing pruned models (runs pruning once per sparsity)
  - Logs all metrics to Wandb
  - Generates summary with best configurations


**Output**:

```
results/compression_sweep/
├── pruned_s50/               # 50% sparsity pruned model
├── pruned_s70/               # 70% sparsity pruned model
├── pruned_s90/               # 90% sparsity pruned model
├── s50_w6_a6/                # 50% + W6A6 quantized model
│   ├── quantized_model.pth
│   └── quantization_config.yaml
├── s50_w6_a8/                # 50% + W6A8 quantized model
├── ...                       # (12 total configurations)
└── sweep_summary.yaml        # Complete results
```

**Example Summary**:

```yaml
🏆 Best Accuracy Config:
  s50_w8_a8: 94.13% accuracy, 5.49× compression

🚀 Best Compression Config:
  s90_w6_a6: 17.99× compression, 92.94% accuracy

⚖️ Best Balanced Config:
  s70_w8_a8: 94.01% accuracy, 8.74× compression
```

**Analyzing Sweep Results**:

After the sweep completes, analyze any configuration:

```bash
python analyze_compression.py \
    --baseline results/baseline/best_model.pth \
    --compressed results/compression_sweep/s70_w8_a8/quantized_model.pth\
    --output-dir results/compression_analysis
```
**Generated Files for analyze_compression**:

```
results/compression_analysis/
└── compression_analysis.txt    # Detailed report with all metrics
```

-----

The script automatically detects bit-widths from the `quantization_config.yaml` file.

**Wandb Parallel Coordinates**:

  - View trade-offs between sparsity, bit-widths, accuracy, and compression
  - Filter configurations by constraints (e.g., accuracy \> 93%)
  - Identify Pareto-optimal solutions

-----


### 5\. Failure Mode Analysis

Analyze model errors using confusion matrices and per-class performance metrics.

**Command**:

```bash
python failure_modes.py --model-path results/baseline/best_model.pth
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

1.  **Animal Confusion** (40% of errors): Cats ↔ Dogs (\~4-5% mutual)
2.  **Vehicle Confusion** (25% of errors): Trucks ↔ Automobiles (\~3-4%)
3.  **Cross-Category** (5% of errors): Birds → Airplanes (\~2-3%)

-----

## 📊 Results

### Baseline Model

| Metric            | Value                        |
| ----------------- | ---------------------------- |
| Architecture      | MobileNetV2 (width\_mult=1.0) |
| Parameters        | 2,236,682                    |
| Model Size (FP32) | 8.5323 MB                    |
| Test Accuracy     | 94.16%                       |

### Compression Results

| Configuration      | Sparsity | Quantization | Accuracy | Size    | Compression | Accuracy Drop |
| ------------------ | -------- | ------------ | -------- | ------- | ----------- | ------------- |
| **Baseline** | 0%       | FP32         | 94.16%   | 8.53 MB | 1.00×       | 0.00%         |
| Prune Only         | 70%      | FP32         | 93.93%   | 2.63 MB | 3.24×       | 0.23%         |
| **Best Accuracy** | 50%      | W8A8         | 94.13%   | 1.55 MB | 5.49×       | 0.03%         |
| **Best Balanced** | 70%      | W8A8         | 94.01%   | 0.98 MB | **8.74×** | 0.15%         |
| **Best Compression** | 90%      | W6A6         | 92.94%   | 0.47 MB | **17.99×** | 1.22%         |

### Detailed Compression Analysis (s70\_w8\_a8)

| Metric                       | Baseline | Compressed | Ratio     |
| ---------------------------- | -------- | ---------- | --------- |
| **(a) Model Size (Storage)** | 8.53 MB  | 0.98 MB    | **8.74×** |
| **(b) Pure Weight Data** | 8.53 MB  | 0.66 MB    | **12.87×**|
| **(c) Activations (per sample)** | 18.25 MB | 5.00 MB    | **3.65×** |

**Component Breakdown (s70\_w8\_a8)**:

  - Quantized Weights: 0.66 MB (67.9%)
  - Metadata: 0.07 MB (6.7%)
  - Sparse Overhead: 0.25 MB (25.4%)
  - **Total: 0.98 MB**

-----

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
  - Overhead: 3 bits × num\_nonzero\_params
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
  - ReLU sparsity: \~19-20% of activations are zero
  - Sparse storage overhead: 3-bit indices reduce effective compression

-----

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
