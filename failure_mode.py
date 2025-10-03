"""
Failure Mode Analysis for Baseline MobileNetV2 Model on CIFAR-10
Generates confusion matrix, misclassification analysis, and visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import os
import argparse
from collections import defaultdict

from src.data_loader import get_cifar10_dataloaders
from src.model import MobileNetV2_CIFAR10
from src.compression.utils import load_model_checkpoint


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def evaluate_with_predictions(model, test_loader, device):
    """Evaluate model and collect all predictions"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_confidences = []
    misclassified_indices = []
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []
    misclassified_confidences = []

    global_idx = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            # Get predictions and confidences
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = probs.max(1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Identify misclassifications
            incorrect_mask = predictions.ne(targets)
            incorrect_indices = torch.where(incorrect_mask)[0]

            for idx in incorrect_indices:
                misclassified_indices.append(global_idx + idx.item())
                misclassified_images.append(data[idx].cpu())
                misclassified_preds.append(predictions[idx].item())
                misclassified_targets.append(targets[idx].item())
                misclassified_confidences.append(confidences[idx].item())

            global_idx += len(data)

    accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)

    return {
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'confidences': np.array(all_confidences),
        'accuracy': accuracy,
        'misclassified_indices': misclassified_indices,
        'misclassified_images': misclassified_images,
        'misclassified_preds': misclassified_preds,
        'misclassified_targets': misclassified_targets,
        'misclassified_confidences': misclassified_confidences
    }


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """Plot confusion matrix with annotations"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def find_most_confused_pairs(cm, class_names, top_k=10):
    """Find the most commonly confused class pairs"""
    confused_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # Exclude diagonal (correct predictions)
                confused_pairs.append({
                    'true_class': class_names[i],
                    'pred_class': class_names[j],
                    'count': cm[i, j],
                    'true_idx': i,
                    'pred_idx': j
                })

    # Sort by count
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)

    return confused_pairs[:top_k]


def plot_per_class_accuracy(cm, class_names, save_path):
    """Plot per-class accuracy bar chart"""
    # Calculate per-class accuracy
    per_class_acc = np.diag(cm) / cm.sum(axis=1) * 100

    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]

    # Create color map (red for low, green for high)
    colors = plt.cm.RdYlGn(sorted_acc / 100)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_classes, sorted_acc, color=colors)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        plt.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class accuracy to: {save_path}")

    return per_class_acc


def plot_confidence_distribution(results, save_path):
    """Plot confidence distribution for correct vs incorrect predictions"""
    correct_mask = results['predictions'] == results['targets']
    correct_confidences = results['confidences'][correct_mask]
    incorrect_confidences = results['confidences'][~correct_mask]

    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red', density=True)
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence distribution to: {save_path}")


def plot_misclassified_examples(results, class_names, save_path, num_examples=20):
    """Plot grid of misclassified examples with predictions"""
    # Get random sample of misclassifications
    num_misclassified = len(results['misclassified_images'])
    sample_size = min(num_examples, num_misclassified)

    if sample_size == 0:
        print("No misclassifications to plot!")
        return

    indices = np.random.choice(num_misclassified, sample_size, replace=False)

    # Calculate grid dimensions
    cols = 5
    rows = (sample_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if sample_size > 1 else [axes]

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        img = results['misclassified_images'][idx]
        true_label = class_names[results['misclassified_targets'][idx]]
        pred_label = class_names[results['misclassified_preds'][idx]]
        confidence = results['misclassified_confidences'][idx]

        # Denormalize image (assuming CIFAR-10 normalization)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                         fontsize=9, color='red')

    # Hide unused subplots
    for i in range(sample_size, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved misclassified examples to: {save_path}")


def analyze_failure_modes(results, cm, class_names, output_dir):
    """Comprehensive failure mode analysis"""
    print("\n" + "=" * 80)
    print("FAILURE MODE ANALYSIS")
    print("=" * 80)

    # Overall accuracy
    print(f"\nOverall Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Total Misclassifications: {len(results['misclassified_indices'])} / {len(results['targets'])}")

    # Per-class accuracy
    per_class_acc = np.diag(cm) / cm.sum(axis=1) * 100
    print(f"\nPer-Class Accuracy:")
    print("-" * 80)
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
        total_samples = cm.sum(axis=1)[i]
        correct = cm[i, i]
        print(f"  {class_name:12s}: {acc:5.2f}% ({correct:4.0f}/{total_samples:4.0f})")

    # Best and worst classes
    best_idx = np.argmax(per_class_acc)
    worst_idx = np.argmin(per_class_acc)
    print(f"\n  Best Class:  {class_names[best_idx]} ({per_class_acc[best_idx]:.2f}%)")
    print(f"  Worst Class: {class_names[worst_idx]} ({per_class_acc[worst_idx]:.2f}%)")

    # Most confused pairs
    confused_pairs = find_most_confused_pairs(cm, class_names, top_k=10)
    print(f"\nTop 10 Most Confused Class Pairs:")
    print("-" * 80)
    for i, pair in enumerate(confused_pairs, 1):
        percentage = pair['count'] / cm[pair['true_idx']].sum() * 100
        print(f"  {i:2d}. {pair['true_class']:12s} → {pair['pred_class']:12s}: "
              f"{pair['count']:4d} ({percentage:5.2f}% of {pair['true_class']})")

    # Confidence analysis
    correct_mask = results['predictions'] == results['targets']
    avg_correct_conf = np.mean(results['confidences'][correct_mask])
    avg_incorrect_conf = np.mean(results['confidences'][~correct_mask])

    print(f"\nConfidence Analysis:")
    print("-" * 80)
    print(f"  Average confidence (correct):   {avg_correct_conf:.4f}")
    print(f"  Average confidence (incorrect): {avg_incorrect_conf:.4f}")
    print(f"  Confidence gap:                 {avg_correct_conf - avg_incorrect_conf:.4f}")

    # High-confidence errors
    high_conf_threshold = 0.9
    high_conf_errors = np.sum((results['confidences'] > high_conf_threshold) & ~correct_mask)
    print(f"\n  High-confidence errors (>{high_conf_threshold}): {high_conf_errors}")

    # Save detailed report
    report_path = os.path.join(output_dir, 'failure_mode_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FAILURE MODE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Total Misclassifications: {len(results['misclassified_indices'])} / {len(results['targets'])}\n\n")

        f.write("Per-Class Accuracy:\n")
        f.write("-" * 80 + "\n")
        for class_name, acc in zip(class_names, per_class_acc):
            f.write(f"{class_name:12s}: {acc:5.2f}%\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("\nSKLEARN CLASSIFICATION REPORT:\n")
        f.write("-" * 80 + "\n")
        report = classification_report(results['targets'], results['predictions'], 
                                      target_names=class_names, digits=4)
        f.write(report)

        f.write("\n" + "-" * 80 + "\n")
        f.write("\nTOP 10 MOST CONFUSED CLASS PAIRS:\n")
        f.write("-" * 80 + "\n")
        for i, pair in enumerate(confused_pairs, 1):
            percentage = pair['count'] / cm[pair['true_idx']].sum() * 100
            f.write(f"{i:2d}. {pair['true_class']:12s} → {pair['pred_class']:12s}: "
                   f"{pair['count']:4d} ({percentage:5.2f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("\nCONFIDENCE ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average confidence (correct):   {avg_correct_conf:.4f}\n")
        f.write(f"Average confidence (incorrect): {avg_incorrect_conf:.4f}\n")
        f.write(f"High-confidence errors (>{high_conf_threshold}): {high_conf_errors}\n")

    print(f"\nDetailed report saved to: {report_path}")
    print("=" * 80 + "\n")


def analyze_confused_categories(confused_pairs):
    """Identify patterns in confused categories"""
    print("\nFAILURE MODE PATTERNS:")
    print("-" * 80)

    # Group similar confusions
    animal_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    vehicle_classes = ['airplane', 'automobile', 'ship', 'truck']

    animal_confusions = 0
    vehicle_confusions = 0
    cross_category = 0

    for pair in confused_pairs:
        true_class = pair['true_class']
        pred_class = pair['pred_class']

        if true_class in animal_classes and pred_class in animal_classes:
            animal_confusions += pair['count']
        elif true_class in vehicle_classes and pred_class in vehicle_classes:
            vehicle_confusions += pair['count']
        else:
            cross_category += pair['count']

    total = animal_confusions + vehicle_confusions + cross_category

    print(f"  Within-category confusions:")
    print(f"    Animals:  {animal_confusions:5d} ({animal_confusions/total*100:5.1f}%)")
    print(f"    Vehicles: {vehicle_confusions:5d} ({vehicle_confusions/total*100:5.1f}%)")
    print(f"  Cross-category confusions: {cross_category:5d} ({cross_category/total*100:5.1f}%)")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Failure Mode Analysis for CIFAR-10 Model')
    parser.add_argument('--model-path', type=str, default='results/baseline/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='results/baseline/failure_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = MobileNetV2_CIFAR10()
    model, _, _ = load_model_checkpoint(args.model_path, model, strict=False)
    model.to(device)

    # Load test data
    print("Loading CIFAR-10 test data...")
    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Evaluate and collect predictions
    print("Evaluating model and collecting predictions...")
    results = evaluate_with_predictions(model, test_loader, device)

    # Compute confusion matrix
    cm = confusion_matrix(results['targets'], results['predictions'])

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion matrix (absolute)
    plot_confusion_matrix(cm, CIFAR10_CLASSES, 
                         os.path.join(args.output_dir, 'confusion_matrix.png'),
                         normalize=False)

    # 2. Normalized confusion matrix
    plot_confusion_matrix(cm, CIFAR10_CLASSES,
                         os.path.join(args.output_dir, 'confusion_matrix_normalized.png'),
                         normalize=True)

    # 3. Per-class accuracy
    per_class_acc = plot_per_class_accuracy(cm, CIFAR10_CLASSES,
                                            os.path.join(args.output_dir, 'per_class_accuracy.png'))

    # 4. Confidence distribution
    plot_confidence_distribution(results, 
                                os.path.join(args.output_dir, 'confidence_distribution.png'))

    # 5. Misclassified examples
    plot_misclassified_examples(results, CIFAR10_CLASSES,
                               os.path.join(args.output_dir, 'misclassified_examples.png'),
                               num_examples=20)

    # Perform detailed analysis
    confused_pairs = find_most_confused_pairs(cm, CIFAR10_CLASSES, top_k=10)
    analyze_failure_modes(results, cm, CIFAR10_CLASSES, args.output_dir)
    analyze_confused_categories(confused_pairs)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
