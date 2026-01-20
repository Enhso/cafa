"""
Threshold Optimization for Multi-Label Classification.

This module provides functions to find optimal decision thresholds for each
class based on the validation set performance. Instead of using a fixed 0.5
threshold for all classes, we tune per-class thresholds to maximize F1 score.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm
from typing import Tuple, Optional


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect all predictions and true labels from a data loader.
    
    Args:
        model: Trained model.
        data_loader: DataLoader to run inference on.
        device: Device to run on.
        use_amp: Whether to use automatic mixed precision.
    
    Returns:
        Tuple of (y_probs, y_true) where:
        - y_probs: numpy array of shape (n_samples, n_classes) with probabilities
        - y_true: numpy array of shape (n_samples, n_classes) with binary labels
    """
    model.eval()
    use_amp = use_amp and device == "cuda"
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm(data_loader, desc="Collecting predictions"):
            embeddings = embeddings.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type=device, enabled=use_amp):
                logits = model(embeddings)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    return y_probs, y_true


def optimize_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    use_amp: bool = True,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    default_threshold: float = 0.5
) -> np.ndarray:
    """
    Find optimal F1 threshold for each class using the validation set.
    
    For each class, we:
    1. Compute precision-recall curve at different thresholds
    2. Calculate F1 score at each threshold: F1 = 2 * (P * R) / (P + R)
    3. Select the threshold that maximizes F1
    
    This is important for imbalanced multi-label classification because:
    - Different classes have different optimal operating points
    - Rare classes may need lower thresholds to catch more positives
    - Common classes may benefit from higher thresholds to reduce FPs
    
    Args:
        model: Trained model (should be the best model from training).
        val_loader: DataLoader for validation set.
        device: Device to run inference on.
        use_amp: Whether to use automatic mixed precision.
        min_threshold: Minimum allowed threshold (avoid extreme values).
        max_threshold: Maximum allowed threshold.
        default_threshold: Default threshold if optimization fails for a class.
    
    Returns:
        Numpy array of shape (n_classes,) containing optimal thresholds.
    """
    print("=" * 60)
    print("Optimizing Per-Class Thresholds")
    print("=" * 60)
    
    # Step 1: Collect all predictions
    y_probs, y_true = collect_predictions(model, val_loader, device, use_amp)
    
    n_samples, n_classes = y_probs.shape
    print(f"Collected predictions: {n_samples} samples, {n_classes} classes")
    
    # Step 2: Find optimal threshold for each class
    optimal_thresholds = np.full(n_classes, default_threshold)
    
    print("Finding optimal thresholds per class...")
    for class_idx in tqdm(range(n_classes), desc="Optimizing thresholds"):
        y_true_class = y_true[:, class_idx]
        y_prob_class = y_probs[:, class_idx]
        
        # Skip if no positive samples (can't compute precision-recall)
        if y_true_class.sum() == 0:
            continue
        
        # Skip if no negative samples
        if y_true_class.sum() == len(y_true_class):
            continue
        
        try:
            # Compute precision-recall curve
            # Returns: precision, recall, thresholds
            # Note: len(thresholds) = len(precision) - 1
            precision, recall, thresholds = precision_recall_curve(
                y_true_class, y_prob_class
            )
            
            # Calculate F1 at each threshold
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
                f1_scores = np.nan_to_num(f1_scores, nan=0.0)
            
            # Find threshold with max F1
            if len(f1_scores) > 0 and f1_scores.max() > 0:
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
                # Clip to reasonable range
                best_threshold = np.clip(best_threshold, min_threshold, max_threshold)
                optimal_thresholds[class_idx] = best_threshold
                
        except Exception as e:
            # Keep default threshold if optimization fails
            pass
    
    # Print statistics
    print(f"\nThreshold statistics:")
    print(f"  Min: {optimal_thresholds.min():.3f}")
    print(f"  Max: {optimal_thresholds.max():.3f}")
    print(f"  Mean: {optimal_thresholds.mean():.3f}")
    print(f"  Median: {np.median(optimal_thresholds):.3f}")
    print(f"  Std: {optimal_thresholds.std():.3f}")
    
    # Count thresholds at default value
    at_default = (optimal_thresholds == default_threshold).sum()
    print(f"  Classes at default ({default_threshold}): {at_default} ({100*at_default/n_classes:.1f}%)")
    
    return optimal_thresholds


def evaluate_with_thresholds(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    thresholds: np.ndarray,
    use_amp: bool = True
) -> dict:
    """
    Evaluate model performance using optimized thresholds.
    
    Args:
        model: Trained model.
        data_loader: DataLoader for evaluation.
        device: Device to run on.
        thresholds: Per-class thresholds array.
        use_amp: Whether to use automatic mixed precision.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    # Collect predictions
    y_probs, y_true = collect_predictions(model, data_loader, device, use_amp)
    
    # Apply per-class thresholds
    y_pred = (y_probs >= thresholds).astype(np.float32)
    
    # Calculate metrics
    # Micro-averaged (treats all predictions equally)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Macro-averaged (average of per-class F1)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted (by support)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics for analysis
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate precision and recall
    from sklearn.metrics import precision_score, recall_score
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    results = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class_f1': per_class_f1,
        'n_samples': len(y_true),
        'n_classes': y_true.shape[1]
    }
    
    print("\nEvaluation Results:")
    print(f"  Micro F1:     {micro_f1:.4f}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Weighted F1:  {weighted_f1:.4f}")
    print(f"  Precision:    {micro_precision:.4f}")
    print(f"  Recall:       {micro_recall:.4f}")
    
    return results


def compare_thresholds(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    optimized_thresholds: np.ndarray,
    fixed_threshold: float = 0.5
) -> None:
    """
    Compare performance of optimized vs fixed thresholds.
    
    Args:
        model: Trained model.
        data_loader: DataLoader for evaluation.
        device: Device to run on.
        optimized_thresholds: Per-class optimized thresholds.
        fixed_threshold: Fixed threshold for comparison.
    """
    print("\n" + "=" * 60)
    print("Threshold Comparison")
    print("=" * 60)
    
    # Collect predictions once
    y_probs, y_true = collect_predictions(model, data_loader, device)
    
    # Fixed threshold
    y_pred_fixed = (y_probs >= fixed_threshold).astype(np.float32)
    f1_fixed = f1_score(y_true, y_pred_fixed, average='micro', zero_division=0)
    
    # Optimized thresholds
    y_pred_opt = (y_probs >= optimized_thresholds).astype(np.float32)
    f1_opt = f1_score(y_true, y_pred_opt, average='micro', zero_division=0)
    
    print(f"\nFixed threshold ({fixed_threshold}):")
    print(f"  Micro F1: {f1_fixed:.4f}")
    
    print(f"\nOptimized thresholds:")
    print(f"  Micro F1: {f1_opt:.4f}")
    
    improvement = (f1_opt - f1_fixed) / f1_fixed * 100 if f1_fixed > 0 else 0
    print(f"\nImprovement: {improvement:+.2f}%")


# Example usage
if __name__ == "__main__":
    print("Threshold optimization module loaded.")
    print("Use optimize_thresholds(model, val_loader, device) to find optimal thresholds.")
