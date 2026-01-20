"""
Main Training Pipeline for Protein Function Prediction.

This script orchestrates the complete training pipeline:
1. Load Gene Ontology and annotations
2. Create train/val splits based on sequence clustering
3. Initialize datasets and data loaders
4. Calculate class weights for imbalanced learning
5. Train the model with early stopping
6. Optimize per-class thresholds
7. Save final artifacts

Usage:
    python main.py --config config.yaml
    
    # Or with command line arguments:
    python main.py \
        --obo_path data/raw/Train/go-basic.obo \
        --annotations_path data/raw/Train/train_terms.tsv \
        --cluster_path data/splits/train_cluster.tsv \
        --embedding_dir data/embeddings/train \
        --output_dir results/experiment_001
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from go_labeler import GOLabeler
from data_splits import create_splits
from dataset import ProteinGODataset
from model import ProteinMLP, calculate_pos_weights, create_loss_function
from trainer import train_model, load_checkpoint
from threshold_optimizer import optimize_thresholds, evaluate_with_thresholds


def load_annotations(annotations_path: str) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    """
    Load protein-GO term annotations from TSV file.
    
    Expected format: protein_id<TAB>go_term (one annotation per line)
    
    Args:
        annotations_path: Path to annotations TSV file.
    
    Returns:
        Tuple of:
        - annotations_list: List of (protein_id, term_id) tuples for GOLabeler
        - annotations_dict: Dict mapping protein_id -> list of GO terms for Dataset
    """
    print(f"Loading annotations from {annotations_path}...")
    
    annotations_list = []
    annotations_dict = defaultdict(list)
    
    with open(annotations_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                protein_id, term_id = parts[0], parts[1]
                annotations_list.append((protein_id, term_id))
                annotations_dict[protein_id].append(term_id)
    
    annotations_dict = dict(annotations_dict)
    
    print(f"Loaded {len(annotations_list)} annotations for {len(annotations_dict)} proteins")
    
    return annotations_list, annotations_dict


def main(args):
    """Main training pipeline."""
    
    print("=" * 70)
    print("Protein Function Prediction - Training Pipeline")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Setup and Configuration
    # =========================================================================
    print("\n[Step 1] Configuration")
    print("-" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"OBO file: {args.obo_path}")
    print(f"Annotations: {args.annotations_path}")
    print(f"Clusters: {args.cluster_path}")
    print(f"Embeddings: {args.embedding_dir}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 2: Load Ontology & Annotations
    # =========================================================================
    print("\n[Step 2] Loading Ontology and Annotations")
    print("-" * 40)
    
    # Load annotations
    annotations_list, annotations_dict = load_annotations(args.annotations_path)
    
    # Initialize GOLabeler with ontology and annotations
    labeler = GOLabeler(
        obo_path=args.obo_path,
        annotations=annotations_list
    )
    
    # Build vocabulary with minimum frequency filtering
    labeler.build_label_vocabulary(min_frequency=args.min_frequency)
    
    print(f"Label vocabulary size: {labeler.vocabulary_size()}")
    
    # =========================================================================
    # Step 3: Create Train/Val Splits
    # =========================================================================
    print("\n[Step 3] Creating Train/Val Splits")
    print("-" * 40)
    
    train_ids, val_ids = create_splits(
        cluster_file=args.cluster_path,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )
    
    # Filter to only proteins that have annotations
    train_ids = train_ids & set(annotations_dict.keys())
    val_ids = val_ids & set(annotations_dict.keys())
    
    print(f"Train proteins (with annotations): {len(train_ids)}")
    print(f"Val proteins (with annotations): {len(val_ids)}")
    
    # =========================================================================
    # Step 4: Initialize Datasets & Data Loaders
    # =========================================================================
    print("\n[Step 4] Creating Datasets and DataLoaders")
    print("-" * 40)
    
    train_dataset = ProteinGODataset(
        protein_ids=list(train_ids),
        labeler=labeler,
        embedding_dir=args.embedding_dir,
        annotations=annotations_dict,
        check_exists=True
    )
    
    val_dataset = ProteinGODataset(
        protein_ids=list(val_ids),
        labeler=labeler,
        embedding_dir=args.embedding_dir,
        annotations=annotations_dict,
        check_exists=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # =========================================================================
    # Step 5: Calculate Loss Weights & Initialize Model
    # =========================================================================
    print("\n[Step 5] Calculating Loss Weights and Initializing Model")
    print("-" * 40)
    
    # Calculate positive class weights for imbalanced learning
    if args.use_pos_weights:
        pos_weights = calculate_pos_weights(train_dataset, batch_size=args.batch_size)
    else:
        pos_weights = None
    
    # Initialize model
    model = ProteinMLP(
        num_classes=labeler.vocabulary_size(),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create loss function
    loss_fn = create_loss_function(pos_weights, device=device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # =========================================================================
    # Step 6: Train the Model
    # =========================================================================
    print("\n[Step 6] Training")
    print("-" * 40)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=str(output_path),
        checkpoint_name="best_model.pt",
        scheduler=scheduler,
        use_amp=(device == "cuda")
    )
    
    # =========================================================================
    # Step 7: Optimize Thresholds
    # =========================================================================
    print("\n[Step 7] Optimizing Per-Class Thresholds")
    print("-" * 40)
    
    # Load best model (already done by train_model, but being explicit)
    best_model_path = output_path / "best_model.pt"
    model, _ = load_checkpoint(model, str(best_model_path), device=device)
    
    # Find optimal thresholds on validation set
    optimal_thresholds = optimize_thresholds(
        model=model,
        val_loader=val_loader,
        device=device,
        use_amp=(device == "cuda")
    )
    
    # Evaluate with optimized thresholds
    print("\nEvaluating with optimized thresholds:")
    eval_results = evaluate_with_thresholds(
        model=model,
        data_loader=val_loader,
        device=device,
        thresholds=optimal_thresholds,
        use_amp=(device == "cuda")
    )
    
    # =========================================================================
    # Step 8: Save Final Artifacts
    # =========================================================================
    print("\n[Step 8] Saving Artifacts")
    print("-" * 40)
    
    # Save thresholds
    thresholds_path = output_path / "thresholds.npy"
    np.save(thresholds_path, optimal_thresholds)
    print(f"Saved thresholds to {thresholds_path}")
    
    # Save labeler vocabulary mappings
    vocab_path = output_path / "vocabulary.npz"
    np.savez(
        vocab_path,
        valid_terms=np.array(labeler.valid_terms),
        term_to_index=labeler.term_to_index,
        index_to_term=labeler.index_to_term
    )
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save training history
    history_path = output_path / "training_history.npz"
    np.savez(
        history_path,
        train_losses=np.array(history['train_losses']),
        val_losses=np.array(history['val_losses']),
        best_epoch=history['best_epoch'],
        best_val_loss=history['best_val_loss']
    )
    print(f"Saved training history to {history_path}")
    
    # Save configuration
    config_path = output_path / "config.txt"
    with open(config_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nResults:\n")
        f.write(f"best_epoch: {history['best_epoch']}\n")
        f.write(f"best_val_loss: {history['best_val_loss']:.4f}\n")
        f.write(f"micro_f1: {eval_results['micro_f1']:.4f}\n")
        f.write(f"macro_f1: {eval_results['macro_f1']:.4f}\n")
    print(f"Saved config to {config_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Optimal thresholds saved to: {thresholds_path}")
    print(f"\nFinal Metrics (Validation Set):")
    print(f"  Best Val Loss: {history['best_val_loss']:.4f}")
    print(f"  Micro F1:      {eval_results['micro_f1']:.4f}")
    print(f"  Macro F1:      {eval_results['macro_f1']:.4f}")
    
    return model, optimal_thresholds, labeler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train protein function prediction model"
    )
    
    # Data paths
    parser.add_argument(
        "--obo_path",
        type=str,
        default="data/raw/Train/go-basic.obo",
        help="Path to GO ontology OBO file"
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="data/raw/Train/train_terms.tsv",
        help="Path to protein-GO annotations TSV"
    )
    parser.add_argument(
        "--cluster_path",
        type=str,
        default="data/splits/train_cluster.tsv",
        help="Path to MMseqs2 cluster file"
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings/train",
        help="Directory containing protein embedding .pt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiment_001",
        help="Output directory for model and artifacts"
    )
    
    # Model architecture
    parser.add_argument("--input_dim", type=int, default=2560, help="Input embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    # Data processing
    parser.add_argument("--min_frequency", type=int, default=50, help="Min GO term frequency")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--use_pos_weights", action="store_true", help="Use positive class weights")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run main pipeline
    model, thresholds, labeler = main(args)
