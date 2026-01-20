"""
Training Utilities for Protein Function Prediction.

This module provides:
- train_model: Main training function with mixed precision and early stopping
- Training loop utilities and helpers
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    patience: int = 5,
    checkpoint_dir: str = "models",
    checkpoint_name: str = "best_model.pt",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = True
) -> Dict[str, Any]:
    """
    Train a model with mixed precision, validation, and early stopping.
    
    Features:
    - Mixed Precision Training (AMP) for faster training on GPU
    - Validation loop after each epoch
    - Early stopping based on validation loss
    - Model checkpointing (saves best model)
    - Learning rate scheduling support
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function (e.g., BCEWithLogitsLoss).
        optimizer: Optimizer (e.g., AdamW).
        device: Device to train on ('cuda' or 'cpu').
        epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for improvement before stopping.
        checkpoint_dir: Directory to save model checkpoints.
        checkpoint_name: Filename for the best model checkpoint.
        scheduler: Optional learning rate scheduler.
        use_amp: Whether to use Automatic Mixed Precision. Default True.
                 Set to False if training on CPU.
    
    Returns:
        Dictionary containing:
        - 'train_losses': List of average training losses per epoch
        - 'val_losses': List of validation losses per epoch
        - 'best_epoch': Epoch with best validation loss
        - 'best_val_loss': Best validation loss achieved
        - 'stopped_early': Whether training stopped early
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_path / checkpoint_name
    
    # Initialize mixed precision scaler
    # Only use AMP on CUDA devices
    use_amp = use_amp and device == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Early stopping state
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"Max Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Checkpoint Path: {best_model_path}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("=" * 60)
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False
        )
        
        for embeddings, labels in train_pbar:
            # Move data to device
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type=device, enabled=use_amp):
                logits = model(embeddings)
                loss = loss_fn(logits, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping (helps with stability)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate loss
            train_loss_sum += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss_sum / train_batches
        train_losses.append(avg_train_loss)
        
        # =====================================================================
        # Validation Phase
        # =====================================================================
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for embeddings, labels in val_pbar:
                # Move data to device
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                # Forward pass (no AMP needed for inference, but keeping for consistency)
                with autocast(device_type=device, enabled=use_amp):
                    logits = model(embeddings)
                    loss = loss_fn(logits, labels)
                
                # Accumulate loss
                val_loss_sum += loss.item()
                val_batches += 1
                
                # Update progress bar
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate average validation loss
        avg_val_loss = val_loss_sum / val_batches
        val_losses.append(avg_val_loss)
        
        # =====================================================================
        # Learning Rate Scheduling
        # =====================================================================
        if scheduler is not None:
            # Handle ReduceLROnPlateau which needs the metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # =====================================================================
        # Epoch Summary
        # =====================================================================
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"LR: {current_lr:.2e} - "
              f"Time: {epoch_time:.1f}s")
        
        # =====================================================================
        # Early Stopping Check
        # =====================================================================
        if avg_val_loss < best_val_loss:
            # Improvement found
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss,
            }, best_model_path)
            
            print(f"  ✓ Val loss improved by {improvement:.4f}. Model saved to {best_model_path}")
        else:
            # No improvement
            patience_counter += 1
            print(f"  ✗ No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                print(f"{'='*60}")
                stopped_early = True
                break
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    if not stopped_early:
        print(f"\n{'='*60}")
        print(f"Training completed after {epochs} epochs")
        print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"{'='*60}")
    
    # Load best model weights
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'stopped_early': stopped_early
    }


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model architecture (must match saved model).
        checkpoint_path: Path to the checkpoint file.
        optimizer: Optional optimizer to load state into.
        device: Device to load model onto.
    
    Returns:
        Tuple of (model, checkpoint_info) where checkpoint_info contains
        epoch, val_loss, etc.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', None),
        'train_loss': checkpoint.get('train_loss', None)
    }
    
    print(f"Loaded checkpoint from epoch {info['epoch']} "
          f"(val_loss: {info['val_loss']:.4f})")
    
    return model, info


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Training Function Demo")
    print("=" * 60)
    
    # Create dummy data for testing
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, input_dim, num_classes):
            self.size = size
            self.input_dim = input_dim
            self.num_classes = num_classes
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            embedding = torch.randn(self.input_dim)
            # Sparse multi-hot labels (5% positive rate)
            label = (torch.rand(self.num_classes) < 0.05).float()
            return embedding, label
    
    # Parameters
    input_dim = 2560
    num_classes = 100
    batch_size = 32
    
    # Create dummy datasets
    train_dataset = DummyDataset(256, input_dim, num_classes)
    val_dataset = DummyDataset(64, input_dim, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Import model (use simple version for demo)
    from model import ProteinMLP
    
    model = ProteinMLP(num_classes=num_classes, input_dim=input_dim)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=5,
        patience=3,
        checkpoint_dir="models",
        use_amp=(device == "cuda")
    )
    
    print(f"\nTraining history:")
    print(f"  Train losses: {[f'{l:.4f}' for l in history['train_losses']]}")
    print(f"  Val losses: {[f'{l:.4f}' for l in history['val_losses']]}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Stopped early: {history['stopped_early']}")
