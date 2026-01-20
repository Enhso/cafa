"""
Neural Network Model and Loss Components for Protein Function Prediction.

This module provides:
- ProteinMLP: A multi-layer perceptron for multi-label classification
- calculate_pos_weights: Function to compute class weights for imbalanced data
- Loss function initialization utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


class ProteinMLP(nn.Module):
    """
    Multi-Layer Perceptron for protein function prediction.
    
    Architecture:
        Input (2560) -> Linear(1024) -> BatchNorm -> ReLU -> Dropout(0.3)
                     -> Linear(num_classes) -> Output
    
    Note: No Sigmoid activation at the output. This is intentional because
    we use BCEWithLogitsLoss which applies Sigmoid internally and is more
    numerically stable than applying Sigmoid + BCELoss separately.
    
    Attributes:
        input_dim (int): Input embedding dimension (default 2560 for ESM-2 3B).
        hidden_dim (int): Hidden layer dimension (default 1024).
        num_classes (int): Number of GO term classes to predict.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 2560,
        hidden_dim: int = 1024,
        dropout: float = 0.3
    ):
        """
        Initialize the ProteinMLP model.
        
        Args:
            num_classes: Number of output classes (GO terms in vocabulary).
            input_dim: Dimension of input embeddings. Default 2560 for ESM-2 3B.
            hidden_dim: Dimension of hidden layer. Default 1024.
            dropout: Dropout probability. Default 0.3.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Hidden layer with BatchNorm, ReLU, and Dropout
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer (no activation - using BCEWithLogitsLoss)
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Logits tensor of shape (batch_size, num_classes).
            Note: These are raw logits, NOT probabilities.
            Apply torch.sigmoid() to get probabilities for inference.
        """
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions (applies Sigmoid).
        
        Use this method during inference to get probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Probability tensor of shape (batch_size, num_classes).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


def calculate_pos_weights(
    train_dataset,
    num_workers: int = 0,
    batch_size: int = 256
) -> torch.Tensor:
    """
    Calculate positive class weights for BCEWithLogitsLoss.
    
    For imbalanced multi-label classification, we weight the positive class
    to account for the class imbalance. This helps the model learn rare
    classes better.
    
    Formula: pos_weight[i] = (num_negative[i]) / (num_positive[i])
                           = (total - num_positive[i]) / num_positive[i]
    
    A pos_weight > 1 increases the loss for false negatives, encouraging
    the model to predict more positives for rare classes.
    
    Args:
        train_dataset: PyTorch Dataset that returns (embedding, label) tuples.
        num_workers: Number of DataLoader workers. Default 0.
        batch_size: Batch size for iterating through dataset. Default 256.
    
    Returns:
        Tensor of shape (num_classes,) containing positive weights for each class.
    """
    # Create a DataLoader for efficient iteration
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Get number of classes from first sample
    _, first_label = train_dataset[0]
    num_classes = first_label.shape[0]
    
    # Accumulate positive counts for each class
    positive_counts = torch.zeros(num_classes, dtype=torch.float64)
    total_samples = 0
    
    print("Calculating positive class weights...")
    for _, labels in tqdm(loader, desc="Computing weights"):
        # labels shape: (batch_size, num_classes)
        positive_counts += labels.sum(dim=0).to(torch.float64)
        total_samples += labels.shape[0]
    
    print(f"Total samples: {total_samples}")
    print(f"Positive counts - Min: {positive_counts.min():.0f}, "
          f"Max: {positive_counts.max():.0f}, "
          f"Mean: {positive_counts.mean():.1f}")
    
    # Calculate pos_weight = (total - positives) / positives
    # Add small epsilon to avoid division by zero for classes with no positives
    epsilon = 1e-7
    negative_counts = total_samples - positive_counts
    pos_weights = negative_counts / (positive_counts + epsilon)
    
    # Clip extreme weights to avoid instability
    # Very rare classes might get huge weights, which can destabilize training
    max_weight = 100.0
    pos_weights = torch.clamp(pos_weights, min=1.0, max=max_weight)
    
    print(f"Pos weights - Min: {pos_weights.min():.2f}, "
          f"Max: {pos_weights.max():.2f}, "
          f"Mean: {pos_weights.mean():.2f}")
    
    return pos_weights.to(torch.float32)


def create_loss_function(
    pos_weights: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> nn.BCEWithLogitsLoss:
    """
    Create BCEWithLogitsLoss with optional positive weights.
    
    BCEWithLogitsLoss combines Sigmoid and BCELoss in a single layer,
    which is more numerically stable than using them separately.
    
    With pos_weight, the loss for each class becomes:
        loss = pos_weight * y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))
    
    Args:
        pos_weights: Optional tensor of positive class weights.
        device: Device to place the loss function on.
    
    Returns:
        Configured BCEWithLogitsLoss instance.
    """
    if pos_weights is not None:
        pos_weights = pos_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        print(f"Created BCEWithLogitsLoss with pos_weight on {device}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Created BCEWithLogitsLoss without pos_weight")
    
    return criterion


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Model and Loss Function Demo")
    print("=" * 60)
    
    # Demo parameters
    batch_size = 4
    input_dim = 2560  # ESM-2 3B embedding dimension
    num_classes = 1000  # Example: 1000 GO terms
    
    # Create model
    model = ProteinMLP(
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=1024,
        dropout=0.3
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, input_dim)
    logits = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    
    # Test predict_proba
    probs = model.predict_proba(dummy_input)
    print(f"\nPredict proba test:")
    print(f"  Probs shape: {probs.shape}")
    print(f"  Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # Demo loss function with random weights
    print("\n" + "-" * 40)
    print("Loss function demo:")
    
    # Simulate imbalanced pos_weights
    dummy_pos_weights = torch.rand(num_classes) * 10 + 1  # Range [1, 11]
    
    # Create loss function
    criterion = create_loss_function(dummy_pos_weights, device="cpu")
    
    # Test loss calculation
    dummy_targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    loss = criterion(logits, dummy_targets)
    print(f"  Loss value: {loss.item():.4f}")
    
    # =========================================================================
    # Full example showing how to use with real data
    # =========================================================================
    print("\n" + "=" * 60)
    print("Usage Example Code:")
    print("=" * 60)
    
    example_code = '''
# --- Full Training Setup Example ---

from dataset import ProteinGODataset
from go_labeler import GOLabeler
from model import ProteinMLP, calculate_pos_weights, create_loss_function

# 1. Setup data (assuming labeler and annotations are ready)
train_dataset = ProteinGODataset(
    protein_ids=train_ids,
    labeler=labeler,
    embedding_dir="data/embeddings/train",
    annotations=annotations_dict
)

# 2. Calculate positive weights for imbalanced classes
pos_weights = calculate_pos_weights(train_dataset)

# 3. Create model
model = ProteinMLP(
    num_classes=labeler.vocabulary_size(),
    input_dim=2560,  # ESM-2 3B
    hidden_dim=1024,
    dropout=0.3
)

# 4. Create loss function with positive weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
criterion = create_loss_function(pos_weights, device=device)

# 5. Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for embeddings, labels in train_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# 6. Inference
model.eval()
with torch.no_grad():
    probs = model.predict_proba(test_embeddings)
    predictions = (probs > 0.5).float()
'''
    print(example_code)
