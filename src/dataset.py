"""
PyTorch Dataset for Protein Function Prediction.

This module provides a Dataset class that loads pre-computed protein embeddings
and their corresponding GO term labels for training neural network models.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import numpy as np

# Import our GOLabeler (assumes it's in the same package)
# from .go_labeler import GOLabeler


class ProteinGODataset(Dataset):
    """
    PyTorch Dataset for protein function prediction.
    
    This dataset loads pre-computed protein embeddings (e.g., from ESM-2)
    and converts GO term annotations into multi-hot binary label vectors
    using the GOLabeler class.
    
    Attributes:
        protein_ids (List[str]): List of protein IDs in this dataset split.
        labeler: GOLabeler instance for converting terms to vectors.
        embedding_dir (Path): Directory containing .pt embedding files.
        annotations (Dict[str, List[str]]): Mapping protein_id -> GO terms.
        valid_protein_ids (List[str]): Protein IDs with available embeddings.
    """
    
    def __init__(
        self,
        protein_ids: List[str],
        labeler,  # GOLabeler instance
        embedding_dir: str,
        annotations: Dict[str, List[str]],
        check_exists: bool = True
    ):
        """
        Initialize the ProteinGODataset.
        
        Args:
            protein_ids: List of protein IDs for this split (train or val).
            labeler: An instance of the GOLabeler class with vocabulary built.
            embedding_dir: Path to folder containing {protein_id}.pt files.
            annotations: Dictionary mapping protein_id -> list of GO term IDs.
            check_exists: If True, filter out proteins without embedding files.
                         Set to False if you're sure all files exist (faster).
        """
        self.labeler = labeler
        self.embedding_dir = Path(embedding_dir)
        self.annotations = annotations
        
        # Optionally filter to only include proteins with existing embeddings
        if check_exists:
            self.protein_ids = []
            missing_count = 0
            for pid in protein_ids:
                embedding_path = self.embedding_dir / f"{pid}.pt"
                if embedding_path.exists():
                    self.protein_ids.append(pid)
                else:
                    missing_count += 1
            
            if missing_count > 0:
                print(f"Warning: {missing_count} proteins missing embeddings, "
                      f"using {len(self.protein_ids)} proteins")
        else:
            self.protein_ids = list(protein_ids)
        
        # Validate that labeler has vocabulary built
        if labeler.vocabulary_size() == 0:
            raise ValueError(
                "GOLabeler vocabulary is empty. "
                "Call labeler.build_label_vocabulary() first."
            )
        
        print(f"ProteinGODataset initialized:")
        print(f"  - Proteins: {len(self.protein_ids)}")
        print(f"  - Label vocabulary size: {labeler.vocabulary_size()}")
        print(f"  - Embedding directory: {self.embedding_dir}")
    
    def __len__(self) -> int:
        """Return the number of proteins in the dataset."""
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single protein sample.
        
        Args:
            idx: Index of the protein in the dataset.
        
        Returns:
            Tuple of (embedding, label) where:
            - embedding: torch.Tensor of shape (embedding_dim,), dtype float32
            - label: torch.Tensor of shape (vocab_size,), dtype float32
                    Multi-hot binary vector (1.0 for present terms, 0.0 otherwise)
        """
        protein_id = self.protein_ids[idx]
        
        # Step 1: Load the pre-computed embedding from disk
        embedding_path = self.embedding_dir / f"{protein_id}.pt"
        embedding = torch.load(embedding_path, weights_only=True)
        
        # Ensure embedding is float32
        embedding = embedding.to(torch.float32)
        
        # Step 2: Get the explicit GO terms for this protein
        # If protein has no annotations, use empty list
        explicit_terms = self.annotations.get(protein_id, [])
        
        # Step 3: Use GOLabeler to generate multi-hot binary label
        # This handles ancestor propagation internally
        label_numpy = self.labeler.get_vector(explicit_terms)
        
        # Convert to torch tensor with float32 dtype
        label = torch.from_numpy(label_numpy).to(torch.float32)
        
        return embedding, label
    
    def get_protein_id(self, idx: int) -> str:
        """Get the protein ID at a given index."""
        return self.protein_ids[idx]
    
    def get_label_counts(self) -> np.ndarray:
        """
        Count label frequencies across the dataset.
        
        Useful for computing class weights for imbalanced learning.
        
        Returns:
            Array of shape (vocab_size,) with count for each label.
        """
        counts = np.zeros(self.labeler.vocabulary_size(), dtype=np.int64)
        
        for protein_id in self.protein_ids:
            terms = self.annotations.get(protein_id, [])
            label_vector = self.labeler.get_vector(terms)
            counts += label_vector.astype(np.int64)
        
        return counts
    
    def get_positive_weight(self) -> torch.Tensor:
        """
        Compute positive class weights for BCEWithLogitsLoss.
        
        For imbalanced multi-label classification, we can weight the
        positive class to account for the imbalance. Weight is computed as:
        pos_weight[i] = (num_negative[i]) / (num_positive[i])
        
        Returns:
            Tensor of shape (vocab_size,) with positive weights.
        """
        counts = self.get_label_counts()
        n_samples = len(self.protein_ids)
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        
        # pos_weight = num_negative / num_positive
        pos_weight = (n_samples - counts) / counts
        
        return torch.from_numpy(pos_weight).to(torch.float32)


class ProteinGODatasetInMemory(Dataset):
    """
    In-memory version of ProteinGODataset for faster training.
    
    Loads all embeddings into memory at initialization. Use this when
    you have sufficient RAM and want to avoid disk I/O during training.
    """
    
    def __init__(
        self,
        protein_ids: List[str],
        labeler,
        embedding_dir: str,
        annotations: Dict[str, List[str]]
    ):
        """Initialize and load all embeddings into memory."""
        self.labeler = labeler
        self.embedding_dir = Path(embedding_dir)
        self.annotations = annotations
        
        # Load all embeddings into memory
        self.embeddings = {}
        self.protein_ids = []
        
        print("Loading embeddings into memory...")
        for pid in protein_ids:
            embedding_path = self.embedding_dir / f"{pid}.pt"
            if embedding_path.exists():
                self.embeddings[pid] = torch.load(
                    embedding_path, weights_only=True
                ).to(torch.float32)
                self.protein_ids.append(pid)
        
        print(f"Loaded {len(self.protein_ids)} embeddings into memory")
        
        # Pre-compute labels
        self.labels = {}
        for pid in self.protein_ids:
            terms = self.annotations.get(pid, [])
            label = torch.from_numpy(
                self.labeler.get_vector(terms)
            ).to(torch.float32)
            self.labels[pid] = label
        
        print(f"Pre-computed {len(self.labels)} label vectors")
    
    def __len__(self) -> int:
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        protein_id = self.protein_ids[idx]
        return self.embeddings[protein_id], self.labels[protein_id]


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from go_labeler import GOLabeler
    
    # Example: Create a small test dataset
    print("=" * 50)
    print("ProteinGODataset Demo")
    print("=" * 50)
    
    # Sample annotations (protein_id -> list of GO terms)
    sample_annotations_list = [
        ("P12345", "GO:0005524"),  # ATP binding
        ("P12345", "GO:0006468"),  # protein phosphorylation
        ("P67890", "GO:0005524"),  # ATP binding
        ("P67890", "GO:0004672"),  # protein kinase activity
    ]
    
    # Convert to dict format
    from collections import defaultdict
    annotations_dict: Dict[str, List[str]] = defaultdict(list)
    for pid, term in sample_annotations_list:
        annotations_dict[pid].append(term)
    annotations_dict = dict(annotations_dict)
    
    print(f"Sample annotations: {annotations_dict}")
    
    # Initialize GOLabeler
    obo_path = "data/raw/Train/go-basic.obo"
    
    if os.path.exists(obo_path):
        labeler = GOLabeler(
            obo_path=obo_path,
            annotations=sample_annotations_list
        )
        labeler.build_label_vocabulary(min_frequency=1)
        
        # Check for embeddings
        embedding_dir = "data/embeddings/train"
        protein_ids = list(annotations_dict.keys())
        
        if os.path.exists(embedding_dir):
            # Create dataset
            dataset = ProteinGODataset(
                protein_ids=protein_ids,
                labeler=labeler,
                embedding_dir=embedding_dir,
                annotations=annotations_dict
            )
            
            if len(dataset) > 0:
                # Get a sample
                embedding, label = dataset[0]
                print(f"\nSample output:")
                print(f"  Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
                print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
                print(f"  Active labels: {int(label.sum())}")
        else:
            print(f"\nEmbedding directory not found: {embedding_dir}")
            print("Run extract_embeddings.py first to generate embeddings.")
    else:
        print(f"OBO file not found: {obo_path}")
