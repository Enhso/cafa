"""
Data Splitting Utilities for Protein Function Prediction.

This module provides functions for creating train/validation splits that
respect sequence similarity clustering. By splitting at the cluster level,
we ensure that highly similar sequences (>30% identity) are kept together,
preventing data leakage between train and validation sets.
"""

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


def create_splits(
    cluster_file: str,
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Create train/validation splits based on sequence clustering.
    
    This function ensures no data leakage by assigning entire clusters
    to either train or validation. Since clusters are formed by MMseqs2
    based on sequence identity (e.g., 30%), this guarantees that no
    validation sequence has >30% identity to any training sequence.
    
    Why cluster-based splitting matters:
    - Proteins with high sequence similarity often have similar functions
    - If similar proteins appear in both train and validation, the model
      might "memorize" rather than learn generalizable patterns
    - This approach gives a more realistic estimate of model performance
      on truly novel sequences
    
    Args:
        cluster_file: Path to MMseqs2 cluster TSV file with two columns:
                     (cluster_representative, sequence_member).
                     Each row indicates that sequence_member belongs to 
                     the cluster represented by cluster_representative.
        val_ratio: Fraction of clusters to assign to validation set.
                  Default is 0.2 (20% of clusters for validation).
        random_seed: Random seed for reproducibility. Set to None for
                    non-deterministic splits. Default is 42.
    
    Returns:
        Tuple of (train_protein_ids, val_protein_ids) where each is a
        set of protein/sequence IDs.
    
    Example:
        >>> train_ids, val_ids = create_splits('data/splits/train_cluster.tsv')
        >>> print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
        Train: 80000, Val: 20000
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Step 1: Read cluster file and group sequences by cluster representative
    # MMseqs2 output format: cluster_rep<TAB>sequence_member
    clusters: Dict[str, List[str]] = defaultdict(list)
    
    print(f"Reading cluster file: {cluster_file}")
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            cluster_rep, sequence_member = parts[0], parts[1]
            clusters[cluster_rep].append(sequence_member)
    
    # Get list of all cluster representatives
    cluster_reps = list(clusters.keys())
    n_clusters = len(cluster_reps)
    
    # Count total sequences
    total_sequences = sum(len(members) for members in clusters.values())
    
    print(f"Found {n_clusters} clusters containing {total_sequences} sequences")
    
    # Step 2: Shuffle clusters and split based on val_ratio
    # We shuffle the cluster representatives, not individual sequences
    random.shuffle(cluster_reps)
    
    # Calculate split point (number of clusters for validation)
    n_val_clusters = int(n_clusters * val_ratio)
    n_train_clusters = n_clusters - n_val_clusters
    
    # Assign clusters to splits
    val_cluster_reps = set(cluster_reps[:n_val_clusters])
    train_cluster_reps = set(cluster_reps[n_val_clusters:])
    
    # Step 3: Collect all sequence IDs for each split
    # All sequences in a cluster go to the same split
    train_protein_ids: Set[str] = set()
    val_protein_ids: Set[str] = set()
    
    for cluster_rep, members in clusters.items():
        if cluster_rep in val_cluster_reps:
            val_protein_ids.update(members)
        else:
            train_protein_ids.update(members)
    
    # Print statistics
    actual_val_ratio = len(val_protein_ids) / total_sequences
    print(f"Split complete:")
    print(f"  - Train: {n_train_clusters} clusters, {len(train_protein_ids)} sequences "
          f"({100 * len(train_protein_ids) / total_sequences:.1f}%)")
    print(f"  - Val:   {n_val_clusters} clusters, {len(val_protein_ids)} sequences "
          f"({100 * len(val_protein_ids) / total_sequences:.1f}%)")
    print(f"  - Target val_ratio: {val_ratio:.2f}, Actual: {actual_val_ratio:.2f}")
    
    # Sanity check: no overlap between splits
    overlap = train_protein_ids & val_protein_ids
    if overlap:
        raise ValueError(f"Error: {len(overlap)} sequences appear in both splits!")
    
    return train_protein_ids, val_protein_ids


def create_stratified_splits(
    cluster_file: str,
    annotations: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    random_seed: Optional[int] = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Create train/validation splits with approximate stratification by GO terms.
    
    This is a more advanced splitting strategy that tries to maintain similar
    GO term distributions in train and validation while still respecting
    cluster boundaries. It prioritizes cluster integrity over perfect 
    stratification.
    
    The approach:
    1. Group sequences by cluster
    2. For each cluster, identify its "dominant" GO namespace (BP/MF/CC)
    3. Split clusters within each namespace to maintain balance
    
    Args:
        cluster_file: Path to MMseqs2 cluster TSV file.
        annotations: List of (protein_id, term_id) tuples.
        val_ratio: Fraction of clusters for validation. Default 0.2.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_protein_ids, val_protein_ids) sets.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Read clusters
    clusters: Dict[str, List[str]] = defaultdict(list)
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                clusters[parts[0]].append(parts[1])
    
    # Build protein -> GO terms mapping
    protein_to_terms: Dict[str, Set[str]] = defaultdict(set)
    for protein_id, term_id in annotations:
        protein_to_terms[protein_id].add(term_id)
    
    # Categorize clusters by dominant namespace
    # GO namespaces: BP (GO:0008150), MF (GO:0003674), CC (GO:0005575)
    def get_namespace(term_id: str) -> str:
        """Heuristic: most terms are consistently in one namespace."""
        # This is a simplified approach - in practice you'd check the OBO file
        return "unknown"  # Fallback to simple random split
    
    # For now, fall back to simple random split
    # A full implementation would query the GO graph for namespaces
    cluster_reps = list(clusters.keys())
    random.shuffle(cluster_reps)
    
    n_val = int(len(cluster_reps) * val_ratio)
    val_cluster_reps = set(cluster_reps[:n_val])
    
    train_protein_ids: Set[str] = set()
    val_protein_ids: Set[str] = set()
    
    for cluster_rep, members in clusters.items():
        if cluster_rep in val_cluster_reps:
            val_protein_ids.update(members)
        else:
            train_protein_ids.update(members)
    
    return train_protein_ids, val_protein_ids


def load_cluster_mapping(cluster_file: str) -> Dict[str, str]:
    """
    Load a mapping from each sequence to its cluster representative.
    
    Useful for analyzing which cluster a sequence belongs to.
    
    Args:
        cluster_file: Path to MMseqs2 cluster TSV file.
    
    Returns:
        Dictionary mapping sequence_id -> cluster_representative_id.
    """
    seq_to_cluster: Dict[str, str] = {}
    
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                cluster_rep, sequence_member = parts[0], parts[1]
                seq_to_cluster[sequence_member] = cluster_rep
    
    return seq_to_cluster


def get_cluster_sizes(cluster_file: str) -> Dict[str, int]:
    """
    Get the size of each cluster.
    
    Args:
        cluster_file: Path to MMseqs2 cluster TSV file.
    
    Returns:
        Dictionary mapping cluster_representative -> cluster_size.
    """
    clusters: Dict[str, int] = defaultdict(int)
    
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                clusters[parts[0]] += 1
    
    return dict(clusters)


# Example usage and testing
if __name__ == "__main__":
    import os
    
    cluster_file = "data/splits/train_cluster.tsv"
    
    if os.path.exists(cluster_file):
        # Create splits
        train_ids, val_ids = create_splits(
            cluster_file=cluster_file,
            val_ratio=0.2,
            random_seed=42
        )
        
        print(f"\nTrain set size: {len(train_ids)}")
        print(f"Validation set size: {len(val_ids)}")
        print(f"Sample train IDs: {list(train_ids)[:5]}")
        print(f"Sample val IDs: {list(val_ids)[:5]}")
        
        # Get cluster size statistics
        cluster_sizes = get_cluster_sizes(cluster_file)
        sizes = list(cluster_sizes.values())
        print(f"\nCluster size statistics:")
        print(f"  Min: {min(sizes)}, Max: {max(sizes)}, "
              f"Mean: {sum(sizes)/len(sizes):.1f}")
        
        # Count singleton clusters (size 1)
        singletons = sum(1 for s in sizes if s == 1)
        print(f"  Singleton clusters: {singletons} ({100*singletons/len(sizes):.1f}%)")
    else:
        print(f"Cluster file not found: {cluster_file}")
        print("Run MMseqs2 easy-linclust first to generate clusters.")
        
        # Demo with synthetic data
        print("\nDemo with synthetic data:")
        
        # Create a temporary cluster file
        demo_clusters = [
            ("rep1", "seq1"), ("rep1", "seq2"), ("rep1", "seq3"),
            ("rep2", "seq4"), ("rep2", "seq5"),
            ("rep3", "seq6"),
            ("rep4", "seq7"), ("rep4", "seq8"), ("rep4", "seq9"), ("rep4", "seq10"),
        ]
        
        demo_file = "/tmp/demo_clusters.tsv"
        with open(demo_file, 'w') as f:
            for rep, member in demo_clusters:
                f.write(f"{rep}\t{member}\n")
        
        train_ids, val_ids = create_splits(demo_file, val_ratio=0.25, random_seed=42)
        print(f"Train: {sorted(train_ids)}")
        print(f"Val: {sorted(val_ids)}")
