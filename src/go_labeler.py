"""
GO Labeler: Handles Gene Ontology logic for protein function prediction.

This module provides the GOLabeler class which:
- Loads and parses the Gene Ontology (GO) graph from OBO format
- Propagates annotations to ancestor terms (True Path Rule)
- Builds a label vocabulary based on term frequencies
- Converts protein annotations to binary vectors
"""

import numpy as np
import obonet
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional


class GOLabeler:
    """
    A class to handle Gene Ontology term processing and label encoding.
    
    The Gene Ontology is a Directed Acyclic Graph (DAG) where edges point from
    child terms to parent terms (is_a relationships). The True Path Rule states
    that if a protein is annotated with a GO term, it is implicitly annotated
    with all ancestor terms as well.
    
    Attributes:
        obo_path (str): Path to the go-basic.obo file.
        annotations (List[Tuple[str, str]]): List of (protein_id, term_id) pairs.
        go_graph (nx.MultiDiGraph): The loaded GO graph from obonet.
        term_to_index (Dict[str, int]): Mapping from GO term ID to vocabulary index.
        index_to_term (Dict[int, str]): Mapping from vocabulary index to GO term ID.
        valid_terms (List[str]): List of GO terms meeting frequency threshold.
    """
    
    def __init__(self, obo_path: str, annotations: List[Tuple[str, str]]):
        """
        Initialize the GOLabeler.
        
        Args:
            obo_path: Path to the go-basic.obo file (e.g., 'data/raw/Train/go-basic.obo').
            annotations: List of (protein_id, term_id) tuples representing 
                        protein-to-GO-term annotations.
        """
        self.obo_path = obo_path
        self.annotations = annotations
        
        # Load the GO graph using obonet
        # The graph has edges from child -> parent (is_a relationships)
        print(f"Loading GO graph from {obo_path}...")
        self.go_graph = obonet.read_obo(obo_path)
        print(f"Loaded GO graph with {self.go_graph.number_of_nodes()} terms "
              f"and {self.go_graph.number_of_edges()} edges.")
        
        # These will be populated by build_label_vocabulary()
        self.term_to_index: Dict[str, int] = {}
        self.index_to_term: Dict[int, str] = {}
        self.valid_terms: List[str] = []
        self._term_frequencies: Dict[str, int] = {}
    
    def _get_ancestors(self, term_id: str) -> Set[str]:
        """
        Get all ancestor terms for a given GO term.
        
        The GO graph in obonet has edges pointing from child to parent
        (following is_a relationships). We use networkx's descendants()
        because in graph terms, following edges from a node leads to its
        "descendants" in the directed graph - but semantically these are
        the GO term's ancestors in the ontology hierarchy.
        
        True Path Rule: If a protein is annotated with term T, it is 
        implicitly annotated with all ancestors of T. For example, if a
        protein has "glucose binding" (GO:0005536), it also has the more
        general "carbohydrate binding" (GO:0030246) and ultimately 
        "molecular_function" (GO:0003674).
        
        Args:
            term_id: A GO term ID (e.g., 'GO:0005536').
            
        Returns:
            Set of all ancestor GO term IDs, including the term itself.
        """
        ancestors = set()
        
        # Check if the term exists in the graph
        if term_id not in self.go_graph:
            return ancestors
        
        # Include the term itself
        ancestors.add(term_id)
        
        # Get all ancestors by following edges in the graph
        # In obonet's GO graph, edges go from child -> parent,
        # so nx.descendants() gives us all reachable parent terms
        try:
            ancestors.update(nx.descendants(self.go_graph, term_id))
        except nx.NetworkXError:
            # Handle case where term might be isolated
            pass
        
        return ancestors
    
    def _propagate_terms(self, terms: List[str]) -> Set[str]:
        """
        Propagate a list of GO terms to include all their ancestors.
        
        This implements the True Path Rule: every annotation implies
        annotations to all more general (ancestor) terms.
        
        Args:
            terms: List of GO term IDs.
            
        Returns:
            Set of all GO terms including original terms and their ancestors.
        """
        all_terms = set()
        for term in terms:
            all_terms.update(self._get_ancestors(term))
        return all_terms
    
    def build_label_vocabulary(self, min_frequency: int = 50) -> None:
        """
        Build the label vocabulary from annotations with frequency filtering.
        
        This method:
        1. Groups annotations by protein to get each protein's direct terms
        2. Propagates each protein's terms to include all ancestors
        3. Counts how often each term appears across all proteins
        4. Filters to keep only terms appearing >= min_frequency times
        5. Creates term <-> index mappings for the vocabulary
        
        The frequency filtering is important because:
        - Rare terms have insufficient training examples
        - Very general terms (like root terms) will naturally have high frequency
        - This creates a balanced vocabulary for multi-label classification
        
        Args:
            min_frequency: Minimum number of proteins a term must annotate
                          to be included in the vocabulary. Default is 50.
        """
        print(f"Building label vocabulary with min_frequency={min_frequency}...")
        
        # Step 1: Group annotations by protein
        # This creates a dict: protein_id -> list of directly annotated terms
        protein_to_terms: Dict[str, List[str]] = defaultdict(list)
        for protein_id, term_id in self.annotations:
            protein_to_terms[protein_id].append(term_id)
        
        print(f"Found {len(protein_to_terms)} unique proteins in annotations.")
        
        # Step 2 & 3: Propagate terms and count frequencies
        # For each protein, propagate its terms to ancestors, then count
        term_counts: Dict[str, int] = defaultdict(int)
        
        for protein_id, terms in protein_to_terms.items():
            # Propagate to get all terms (direct + ancestors) for this protein
            propagated_terms = self._propagate_terms(terms)
            
            # Count each term (count once per protein, not per annotation)
            for term in propagated_terms:
                term_counts[term] += 1
        
        # Store frequencies for potential later use
        self._term_frequencies = dict(term_counts)
        
        # Step 4: Filter terms by minimum frequency
        self.valid_terms = sorted([
            term for term, count in term_counts.items()
            if count >= min_frequency
        ])
        
        print(f"Total unique terms after propagation: {len(term_counts)}")
        print(f"Terms meeting min_frequency threshold: {len(self.valid_terms)}")
        
        # Step 5: Create mappings
        # term_to_index: GO:0005575 -> 0, GO:0008150 -> 1, etc.
        # index_to_term: 0 -> GO:0005575, 1 -> GO:0008150, etc.
        self.term_to_index = {term: idx for idx, term in enumerate(self.valid_terms)}
        self.index_to_term = {idx: term for idx, term in enumerate(self.valid_terms)}
        
        # Print some statistics about the vocabulary
        if self.valid_terms:
            frequencies = [term_counts[t] for t in self.valid_terms]
            print(f"Vocabulary frequency stats - Min: {min(frequencies)}, "
                  f"Max: {max(frequencies)}, Mean: {np.mean(frequencies):.1f}")
    
    def get_vector(self, protein_terms: List[str]) -> np.ndarray:
        """
        Convert a protein's GO terms to a binary vector.
        
        This method:
        1. Takes the protein's directly annotated GO terms
        2. Propagates to include all ancestor terms (True Path Rule)
        3. Creates a binary vector where position i is 1 if the protein
           has the term at index i in our vocabulary
        
        Args:
            protein_terms: List of GO term IDs directly annotated to a protein.
            
        Returns:
            Binary numpy array of shape (len(valid_terms),) where:
            - 1 indicates the protein has that GO term (directly or via propagation)
            - 0 indicates the protein does not have that GO term
            
        Raises:
            ValueError: If vocabulary hasn't been built yet.
        """
        if not self.valid_terms:
            raise ValueError(
                "Vocabulary not built. Call build_label_vocabulary() first."
            )
        
        # Initialize binary vector with zeros
        vector = np.zeros(len(self.valid_terms), dtype=np.float32)
        
        # Propagate the protein's terms to include ancestors
        # This ensures we follow the True Path Rule
        propagated_terms = self._propagate_terms(protein_terms)
        
        # Set 1 for each term that's in our vocabulary
        for term in propagated_terms:
            if term in self.term_to_index:
                idx = self.term_to_index[term]
                vector[idx] = 1.0
        
        return vector
    
    def get_term_frequency(self, term_id: str) -> int:
        """
        Get the frequency count for a specific GO term.
        
        Args:
            term_id: A GO term ID.
            
        Returns:
            Number of proteins annotated with this term (after propagation),
            or 0 if the term was not seen.
        """
        return self._term_frequencies.get(term_id, 0)
    
    def vocabulary_size(self) -> int:
        """Return the size of the label vocabulary."""
        return len(self.valid_terms)
    
    def __repr__(self) -> str:
        return (f"GOLabeler(obo_path='{self.obo_path}', "
                f"n_annotations={len(self.annotations)}, "
                f"vocabulary_size={self.vocabulary_size()})")


# Example usage and testing
if __name__ == "__main__":
    # Example: Load annotations from a file (adjust path as needed)
    # In practice, you'd load these from train_terms.tsv
    
    # Sample annotations for demonstration
    sample_annotations = [
        ("protein_1", "GO:0005524"),  # ATP binding
        ("protein_1", "GO:0006468"),  # protein phosphorylation
        ("protein_2", "GO:0005524"),  # ATP binding
        ("protein_2", "GO:0004672"),  # protein kinase activity
        ("protein_3", "GO:0003677"),  # DNA binding
    ]
    
    # Initialize the labeler
    labeler = GOLabeler(
        obo_path="data/raw/Train/go-basic.obo",
        annotations=sample_annotations
    )
    
    # Build vocabulary with a low threshold for this small example
    labeler.build_label_vocabulary(min_frequency=1)
    
    print(f"\nVocabulary size: {labeler.vocabulary_size()}")
    print(f"First 10 terms: {labeler.valid_terms[:10]}")
    
    # Get vector for a protein
    test_terms = ["GO:0005524", "GO:0006468"]
    vector = labeler.get_vector(test_terms)
    print(f"\nVector for {test_terms}:")
    print(f"Shape: {vector.shape}")
    print(f"Non-zero positions: {np.where(vector == 1)[0]}")
    print(f"Number of active terms: {int(vector.sum())}")
