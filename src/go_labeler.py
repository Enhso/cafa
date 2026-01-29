"""
GO Labeler: Handles Gene Ontology logic for protein function prediction.
Updated to support public term propagation and label map serialization.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import obonet


class GOLabeler:
    """
    A class to handle Gene Ontology term processing and label encoding.
    """

    def __init__(
        self, obo_path: str, annotations: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize the GOLabeler.

        Args:
            obo_path: Path to the go-basic.obo file.
            annotations: List of (protein_id, term_id) tuples. Optional if loading
                         an existing label map.
        """
        self.obo_path = obo_path
        self.annotations = annotations if annotations else []

        print(f"Loading GO graph from {obo_path}...")
        self.go_graph = obonet.read_obo(obo_path)
        print(f"Loaded GO graph with {self.go_graph.number_of_nodes()} terms.")

        self.term_to_index: Dict[str, int] = {}
        self.index_to_term: Dict[int, str] = {}
        self.valid_terms: List[str] = []
        self._term_frequencies: Dict[str, int] = {}

    def get_ancestors(self, term_id: str) -> Set[str]:
        """
        Get all ancestor terms for a given GO term (Public).
        """
        ancestors = set()
        if term_id not in self.go_graph:
            return ancestors

        ancestors.add(term_id)
        try:
            # In obonet, edges go Child -> Parent, so descendants are ancestors
            ancestors.update(nx.descendants(self.go_graph, term_id))
        except nx.NetworkXError:
            pass
        return ancestors

    def propagate_terms(self, terms: List[str]) -> Set[str]:
        """
        Propagate a list of GO terms to include all their ancestors (Public).
        """
        all_terms = set()
        for term in terms:
            all_terms.update(self.get_ancestors(term))
        return all_terms

    def build_label_vocabulary(self, min_frequency: int = 50) -> None:
        """Build the label vocabulary from annotations with frequency filtering."""
        if not self.annotations:
            raise ValueError("No annotations provided to build vocabulary.")

        print(f"Building label vocabulary with min_frequency={min_frequency}...")

        # 1. Group by protein
        protein_to_terms: Dict[str, List[str]] = defaultdict(list)
        for protein_id, term_id in self.annotations:
            protein_to_terms[protein_id].append(term_id)

        # 2. Propagate and Count
        term_counts: Dict[str, int] = defaultdict(int)
        for protein_id, terms in protein_to_terms.items():
            propagated_terms = self.propagate_terms(terms)
            for term in propagated_terms:
                term_counts[term] += 1

        self._term_frequencies = dict(term_counts)

        # 3. Filter
        self.valid_terms = sorted(
            [t for t, c in term_counts.items() if c >= min_frequency]
        )

        # 4. Create Mappings
        self.term_to_index = {t: i for i, t in enumerate(self.valid_terms)}
        self.index_to_term = {i: t for i, t in enumerate(self.valid_terms)}

        print(f"Vocabulary built. Size: {len(self.valid_terms)}")

    def save_label_map(self, path: str) -> None:
        """Save the term_to_index mapping to a JSON file (Deployment)."""
        with open(path, "w") as f:
            json.dump(self.term_to_index, f, indent=2)
        print(f"Saved label map to {path}")

    def load_label_map(self, path: str) -> None:
        """Load a term_to_index mapping from a JSON file (Inference)."""
        with open(path, "r") as f:
            self.term_to_index = json.load(f)

        # Reconstruct reverse mapping and valid_terms list
        self.index_to_term = {v: k for k, v in self.term_to_index.items()}
        # Sort by index to ensure order matches
        sorted_pairs = sorted(self.index_to_term.items())
        self.valid_terms = [k for _, k in sorted_pairs]

        print(f"Loaded label map with {len(self.valid_terms)} terms from {path}")

    def get_vector(self, protein_terms: List[str]) -> np.ndarray:
        """Convert protein terms to a binary vector using the vocabulary."""
        if not self.valid_terms:
            raise ValueError("Vocabulary not built or loaded.")

        vector = np.zeros(len(self.valid_terms), dtype=np.float32)
        propagated = self.propagate_terms(protein_terms)

        for term in propagated:
            if term in self.term_to_index:
                vector[self.term_to_index[term]] = 1.0
        return vector

    def vocabulary_size(self) -> int:
        return len(self.valid_terms)
