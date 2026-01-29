#!/usr/bin/env python3
"""
Physiological feature computation and k-mer vectorizer utilities.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def compute_physio_properties(sequence: str) -> np.ndarray:
    """
    Compute normalized physiological properties for a protein sequence.

    Returns normalized values for:
      - Isoelectric Point (0..1 scaled by 14)
      - Aromaticity (0..1)
      - Helix Fraction (0..1)
      - Sheet Fraction (0..1)
      - Molecular Weight (log10-scaled, normalized roughly to 0..1)

    Args:
        sequence: Amino acid sequence.

    Returns:
        np.ndarray of shape (5,) with dtype float32.
    """
    seq = sequence.strip().upper()
    if not seq:
        return np.zeros(5, dtype=np.float32)

    analysis = ProteinAnalysis(seq)

    try:
        pI = analysis.isoelectric_point()
    except Exception:
        pI = 7.0

    aromaticity = float(analysis.aromaticity())
    helix, turn, sheet = analysis.secondary_structure_fraction()
    mw = float(analysis.molecular_weight())

    # Normalizations
    pI_norm = max(0.0, min(1.0, pI / 14.0))
    aromaticity_norm = max(0.0, min(1.0, aromaticity))
    helix_norm = max(0.0, min(1.0, helix))
    sheet_norm = max(0.0, min(1.0, sheet))

    # Log-scaled MW normalization: proteins typically range ~1e3 to 1e6 Da
    log_mw = math.log10(max(mw, 1.0))
    mw_norm = (log_mw - 3.0) / (6.0 - 3.0)
    mw_norm = max(0.0, min(1.0, mw_norm))

    return np.array(
        [pI_norm, aromaticity_norm, helix_norm, sheet_norm, mw_norm],
        dtype=np.float32,
    )


@dataclass
class KmerVectorizer:
    """
    3-mer TF-IDF vectorizer with TruncatedSVD dimensionality reduction.

    Workflow:
      - fit(sequences): build vocabulary, compute IDF, fit TruncatedSVD
      - transform(sequences): TF-IDF -> SVD reduced vectors

    Save/Load:
      - save(path)
      - load(path)
    """

    k: int = 3
    n_components: int = 500
    random_state: int = 0

    def __post_init__(self) -> None:
        self.vocab_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.svd_: Optional[TruncatedSVD] = None
        self._fitted: bool = False

    def fit(self, sequences: Sequence[str]) -> "KmerVectorizer":
        """
        Fit vocabulary, IDF, and SVD components on sequences.
        """
        vocab, df_counts = self._build_vocab(sequences)
        self.vocab_ = vocab

        n_docs = len(sequences)
        idf = np.log((1.0 + n_docs) / (1.0 + df_counts)) + 1.0
        self.idf_ = idf.astype(np.float32)

        tf = self._build_tf_matrix(sequences, self.vocab_)
        tfidf = tf.multiply(self.idf_)

        n_features = tfidf.shape[1]
        if n_features >= 2:
            n_components = min(self.n_components, n_features - 1)
            self.svd_ = TruncatedSVD(
                n_components=n_components,
                random_state=self.random_state,
            ).fit(tfidf)
        else:
            self.svd_ = None

        self._fitted = True
        return self

    def transform(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Transform sequences into reduced TF-IDF vectors.

        Returns:
            np.ndarray of shape (n_samples, n_components or n_features)
        """
        if not self._fitted or self.idf_ is None:
            raise RuntimeError("KmerVectorizer is not fitted. Call fit() first.")

        tf = self._build_tf_matrix(sequences, self.vocab_)
        tfidf = tf.multiply(self.idf_)

        if self.svd_ is None:
            return tfidf.toarray().astype(np.float32)

        return self.svd_.transform(tfidf).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """
        Save vectorizer state to disk.
        """
        path = Path(path)
        state = {
            "k": self.k,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "vocab_": self.vocab_,
            "idf_": self.idf_,
            "svd_": self.svd_,
            "_fitted": self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "KmerVectorizer":
        """
        Load vectorizer state from disk.
        """
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        obj = cls(
            k=state["k"],
            n_components=state["n_components"],
            random_state=state["random_state"],
        )
        obj.vocab_ = state["vocab_"]
        obj.idf_ = state["idf_"]
        obj.svd_ = state["svd_"]
        obj._fitted = state["_fitted"]
        return obj

    def _build_vocab(
        self, sequences: Sequence[str]
    ) -> tuple[Dict[str, int], np.ndarray]:
        """
        Build k-mer vocabulary and document frequency counts.
        """
        df_counts: Dict[str, int] = {}
        for seq in sequences:
            seq = str(seq).strip().upper()
            seen = set()
            for kmer in self._iter_kmers(seq):
                if kmer in seen:
                    continue
                seen.add(kmer)
                df_counts[kmer] = df_counts.get(kmer, 0) + 1

        vocab = {kmer: i for i, kmer in enumerate(sorted(df_counts.keys()))}
        df_array = np.zeros(len(vocab), dtype=np.float32)
        for kmer, idx in vocab.items():
            df_array[idx] = float(df_counts[kmer])

        return vocab, df_array

    def _build_tf_matrix(
        self, sequences: Sequence[str], vocab: Dict[str, int]
    ) -> sparse.csr_matrix:
        """
        Build sparse term-frequency matrix.
        """
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []

        for row_idx, seq in enumerate(sequences):
            seq = str(seq).strip().upper()
            counts: Dict[int, int] = {}
            for kmer in self._iter_kmers(seq):
                idx = vocab.get(kmer)
                if idx is None:
                    continue
                counts[idx] = counts.get(idx, 0) + 1
            for idx, cnt in counts.items():
                rows.append(row_idx)
                cols.append(idx)
                data.append(cnt)

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(sequences), len(vocab)),
            dtype=np.float32,
        )
        return mat

    def _iter_kmers(self, sequence: str) -> Iterable[str]:
        if len(sequence) < self.k:
            return []
        return (sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1))
