#!/usr/bin/env python3
"""
PyTorch Dataset for Protein Function Prediction (Phase 2).

This module defines:
- Vocabulary: amino-acid to integer encoding (20 AAs + PAD + UNK)
- ProteinDataset: loads ESM-2 embeddings, encodes sequences, and builds multi-hot labels
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from src.features.stats import KmerVectorizer, compute_physio_properties
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Vocabulary:
    """
    Amino acid vocabulary with PAD and UNK tokens.

    Token indices:
    - PAD: 0
    - UNK: 1
    - Standard 20 amino acids: 2..21
    """

    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_aa_tokens", list("ACDEFGHIKLMNPQRSTVWY"))
        token_to_idx = {self.pad_token: 0, self.unk_token: 1}
        for i, aa in enumerate(self._aa_tokens, start=2):
            token_to_idx[aa] = i
        object.__setattr__(self, "_token_to_idx", token_to_idx)

    @property
    def pad_index(self) -> int:
        return 0

    @property
    def unk_index(self) -> int:
        return 1

    def encode(self, sequence: str, max_len: int) -> torch.Tensor:
        """
        Encode a sequence to a padded/truncated integer tensor.

        Args:
            sequence: Protein sequence string.
            max_len: Maximum length for padding/truncation.

        Returns:
            Tensor of shape (max_len,) with dtype torch.long.
        """
        seq = sequence.strip().upper()
        indices: List[int] = []
        for ch in seq:
            indices.append(self._token_to_idx.get(ch, self.unk_index))

        if len(indices) >= max_len:
            indices = indices[:max_len]
        else:
            indices.extend([self.pad_index] * (max_len - len(indices)))

        return torch.tensor(indices, dtype=torch.long)


class ProteinDataset(Dataset):
    """
    Dataset that loads embeddings from disk, encodes sequences, and returns multi-hot labels.

    Each item returns a dictionary with:
        protein_id (str)
        esm_embedding (torch.Tensor)
        sequence_encoding (torch.Tensor)
        labels (torch.Tensor)
        stat_features (torch.Tensor)
        homology_vector (torch.Tensor)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        embeddings_dir: str | Path,
        label_map: Mapping[str, int],
        max_len: int = 1022,
        kmer_vectorizer: KmerVectorizer | None = None,
        homology_dict: Optional[Mapping[str, torch.Tensor]] = None,
        densify_homology: bool = False,
    ) -> None:
        """
        Args:
            dataframe: DataFrame with columns protein_id, sequence, go_terms (semicolon-separated).
            embeddings_dir: Path to folder containing {source}|{protein_id}|{gene_name}.pt files.
            label_map: Mapping from GO ID to integer index [0..N-1].
            max_len: Max sequence length for padding/truncation (default: 1022).
            kmer_vectorizer: Pre-fitted KmerVectorizer for k-mer TF-IDF features.
            homology_dict: Optional mapping protein_id -> sparse homology vector.
            densify_homology: If True, return dense homology vectors.
        """
        required_cols = {"protein_id", "sequence", "go_terms"}
        missing = required_cols - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in dataframe: {sorted(missing)}"
            )

        self.df = dataframe.reset_index(drop=True)
        self.embeddings_dir = Path(embeddings_dir)
        self.label_map = dict(label_map)
        self.max_len = max_len
        self.vocab = Vocabulary()
        self.kmer_vectorizer = kmer_vectorizer
        self.homology_dict = dict(homology_dict) if homology_dict is not None else {}
        self.densify_homology = densify_homology

        if not self.embeddings_dir.exists():
            raise FileNotFoundError(
                f"Embeddings directory not found: {self.embeddings_dir}"
            )

        self.embedding_index = self._build_embedding_index(self.embeddings_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        protein_id = str(row["protein_id"])
        sequence = str(row["sequence"])
        go_terms = str(row["go_terms"])

        embedding_path = self.embedding_index.get(protein_id)
        if embedding_path is None:
            raise FileNotFoundError(
                f"Embedding file not found for protein_id '{protein_id}'"
            )

        esm_embedding = torch.load(embedding_path, map_location="cpu")

        sequence_encoding = self.vocab.encode(sequence, self.max_len)

        labels = self._build_multihot(go_terms)

        physio = compute_physio_properties(sequence)
        if self.kmer_vectorizer is None:
            raise ValueError("kmer_vectorizer is required to compute stat_features.")
        kmer_vec = self.kmer_vectorizer.transform([sequence])[0]
        stat_features = torch.from_numpy(
            np.concatenate([physio, kmer_vec]).astype(np.float32)
        )

        homology_vec = self.homology_dict.get(protein_id)
        if homology_vec is None:
            if self.densify_homology:
                homology_vec = torch.zeros(len(self.label_map), dtype=torch.float32)
            else:
                homology_vec = torch.sparse_coo_tensor(
                    size=(len(self.label_map),), dtype=torch.float32
                )
        elif self.densify_homology and homology_vec.is_sparse:
            homology_vec = homology_vec.to_dense()

        return {
            "protein_id": protein_id,
            "esm_embedding": esm_embedding,
            "sequence_encoding": sequence_encoding,
            "labels": labels,
            "stat_features": stat_features,
            "homology_vector": homology_vec,
        }

    def _build_multihot(self, go_terms: str) -> torch.Tensor:
        """
        Build a multi-hot label vector from semicolon-separated GO terms.
        """
        num_labels = len(self.label_map)
        vector = torch.zeros(num_labels, dtype=torch.float32)

        if go_terms:
            for term in go_terms.split(";"):
                term = term.strip()
                if not term:
                    continue
                idx = self.label_map.get(term)
                if idx is not None:
                    vector[idx] = 1.0

        return vector

    @staticmethod
    def _build_embedding_index(embeddings_dir: Path) -> Dict[str, Path]:
        """
        Map protein_id -> embedding file path.

        Expected filename: {source}|{protein_id}|{gene_name}.pt
        """
        mapping: Dict[str, Path] = {}
        for path in embeddings_dir.glob("*.pt"):
            name = path.name
            if "|" in name:
                parts = name.split("|")
                if len(parts) >= 2:
                    protein_id = parts[1].strip()
                else:
                    protein_id = path.stem
            else:
                protein_id = path.stem

            if protein_id and protein_id not in mapping:
                mapping[protein_id] = path

        return mapping
