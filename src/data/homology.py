"""
Homology residual utilities for MMseqs2 hits.

This module builds a per-query sparse GO-term score vector using MMseqs2
bit-scores aggregated across homologous targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

import pandas as pd
import torch


def load_target_go_map(
    metadata_path: Path, filter_aspects: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """
    Loads a metadata table and builds a {protein_id: {go_terms}} lookup.

    Args:
        metadata_path: Path to CSV/TSV with columns 'protein_id', 'go_term'.
        filter_aspects: Optional set of aspects to keep (e.g., {'BP', 'MF'}).

    Returns:
        Dictionary mapping protein_id -> set of GO term strings.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Auto-detect separator
    try:
        df = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
    except Exception as e:
        raise ValueError(f"Could not parse metadata file: {e}")

    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]

    # Map columns
    col_map = {}
    for c in df.columns:
        if "protein" in c or "entry" in c:
            col_map["id"] = c
        elif "term" in c or "go_id" in c:
            col_map["term"] = c
        elif "aspect" in c:
            col_map["aspect"] = c

    if "id" not in col_map or "term" not in col_map:
        raise ValueError(
            f"Required columns (protein_id, go_term) not found. Found: {df.columns}"
        )

    # Filter aspect
    if filter_aspects and "aspect" in col_map:
        df = df[df[col_map["aspect"]].isin(filter_aspects)]

    # Group by ID
    grouped = df.groupby(col_map["id"])[col_map["term"]].apply(set).to_dict()
    return grouped


def build_homology_matrix(
    query_ids: Iterable[str],
    mmseqs_file: str | Path,
    target_go_map: Dict[str, Set[str]],
    label_map: Mapping[str, int],
) -> Dict[str, torch.Tensor]:
    """
    Build homology residual vectors for query proteins.

    Args:
        query_ids: Iterable of query protein IDs to include.
        mmseqs_file: Path to MMseqs2 .m8 file.
        target_go_map: Lookup dict {target_id: {go_terms}}.
                       Ensure these terms are propagated if needed!
        label_map: Mapping GO term -> index (0..N-1).

    Returns:
        Dictionary mapping protein_id -> sparse torch tensor of shape (N,).
    """
    mmseqs_path = Path(mmseqs_file)
    if not mmseqs_path.exists():
        raise FileNotFoundError(f"MMseqs2 file not found: {mmseqs_path}")

    query_set = set(str(q) for q in query_ids)
    num_labels = len(label_map)

    # Accumulator: query -> go_term -> total_bit_score
    acc: Dict[str, Dict[str, float]] = {qid: {} for qid in query_set}

    with mmseqs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            query, target = parts[0], parts[1]

            # Robust bit-score extraction
            # Standard m8 (12 cols) -> score is at index 11
            # Custom 3-col output -> score is at index 2
            bit_score_str = parts[11] if len(parts) >= 12 else parts[-1]

            if query not in query_set:
                continue

            try:
                bit_score = float(bit_score_str)
            except ValueError:
                continue

            # Lookup target terms
            target_terms = target_go_map.get(target)
            if not target_terms:
                continue

            for term in target_terms:
                if term in label_map:
                    acc[query][term] = acc[query].get(term, 0.0) + bit_score

    # Convert to Sparse Tensors
    output: Dict[str, torch.Tensor] = {}

    for qid in query_set:
        term_scores = acc.get(qid, {})

        if not term_scores:
            output[qid] = torch.sparse_coo_tensor(
                size=(num_labels,), dtype=torch.float32
            )
            continue

        # Min-Max Normalization
        values = list(term_scores.values())
        v_min, v_max = min(values), max(values)
        denom = v_max - v_min

        if denom > 1e-9:
            norm_scores = {k: (v - v_min) / denom for k, v in term_scores.items()}
        else:
            norm_scores = {k: 1.0 for k in term_scores}

        indices = []
        vals = []
        for term, score in norm_scores.items():
            idx = label_map[term]
            indices.append([idx])
            vals.append(score)

        idx_tensor = torch.tensor(indices, dtype=torch.long).t()
        val_tensor = torch.tensor(vals, dtype=torch.float32)

        output[qid] = torch.sparse_coo_tensor(
            idx_tensor, val_tensor, size=(num_labels,), dtype=torch.float32
        )

    return output
