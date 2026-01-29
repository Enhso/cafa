#!/usr/bin/env python3
"""
Preprocess raw data into a training metadata CSV.

Inputs:
- labels.tsv: TSV with columns protein_id, go_term, aspect (positive labels only)
- sequences.fasta: FASTA with headers matching protein_id
- embeddings_dir: directory containing {source}|{protein_id}|{gene_name}.pt files

Output:
- data/metadata.csv with columns: protein_id, sequence, go_terms
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from Bio import SeqIO

DEFAULT_LABELS = "labels.tsv"
DEFAULT_FASTA = "sequences.fasta"
DEFAULT_EMB_DIR = "embeddings_dir"
DEFAULT_OUTPUT = "data/metadata.csv"


def normalize_id(id_str: str) -> str:
    """Standardize IDs across all sources."""
    if "|" in id_str:
        parts = id_str.split("|")
        if len(parts) >= 2:
            return parts[1].strip()
    return id_str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata CSV from labels, sequences, and embeddings."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=DEFAULT_LABELS,
        help="Path to labels TSV (protein_id, go_term, aspect).",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=DEFAULT_FASTA,
        help="Path to sequences FASTA.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=DEFAULT_EMB_DIR,
        help="Path to embeddings directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to output metadata CSV.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path, sep="\t", dtype=str)
    if not {"protein_id", "go_term"}.issubset(df.columns):
        raise ValueError(
            "labels.tsv must contain at least 'protein_id' and 'go_term' columns."
        )
    df = df[["protein_id", "go_term"]].dropna()
    df["protein_id"] = df["protein_id"].str.strip()
    df["go_term"] = df["go_term"].str.strip()
    df = df[(df["protein_id"] != "") & (df["go_term"] != "")]
    grouped = (
        df.groupby("protein_id")["go_term"]
        .apply(lambda terms: sorted(set(terms)))
        .reset_index()
    )
    grouped["go_terms"] = grouped["go_term"].apply(lambda terms: ";".join(terms))
    return grouped[["protein_id", "go_terms"]]


def load_sequences(fasta_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        clean_id = normalize_id(record.id)
        sequences[clean_id] = str(record.seq)
    return sequences


def load_embedding_ids(emb_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    for path in emb_dir.glob("*.pt"):
        name = path.name
        if "|" in name:
            parts = name.split("|")
            if len(parts) >= 2:
                protein_id = parts[1].strip()
            else:
                protein_id = path.stem
        else:
            protein_id = path.stem
        if protein_id:
            ids.add(protein_id)
    return ids


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels)
    fasta_path = Path(args.fasta)
    emb_dir = Path(args.embeddings_dir)
    output_path = Path(args.output)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels TSV not found: {labels_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    labels_df = load_labels(labels_path)
    sequences = load_sequences(fasta_path)
    embedding_ids = load_embedding_ids(emb_dir)

    label_ids = set(labels_df["protein_id"])
    fasta_ids = set(sequences.keys())

    intersect_ids = label_ids & fasta_ids & embedding_ids

    dropped_total = len(label_ids) - len(intersect_ids)
    print(
        f"Dropped {dropped_total} proteins missing sequence or embedding "
        f"(kept {len(intersect_ids)})."
    )

    filtered = labels_df[labels_df["protein_id"].isin(intersect_ids)].copy()
    filtered["sequence"] = filtered["protein_id"].map(sequences)

    filtered = filtered[["protein_id", "sequence", "go_terms"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"Wrote metadata CSV: {output_path}")


if __name__ == "__main__":
    main()
