#!/usr/bin/env python3
"""
Create a small, consistent debug dataset from large FASTA, TSV, and embeddings.

Outputs:
- data/debug/sequences.fasta
- data/debug/labels.tsv
- data/debug/embeddings/*.pt

Requirements:
- Uses Bio.SeqIO for FASTA parsing
- Uses argparse
- Uses shutil for copying embeddings
- Uses tqdm for progress bar while copying embeddings
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from Bio import SeqIO
from tqdm import tqdm

# Updated to use the correct training labels by default
DEFAULT_FASTA = "data/raw/Train/train_sequences.fasta"
DEFAULT_LABELS = "data/raw/Train/train_terms.tsv"
DEFAULT_EMB_DIR = "data/embeddings"
DEFAULT_OUT_DIR = "data/debug"
DEFAULT_SAMPLE_SIZE = 2000


def normalize_id(id_str: str) -> str:
    """
    Extracts core ID (e.g., P12345) from formats like 'sp|P12345|NAME'.
    Ensures consistency across FASTA, TSV, and File Names.
    """
    if "|" in id_str:
        parts = id_str.split("|")
        # For 'sp|P12345|NAME', the ID is usually the second element
        if len(parts) >= 2:
            return parts[1].strip()
    return id_str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a debug dataset subsample from FASTA, labels, and embeddings."
    )
    parser.add_argument("--fasta", type=str, default=DEFAULT_FASTA)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMB_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_label_ids(labels_path: Path) -> Set[str]:
    """Load normalized protein IDs from the labels TSV."""
    label_ids: Set[str] = set()
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            raw_id = parts[0].strip()
            # Skip the header row
            if raw_id.lower() in ["entryid", "protein_id"]:
                continue
            if raw_id:
                label_ids.add(normalize_id(raw_id))
    return label_ids


def build_embedding_index(emb_dir: Path) -> Dict[str, Path]:
    """Map normalized protein_id to embedding file path."""
    mapping: Dict[str, Path] = {}
    # Use rglob to find files in subdirectories
    for path in emb_dir.rglob("*.pt"):
        protein_id = normalize_id(path.stem)
        if protein_id and protein_id not in mapping:
            mapping[protein_id] = path
    return mapping


def choose_ids(
    fasta_path: Path,
    label_ids: Set[str],
    emb_index: Dict[str, Path],
    sample_size: int,
    seed: int,
) -> Tuple[List[str], Dict[str, Path]]:
    """Choose IDs that exist in all three sources using normalized matching."""
    rng = random.Random(seed)
    reservoir: List[str] = []
    # Store full original FASTA IDs for the reservoir to preserve FASTA formatting
    full_id_reservoir: List[str] = []
    eligible_count = 0

    print("Scanning FASTA for eligible proteins...")
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        norm_id = normalize_id(record.id)

        if norm_id not in label_ids or norm_id not in emb_index:
            continue

        eligible_count += 1
        if len(full_id_reservoir) < sample_size:
            full_id_reservoir.append(record.id)
        else:
            j = rng.randint(1, eligible_count)
            if j <= sample_size:
                full_id_reservoir[j - 1] = record.id

    chosen_embeddings = {
        normalize_id(fid): emb_index[normalize_id(fid)] for fid in full_id_reservoir
    }
    return full_id_reservoir, chosen_embeddings


def write_fasta_subset(
    fasta_path: Path, output_path: Path, chosen_full_ids: Set[str]
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            if record.id in chosen_full_ids:
                SeqIO.write(record, handle, "fasta")
                count += 1
    return count


def write_labels_subset(
    labels_path: Path, output_path: Path, chosen_norm_ids: Set[str]
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    header_written = False
    with (
        labels_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            raw_id = parts[0].strip()

            # Identify and write header
            if raw_id.lower() in ["entryid", "protein_id"]:
                if not header_written:
                    dst.write(line + "\n")
                    header_written = True
                continue

            if normalize_id(raw_id) in chosen_norm_ids:
                dst.write(line + "\n")
                count += 1
    return count


def main() -> None:
    args = parse_args()
    fasta_path, labels_path = Path(args.fasta), Path(args.labels)
    emb_dir, out_dir = Path(args.embeddings), Path(args.output)

    for p in [fasta_path, labels_path, emb_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print(f"Loading metadata...")
    label_ids = load_label_ids(labels_path)
    emb_index = build_embedding_index(emb_dir)

    print(f"Found {len(label_ids)} labeled proteins and {len(emb_index)} embeddings.")

    chosen_full_ids, chosen_embeddings = choose_ids(
        fasta_path, label_ids, emb_index, args.sample_size, args.seed
    )

    chosen_norm_ids = set(chosen_embeddings.keys())
    full_ids_set = set(chosen_full_ids)

    seq_count = write_fasta_subset(
        fasta_path, out_dir / "sequences.fasta", full_ids_set
    )
    label_count = write_labels_subset(
        labels_path, out_dir / "labels.tsv", chosen_norm_ids
    )

    # Copy embeddings
    emb_out = out_dir / "embeddings"
    emb_out.mkdir(parents=True, exist_ok=True)
    for _, src_path in tqdm(chosen_embeddings.items(), desc="Copying embeddings"):
        shutil.copy2(src_path, emb_out / src_path.name)

    print(f"\nDebug dataset created in {out_dir}:")
    print(f"  Sequences: {seq_count}")
    print(f"  Labels:    {label_count}")
    print(f"  Embeds:    {len(chosen_embeddings)}")


if __name__ == "__main__":
    main()
