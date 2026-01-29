#!/usr/bin/env python3
"""
Cluster-based iterative stratified splitting utilities.
Fixed: ValueError in newer scikit-learn versions by setting random_state=None
in IterativeStratification and shuffling manually beforehand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification


def _parse_terms(go_terms: str) -> List[str]:
    if go_terms is None or (isinstance(go_terms, float) and np.isnan(go_terms)):
        return []
    return [t.strip() for t in str(go_terms).split(";") if t.strip()]


def load_and_join_data(cluster_path: Path, metadata_path: Path) -> pd.DataFrame:
    """
    Inner joins cluster information and GO metadata.
    Handles a cluster file with NO headers.
    """
    # Load cluster file: No header, two columns [cluster_id, protein_id]
    clusters_df = pd.read_csv(
        cluster_path, sep="\t", header=None, names=["cluster_id", "protein_id"]
    )

    # Load metadata file [protein_id, sequence, go_terms]
    metadata_df = pd.read_csv(metadata_path)

    # Clean IDs to ensure matching works
    clusters_df["protein_id"] = clusters_df["protein_id"].astype(str).str.strip()
    metadata_df["protein_id"] = metadata_df["protein_id"].astype(str).str.strip()

    # Inner Join on protein_id
    joined_df = pd.merge(clusters_df, metadata_df, on="protein_id", how="inner")

    # Validation: Flag discrepancy if metadata rows != joined rows
    meta_count = len(metadata_df)
    joined_count = len(joined_df)

    if meta_count != joined_count:
        print(f"⚠️ DISCREPANCY DETECTED:")
        print(f"  Metadata rows: {meta_count}")
        print(f"  Joined rows:   {joined_count}")
        print(
            f"  Difference:    {abs(meta_count - joined_count)} proteins were lost in the join."
        )
    else:
        print(
            f"✅ Validation successful: Metadata and Joined counts match ({joined_count})."
        )

    return joined_df[["protein_id", "cluster_id", "go_terms"]]


def _iterative_split_indices(
    Y: np.ndarray,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits indices using Iterative Stratification.
    Note: random_state is set to None to avoid sklearn compatibility errors.
    """
    if test_size <= 0.0:
        idx = np.arange(Y.shape[0])
        return idx, np.array([], dtype=int)
    if test_size >= 1.0:
        idx = np.arange(Y.shape[0])
        return np.array([], dtype=int), idx

    X = np.zeros((Y.shape[0], 1))

    # FIX: random_state=None because skmultilearn doesn't support shuffle=True
    # but sklearn throws an error if random_state is set without shuffle.
    splitter = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[1.0 - test_size, test_size],
        random_state=None,
    )
    train_idx, test_idx = next(splitter.split(X, Y))
    return train_idx, test_idx


def _build_cluster_labels(
    df: pd.DataFrame,
) -> Tuple[List[str], List[List[str]], List[str], np.ndarray]:
    grouped = {}
    for _, row in df.iterrows():
        cid = str(row["cluster_id"])
        terms = _parse_terms(row["go_terms"])
        grouped.setdefault(cid, set()).update(terms)

    cluster_ids = sorted(grouped.keys())
    cluster_terms = [sorted(grouped[cid]) for cid in cluster_ids]
    all_terms = sorted({t for terms in cluster_terms for t in terms})
    term_index = {t: i for i, t in enumerate(all_terms)}

    Y = np.zeros((len(cluster_ids), len(all_terms)), dtype=int)
    for i, terms in enumerate(cluster_terms):
        for t in terms:
            Y[i, term_index[t]] = 1

    return cluster_ids, cluster_terms, all_terms, Y


def _find_forced_train_clusters(cluster_ids: Sequence[str], Y: np.ndarray) -> Set[str]:
    if Y.size == 0:
        return set()
    term_counts = Y.sum(axis=0)
    rare_term_indices = np.where(term_counts == 1)[0]
    forced = set()
    for term_idx in rare_term_indices:
        cluster_idx = int(np.where(Y[:, term_idx] == 1)[0][0])
        forced.add(cluster_ids[cluster_idx])
    return forced


def create_stratified_splits(
    df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the joined dataframe into train, val, and test based on clusters.
    Manually shuffles data to ensure randomness since the splitter can't use seed.
    """
    # 0. Manual Shuffling for reproducibility
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    cluster_ids, _, _, Y = _build_cluster_labels(df)
    forced_train_clusters = _find_forced_train_clusters(cluster_ids, Y)
    total_clusters = len(cluster_ids)
    candidate_clusters = [c for c in cluster_ids if c not in forced_train_clusters]

    if total_clusters == 0:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    if not candidate_clusters:
        train_clusters, val_clusters, test_clusters = set(cluster_ids), set(), set()
    else:
        desired_test = int(round(total_clusters * test_size))
        desired_val = int(round(total_clusters * val_size))
        candidate_indices = [cluster_ids.index(c) for c in candidate_clusters]
        Y_candidates = Y[candidate_indices]

        c_test_size = min(max(desired_test / len(candidate_clusters), 0.0), 1.0)
        trainval_idx, test_idx = _iterative_split_indices(Y_candidates, c_test_size)

        candidate_trainval = [candidate_clusters[i] for i in trainval_idx]
        candidate_test = [candidate_clusters[i] for i in test_idx]

        rem = len(candidate_trainval)
        c_val_size = min(max(desired_val / rem, 0.0), 1.0) if rem > 0 else 0.0

        if rem > 0 and c_val_size > 0.0:
            train_idx, val_idx = _iterative_split_indices(
                Y_candidates[trainval_idx], c_val_size
            )
            candidate_train = [candidate_trainval[i] for i in train_idx]
            candidate_val = [candidate_trainval[i] for i in val_idx]
        else:
            candidate_train, candidate_val = candidate_trainval, []

        train_clusters = set(candidate_train) | forced_train_clusters
        val_clusters, test_clusters = set(candidate_val), set(candidate_test)

    return (
        df[df["cluster_id"].astype(str).isin(train_clusters)].copy(),
        df[df["cluster_id"].astype(str).isin(val_clusters)].copy(),
        df[df["cluster_id"].astype(str).isin(test_clusters)].copy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Join data and split into train/val/test JSONs."
    )
    parser.add_argument(
        "--clusters",
        type=str,
        required=True,
        help="Path to headerless train_cluster.tsv",
    )
    parser.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata.csv"
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load, Join, and Validate
    print("Loading data...")
    joined_df = load_and_join_data(Path(args.clusters), Path(args.metadata))

    # 2. Split at Cluster Level
    print("Creating stratified splits...")
    train_df, val_df, test_df = create_stratified_splits(
        joined_df, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    # 3. Export
    outputs = {
        "train_ids.json": train_df,
        "val_ids.json": val_df,
        "test_ids.json": test_df,
    }

    for filename, df in outputs.items():
        ids = df["protein_id"].tolist()
        with open(filename, "w") as f:
            json.dump(ids, f, indent=2)
        print(f"✅ Successfully wrote {len(ids)} IDs to {filename}")


if __name__ == "__main__":
    main()
