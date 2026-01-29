#!/usr/bin/env python3
"""
Inference utilities: threshold optimization, GO consistency enforcement, and prediction.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _term_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _micro_f1(y_true, y_pred)


def _resolve_labels_and_ontology(
    val_labels: object,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Supports:
      - val_labels as np.ndarray
      - val_labels as dict with keys {"labels", "term_ontology"}
    """
    if isinstance(val_labels, dict):
        labels = val_labels.get("labels")
        ontology = val_labels.get("term_ontology")
        if labels is None:
            raise ValueError("val_labels dict must include 'labels'.")
        return np.asarray(labels), ontology
    return np.asarray(val_labels), None


def optimize_thresholds(val_probs: np.ndarray, val_labels: object) -> np.ndarray:
    """
    Optimize thresholds using hybrid global/local strategy.

    Steps:
      1) Best global threshold per ontology (MF/BP/CC) by Micro-F1.
      2) Best local threshold per term by Term-F1.
      3) Use global for terms with <10 positives; else local.

    If ontology info is not provided, a single global threshold is used.

    Args:
        val_probs: (N, C) probabilities.
        val_labels: (N, C) labels or dict with {"labels": ..., "term_ontology": ...}.

    Returns:
        thresholds: (C,) float32 vector.
    """
    labels, term_ontology = _resolve_labels_and_ontology(val_labels)
    probs = np.asarray(val_probs)

    if probs.shape != labels.shape:
        raise ValueError("val_probs and val_labels must have the same shape.")

    n_classes = probs.shape[1]
    thresholds = np.zeros(n_classes, dtype=np.float32)

    # Candidate thresholds
    candidates = np.linspace(0.0, 1.0, 101)

    # Global thresholds per ontology
    if term_ontology is None:
        # Single global threshold
        best_t = 0.5
        best_f1 = -1.0
        for t in candidates:
            pred = (probs >= t).astype(np.int32)
            f1 = _micro_f1(labels, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        global_thresholds = {"ALL": best_t}
        term_ontology = ["ALL"] * n_classes
    else:
        if len(term_ontology) != n_classes:
            raise ValueError("term_ontology length must match number of classes.")
        global_thresholds = {}
        for ont in ["MF", "BP", "CC"]:
            idx = [i for i, o in enumerate(term_ontology) if o == ont]
            if not idx:
                continue
            best_t = 0.5
            best_f1 = -1.0
            for t in candidates:
                pred = (probs[:, idx] >= t).astype(np.int32)
                f1 = _micro_f1(labels[:, idx], pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            global_thresholds[ont] = best_t

    # Local thresholds per term
    local_thresholds = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        best_t = 0.5
        best_f1 = -1.0
        for t in candidates:
            pred = (probs[:, i] >= t).astype(np.int32)
            f1 = _term_f1(labels[:, i], pred)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        local_thresholds[i] = best_t

    # Combine with safety latch
    pos_counts = labels.sum(axis=0)
    for i in range(n_classes):
        if pos_counts[i] < 10:
            ont = term_ontology[i]
            thresholds[i] = global_thresholds.get(ont, 0.5)
        else:
            thresholds[i] = local_thresholds[i]

    return thresholds


def enforce_consistency(
    probs: np.ndarray | torch.Tensor, edge_index: torch.Tensor
) -> np.ndarray:
    """
    Enforce GO DAG consistency with bottom-up propagation.

    Rule: Prob_Parent = max(Prob_Parent, Prob_Child)

    Args:
        probs: (N, C) array/tensor of probabilities or binary predictions.
        edge_index: (2, E) tensor (child -> parent).

    Returns:
        Updated probabilities as numpy array.
    """
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)

    edge = edge_index.detach().cpu().numpy()
    child_nodes = edge[0].astype(int)
    parent_nodes = edge[1].astype(int)

    # Repeated passes to propagate upward
    num_nodes = probs_np.shape[1]
    for _ in range(num_nodes):
        updated = False
        for c, p in zip(child_nodes, parent_nodes):
            child_vals = probs_np[:, c]
            parent_vals = probs_np[:, p]
            new_vals = np.maximum(parent_vals, child_vals)
            if not np.array_equal(new_vals, parent_vals):
                probs_np[:, p] = new_vals
                updated = True
        if not updated:
            break

    return probs_np


def predict(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    thresholds: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict with thresholds and consistency enforcement.

    Args:
        model: trained model.
        dataloader: iterable of batches.
        thresholds: (C,) thresholds vector.
        edge_index: (2, E) child->parent edges.
        device: torch.device.

    Returns:
        Binary predictions array (N, C).
    """
    model.eval()
    preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            esm_embedding = batch["esm_embedding"].to(device)
            sequence_encoding = batch["sequence_encoding"].to(device)
            stat_features = batch["stat_features"].to(device)
            homology_vector = batch["homology_vector"].to(device)
            if homology_vector.is_sparse:
                homology_vector = homology_vector.to_dense()

            logits = model(
                esm_embedding=esm_embedding,
                sequence_encoding=sequence_encoding,
                stat_features=stat_features,
                homology_vector=homology_vector,
            )
            probs = torch.sigmoid(logits)

            # Apply thresholds first
            thr = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype)
            binary = (probs >= thr).float()

            # Consistency enforcement on binary predictions
            binary_np = enforce_consistency(binary, edge_index)
            preds.append((binary_np >= 0.5).astype(np.int32))

    if not preds:
        return np.zeros((0, thresholds.shape[0]), dtype=np.int32)

    return np.vstack(preds)
