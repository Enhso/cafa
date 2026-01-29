#!/usr/bin/env python3
"""
Training utilities with warmup scheduler and alpha-first epoch logic.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """
    Build AdamW optimizer and cosine scheduler with warmup.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def apply_alpha_first_freeze(model: nn.Module, epoch: int) -> None:
    """
    Alpha-first logic:
      - epoch == 0: freeze encoder and head.linear weights,
        only train head.alpha and head.linear.bias (if exists).
      - epoch > 0: unfreeze everything.
    """
    if epoch == 0:
        if hasattr(model, "encoder"):
            _set_requires_grad(model.encoder, False)

        if hasattr(model, "head") and hasattr(model.head, "linear"):
            _set_requires_grad(model.head.linear, False)

            # Allow bias if present
            if model.head.linear.bias is not None:
                model.head.linear.bias.requires_grad = True

        if hasattr(model, "head") and hasattr(model.head, "alpha"):
            model.head.alpha.requires_grad = True
    else:
        _set_requires_grad(model, True)


def train_epoch(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    asl_loss_fn: nn.Module,
    hier_loss_fn: nn.Module,
    epoch: int,
    device: torch.device,
    lambda_h: float = 1.0,
) -> Dict[str, float]:
    """
    Train one epoch with alpha-first logic and separate logging for ASL and hierarchy.

    Args:
        model: ProteinFunctionModel.
        dataloader: Iterable of batches from DataLoader.
        optimizer: AdamW optimizer.
        scheduler: cosine warmup scheduler.
        asl_loss_fn: AsymmetricLossOptimized instance.
        hier_loss_fn: HierarchicalLoss instance.
        epoch: current epoch index.
        device: torch.device.
        lambda_h: weight for hierarchical loss.

    Returns:
        Dict with average losses: total, asl, hier.
    """
    model.train()
    apply_alpha_first_freeze(model, epoch)

    total_loss = 0.0
    total_asl = 0.0
    total_hier = 0.0
    n_batches = 0

    for batch in dataloader:
        esm_embedding = batch["esm_embedding"].to(device)
        sequence_encoding = batch["sequence_encoding"].to(device)
        stat_features = batch["stat_features"].to(device)
        labels = batch["labels"].to(device)
        homology_vector = batch["homology_vector"].to(device)
        if homology_vector.is_sparse:
            homology_vector = homology_vector.to_dense()

        logits = model(
            esm_embedding=esm_embedding,
            sequence_encoding=sequence_encoding,
            stat_features=stat_features,
            homology_vector=homology_vector,
        )

        asl_loss = asl_loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        hier_loss = hier_loss_fn(probs)

        loss = asl_loss + lambda_h * hier_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach().item())
        total_asl += float(asl_loss.detach().item())
        total_hier += float(hier_loss.detach().item())
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "asl": 0.0, "hier": 0.0}

    return {
        "loss": total_loss / n_batches,
        "asl": total_asl / n_batches,
        "hier": total_hier / n_batches,
    }
