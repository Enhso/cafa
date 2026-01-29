#!/usr/bin/env python3
"""
Asymmetric, hierarchical, and combined losses for training.
"""

from __future__ import annotations

import torch
from torch import nn


class AsymmetricLossOptimized(nn.Module):
    """
    Optimized Asymmetric Loss (ASL) for multi-label classification.

    L = - y * L_pos - (1 - y) * L_neg

    Args:
        gamma_neg: focusing parameter for negative samples.
        gamma_pos: focusing parameter for positive samples.
        clip: probability clipping for negatives (shifts probability).
    """

    def __init__(
        self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits.
            targets: (B, C) binary targets in {0,1}.

        Returns:
            Scalar loss.
        """
        targets = targets.to(dtype=logits.dtype)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        # Basic CE
        loss_pos = targets * torch.log(torch.clamp(xs_pos, min=1e-8))
        loss_neg = (1.0 - targets) * torch.log(torch.clamp(xs_neg, min=1e-8))

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.no_grad():
                pt = xs_pos * targets + xs_neg * (1.0 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                weight = torch.pow(1.0 - pt, gamma)

            loss = -(loss_pos + loss_neg) * weight
        else:
            loss = -(loss_pos + loss_neg)

        return loss.mean()


class HierarchicalLoss(nn.Module):
    """
    Hierarchical consistency loss for GO DAG.

    Penalizes cases where P(child) > P(parent).
    edge_index is shaped (2, E), with edges child -> parent.
    """

    def __init__(self, edge_index: torch.Tensor) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, Num_Edges)")
        self.register_buffer("edge_index", edge_index.long())

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, C) probabilities in [0,1] (after sigmoid).

        Returns:
            Scalar violation loss.
        """
        child_idx = self.edge_index[0]
        parent_idx = self.edge_index[1]

        child_probs = probs[:, child_idx]
        parent_probs = probs[:, parent_idx]

        violation = torch.relu(child_probs - parent_probs)
        return violation.sum()


class CombinedLoss(nn.Module):
    """
    Combined loss: AsymmetricLossOptimized + lambda * HierarchicalLoss
    """

    def __init__(
        self, asl: AsymmetricLossOptimized, hl: HierarchicalLoss, lambda_h: float = 1.0
    ) -> None:
        super().__init__()
        self.asl = asl
        self.hl = hl
        self.lambda_h = lambda_h

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits.
            targets: (B, C) binary targets.

        Returns:
            Scalar combined loss.
        """
        asl_loss = self.asl(logits, targets)
        probs = torch.sigmoid(logits)
        h_loss = self.hl(probs)
        return asl_loss + self.lambda_h * h_loss
