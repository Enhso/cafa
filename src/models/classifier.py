#!/usr/bin/env python3
"""
Residual classifier head and combined protein function model.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import DualPathEncoder


class ResidualHead(nn.Module):
    """
    Residual classifier head that adds a homology bias to neural logits.
    """

    def __init__(self, input_dim: int = 1024, num_classes: int = 40000) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.alpha = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))

    def forward(
        self, embedding: torch.Tensor, homology_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embedding: (B, input_dim)
            homology_vector: (B, num_classes)

        Returns:
            final_logits: (B, num_classes)
        """
        neural_logits = self.linear(embedding)
        homology_bias = self.alpha * homology_vector
        final_logits = neural_logits + homology_bias
        return final_logits


class ProteinFunctionModel(nn.Module):
    """
    Wrapper model combining DualPathEncoder and ResidualHead.
    """

    def __init__(
        self,
        encoder: DualPathEncoder,
        head: ResidualHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        esm_embedding: torch.Tensor,
        sequence_encoding: torch.Tensor,
        stat_features: torch.Tensor,
        homology_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            esm_embedding: (B, esm_dim)
            sequence_encoding: (B, L)
            stat_features: (B, stat_dim)
            homology_vector: (B, num_classes)

        Returns:
            final_logits: (B, num_classes)
        """
        fused = self.encoder(esm_embedding, sequence_encoding, stat_features)
        logits = self.head(fused, homology_vector)
        return logits
