#!/usr/bin/env python3
"""
Dual-path encoder for protein function prediction.

Inputs:
- esm_embedding: global ESM-2 embedding (B, esm_dim)
- sequence_encoding: tokenized sequence (B, L)
- stat_features: prior features (B, stat_dim)

Output:
- fused embedding (B, 1024)
"""

from __future__ import annotations

import torch
from torch import nn


class DualPathEncoder(nn.Module):
    """
    Dual-path encoder that fuses:
      - Path A: linear projection of ESM embeddings
      - Path B: CNN over sequence + stat features
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        vocab_size: int = 22,
        seq_embed_dim: int = 64,
        conv_channels: int = 128,
        stat_dim: int = 505,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        # Path A: ESM embedding projection
        self.path_a = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
        )

        # Path B: sequence CNN (Inception-style)
        self.seq_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=seq_embed_dim,
            padding_idx=pad_idx,
        )

        self.conv3 = nn.Conv1d(seq_embed_dim, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(seq_embed_dim, conv_channels, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(seq_embed_dim, conv_channels, kernel_size=9, padding=4)

        self.path_b_proj = nn.Sequential(
            nn.Linear(conv_channels * 3 + stat_dim, 512),
            nn.ReLU(),
        )

    def forward(
        self,
        esm_embedding: torch.Tensor,
        sequence_encoding: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            esm_embedding: (B, esm_dim)
            sequence_encoding: (B, L)
            stat_features: (B, stat_dim)

        Returns:
            fused embedding: (B, 1024)
        """
        # Path A
        path_a_out = self.path_a(esm_embedding)

        # Path B: embedding -> convs -> global max pool
        seq_emb = self.seq_embedding(sequence_encoding)  # (B, L, E)
        seq_emb = seq_emb.transpose(1, 2)  # (B, E, L)

        c3 = torch.relu(self.conv3(seq_emb))
        c5 = torch.relu(self.conv5(seq_emb))
        c9 = torch.relu(self.conv9(seq_emb))

        p3 = torch.amax(c3, dim=2)
        p5 = torch.amax(c5, dim=2)
        p9 = torch.amax(c9, dim=2)

        cnn_features = torch.cat([p3, p5, p9], dim=1)
        path_b_in = torch.cat([cnn_features, stat_features], dim=1)
        path_b_out = self.path_b_proj(path_b_in)

        # Fusion
        fused = torch.cat([path_a_out, path_b_out], dim=1)
        return fused
