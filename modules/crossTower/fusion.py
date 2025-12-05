#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/modules/crossTower/fusion.py
"""
fusion.py
- CrossAttentionFusion: 結構塔 query 語意塔的 cross-attention 融合層
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Query = graph tower (structure-informed)
    Key/Value = semantic tower (PLM-informed)
    """
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_graph, h_sem):
        """
        h_graph: [N, D]  — structure-informed
        h_sem:   [N, D]  — semantic-informed
        """
        h_graph_b = h_graph.unsqueeze(0)  # [1, N, D]
        h_sem_b   = h_sem.unsqueeze(0)    # [1, N, D]

        out, _ = self.attn(
            query=h_graph_b,
            key=h_sem_b,
            value=h_sem_b
        )  # [1, N, D]
        out = out.squeeze(0)             # [N, D]
        return self.norm(h_graph + out)  # 殘差 + LN

