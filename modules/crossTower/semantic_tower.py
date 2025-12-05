# Research/IMCEE/modules/crossTower/semantic_tower.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_tower.py
- SemanticTower: h_text (+ node-explain from edges) → hidden_dim
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class SemanticTower(nn.Module):
    """
    Semantic Tower
      - input:  h_text [N, d_plm]
      - optional: edge_features → 聚合成 node_explain [N, expl_dim]
      - output: h_sem [N, out_dim]
    不包含 PLM，本身不處理 B,S,T。
    """
    def __init__(self, in_dim, out_dim,
                 expl_dim=0, dropout=0.1, use_explain=True):
        super().__init__()

        self.use_explain = use_explain and expl_dim > 0
        self.expl_dim = expl_dim

        input_dim = in_dim + (expl_dim if self.use_explain else 0)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    # ---- edge → node explain ----
    def _node_explain_from_edge(self, edge_index, edge_features,
                                num_nodes, edge_task_mask):
        """
        只用 task edges 產生 node-level explain
        edge_index:     [2, E]
        edge_features:  [E, expl_dim]
        edge_task_mask: [E] bool
        num_nodes:      int
        return:         [N, expl_dim]
        """
        src = edge_index[0][edge_task_mask]
        tgt = edge_index[1][edge_task_mask]
        feat = edge_features[edge_task_mask]  # [E_task, expl_dim]

        all_idx = torch.cat([src, tgt], dim=0)
        all_feat = torch.cat([feat, feat], dim=0)
        return scatter_mean(all_feat, all_idx, dim=0, dim_size=num_nodes)  # [N, expl_dim]

    def forward(self, h_text,
                edge_index=None,
                edge_features=None,
                num_nodes=None,
                edge_task_mask=None):
        """
        h_text:        [N, d_plm]
        edge_index:     [2, E]          (optional, 用於 explain)
        edge_features:  [E, expl_dim]   (optional)
        num_nodes:      int             (optional)
        edge_task_mask: [E] bool        (optional)
        return:         [N, out_dim]
        """
        node_explain = None
        if self.use_explain and edge_features is not None:
            node_explain = self._node_explain_from_edge(
                edge_index,
                edge_features,
                num_nodes,
                edge_task_mask,
            )  # [N, expl_dim]

        if self.use_explain and node_explain is not None:
            h = torch.cat([h_text, node_explain], dim=-1)
        else:
            h = h_text

        return self.proj(h)  # [N, out_dim]
