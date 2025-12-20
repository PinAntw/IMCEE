#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/modules/crossTower/structural_tower.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_edge 

class StructuralGraphTower(nn.Module):
    def __init__(self, 
                 hidden_dim,       # 輸入維度 (通常來自 Text Encoder)
                 internal_dim=768, # RGCN 內部運算維度
                 num_gnn_layers=2, 
                 dropout=0.3,      
                 num_relations=6,  # [Updated] 對應新的 dataset.py 定義的 6 種關係
                 num_bases=4,      # Basis Decomposition 以減少參數
                 edge_dropout=0.2): 
        super().__init__()

        self.edge_dropout = edge_dropout
        self.dropout_rate = dropout
        self.num_relations = num_relations

        # -----------------------------------------------------------
        # Relation Types Mapping (Based on Dataset V7):
        # 0: utt -> super anchoring
        # 1: Temporal Dist 1 (相鄰)
        # 2: Temporal Dist 2 (隔一句)
        # 3: Type 3：context window edges（preceding utt -> super, w=4）
        # 4: Speaker Edge (同說話者)
        # 5: Global Edge (Conversation Node <-> All Utts)
        # -----------------------------------------------------------

        # 1. [Entry Projection]
        self.input_proj = nn.Linear(hidden_dim, internal_dim)

        # 2. [RGCN Layers] 
        self.conv_layers = nn.ModuleList()
        self.bns = nn.ModuleList() 

        for _ in range(num_gnn_layers):
            self.conv_layers.append(
                RGCNConv(
                    in_channels=internal_dim,
                    out_channels=internal_dim,
                    num_relations=num_relations,
                    num_bases=num_bases 
                )
            )
            self.bns.append(nn.BatchNorm1d(internal_dim))

        # 3. [Exit Projection]
        self.output_proj = nn.Linear(internal_dim, hidden_dim)
        self.output_ln = nn.LayerNorm(hidden_dim) 

    def forward(self, h_text, batch):
        """
        Args:
            h_text: [Total_Nodes, hidden_dim] 
                    這裡的 Total_Nodes 包含 Utterances + ConvNode + CauseNode + TargetNode
            batch: PyG Batch object，包含 edge_index, edge_types 等
        """        
        # 1. Input Projection
        x = self.input_proj(h_text)
        x = F.relu(x) 
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 2. Edge Dropout (Training only)
        # 這會隨機丟棄一些邊，增加模型魯棒性
        edge_index = batch.edge_index
        edge_type = batch.edge_types
        
        if self.training and self.edge_dropout > 0:
            # force_undirected=False 因為我們的邊是有向的 (除了 Global Edge 是雙向建立)
            edge_index, mask = dropout_edge(edge_index, p=self.edge_dropout, force_undirected=False)
            edge_type = edge_type[mask]

        # 3. RGCN Loop
        for conv, bn in zip(self.conv_layers, self.bns):
            h_in = x 
            
            # RGCNConv 核心：
            # 它會根據 edge_type 選擇不同的權重矩陣來聚合鄰居資訊
            # Cause Node 會透過 Type 0 的邊，專注於聚合它對應的那個 Utterance 的資訊
            x = conv(x, edge_index, edge_type)
            
            x = F.relu(x)
            x = x + h_in  # Residual Connection
            x = bn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 4. Output Projection
        h_final = self.output_proj(x)
        h_final = self.output_ln(h_final)

        return h_final
