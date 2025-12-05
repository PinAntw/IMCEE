# Research/IMCEE/modules/crossTower/cross_tower_model.py
from .text_encoder import TextNodeEncoder
from .semantic_tower import SemanticTower
from .structural_tower import StructuralGraphTower
# from .fusion import CrossAttentionFusion 
from torch_scatter import scatter_max
from torch_geometric.utils import to_dense_batch
import torch
import torch.nn as nn

class CrossTowerCausalModel(nn.Module):
    def __init__(self, text_model_name, hidden_dim, expl_dim,
                 num_speakers, num_emotions,
                 dropout=0.1,        
                 gnn_dropout=0.3,    
                 num_gnn_layers=3,
                 freeze_text=True, use_explain=True):
        super().__init__()

        self.use_explain = use_explain and expl_dim > 0
        self.expl_dim = expl_dim
        self.hidden_dim = hidden_dim

        # 0. Text Encoder
        self.text_encoder = TextNodeEncoder(
            text_model_name=text_model_name,
            freeze_text=freeze_text,
            num_speakers=num_speakers, 
            num_emotions=num_emotions,
            dropout=dropout 
        )
        d_sem = self.text_encoder.d_sem 

        # 1. Semantic Projector
        self.semantic_tower = SemanticTower(
            in_dim=d_sem,
            out_dim=hidden_dim,
            expl_dim=(expl_dim if self.use_explain else 0),
            dropout=dropout,
        )

        # 2. Structural Tower (Graph)
        self.structural_tower = StructuralGraphTower(
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=gnn_dropout,   
        )

        # 3. Fusion Strategy (Bypassed)
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Edge-level explain MLP
        if self.use_explain:
            self.edge_explain_mlp = nn.Sequential(
                nn.Linear(expl_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # 每個節點: [h_graph; h_text] = 2 * hidden_dim
            # 兩個節點 (Cand + Target) = 4 * hidden_dim
            # 解釋特徵 = hidden_dim (因為經過了 MLP)
            pred_in_dim = (hidden_dim * 2) * 2 + hidden_dim
        else:
            self.edge_explain_mlp = None
            pred_in_dim = (hidden_dim * 2) * 2

        # 5. Predictor
        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        """
        batch: 
          現在是一個 Batch Object，包含多個獨立的子圖。
          關鍵屬性:
          - target_node_indices: [B, 2] (Column 0: Cause Node, Column 1: Target Node)
          - edge_features: [Total_Edges, Expl_Dim] (只有特定邊有值，其他為0)
        """

        # 0. Text Encoder
        h_text = self.text_encoder(
            batch.input_ids,
            batch.token_mask,
            batch.utterance_mask,
            batch.speaker_ids_padded,
            batch.emotion_ids_padded
        )  # [Total_Nodes, hidden_dim]

        # 1. Structural tower encoding (Graph)
        h_graph = self.structural_tower(h_text, batch)  # [Total_Nodes, hidden_dim]
        
        # 2. Skip Connection Fusion
        # h_final: [Total_Nodes, hidden_dim * 2]
        h_final = torch.cat([h_graph, h_text], dim=-1)

        # ==========================================
        # 從 Dense Batch 中提取 Cause/Target Node
        # ==========================================
        
        # 將平坦的 [Total_Nodes, D] 轉換回 [Batch_Size, Max_Nodes, D]
        h_dense, mask = to_dense_batch(h_final, batch.batch) 
        
        cause_idx = batch.target_node_indices[:, 0]  # [B]
        target_idx = batch.target_node_indices[:, 1] # [B]
        
        batch_indices = torch.arange(h_dense.size(0), device=h_dense.device)
        
        # 取出特徵
        h_c = h_dense[batch_indices, cause_idx]   # [B, 2*hidden_dim]
        h_t = h_dense[batch_indices, target_idx]  # [B, 2*hidden_dim]

        feat = [h_c, h_t]

        # ==========================================
        # 提取 Edge Features (Explain)
        # ==========================================
        if self.use_explain and hasattr(batch, "edge_features"):
            # 1. 找出每條邊屬於哪張圖
            edge_batch_id = batch.batch[batch.edge_index[0]]
            
            # 2. 聚合 (Scatter Max) - 取出該圖中最大的特徵 (因為非 Target 邊都是 0)
            e_feat, _ = scatter_max(batch.edge_features, edge_batch_id, dim=0)
            
            # 3. MLP 轉換
            e = self.edge_explain_mlp(e_feat) # [B, hidden_dim]
            feat.append(e)

        # 串接所有特徵 -> [B, pred_in_dim]
        edge_repr = torch.cat(feat, dim=-1)
        
        logits = self.predictor(edge_repr).squeeze(-1)
        return logits