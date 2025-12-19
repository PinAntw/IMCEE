# Research/IMCEE/modules/crossTower/cross_tower_model.py
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max
from .text_encoder import TextNodeEncoder
from .semantic_tower import SemanticTower
from .structural_tower import StructuralGraphTower

class CrossTowerCausalModel(nn.Module):
    def __init__(
        self,
        text_model_name,
        hidden_dim,
        expl_dim,
        num_speakers,
        num_emotions,
        dropout=0.1,
        gnn_dropout=0.3,
        num_gnn_layers=3,
        freeze_text=True,
        use_explain=True,
        expl_space_dim=0,
        use_expl_space=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_explain = use_explain and expl_dim > 0
        self.use_expl_space = use_expl_space and expl_space_dim > 0
        self.expl_space_dim = expl_space_dim

        # 0. Text Encoder
        self.text_encoder = TextNodeEncoder(
            text_model_name=text_model_name,
            freeze_text=freeze_text,
            num_speakers=num_speakers,
            num_emotions=num_emotions,
            dropout=dropout,
        )
        d_sem = self.text_encoder.d_sem

        # 1. Semantic Tower
        self.semantic_tower = SemanticTower(
            in_dim=d_sem,
            out_dim=hidden_dim,
            expl_dim=(expl_dim if self.use_explain else 0),
            dropout=dropout,
        )

        # 2. Structural Tower
        self.structural_tower = StructuralGraphTower(
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=gnn_dropout,
        )

        # 3. [新增] 顯式距離編碼器 (Explicit Distance Embedder)
        # 定義 32 種距離的 embedding
        self.dist_embedder = nn.Embedding(32, hidden_dim)

        # 4. Teacher Projector (Student Head) - 用於計算 Distillation Loss
        # 這是讓 Student 練內功用的，即使 Predictor 不用它，它也要存在
        self.student_expl_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5. Teacher Projector (Teacher Head) - 用於注入 LLM 解釋
        if self.use_expl_space:
            self.expl_space_mlp = nn.Sequential(
                nn.Linear(expl_space_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.expl_space_mlp = None

        # 6. Predictor (The "Judge")
        # 輸入特徵串接：
        # - h_c (Student Cause): 2H
        # - h_t (Student Target): 2H
        # - h_dist (Explicit Distance): H
        # - z_teacher (LLM Reasoning): H (若有開啟)
        
        base_dim = hidden_dim * 4 # Student
        dist_dim = hidden_dim     # Distance
        teacher_dim = hidden_dim if self.use_expl_space else 0 # Teacher
        edge_expl_dim = hidden_dim if self.use_explain else 0 # Old Explain
        
        pred_in_dim = base_dim + dist_dim + teacher_dim + edge_expl_dim

        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.ReLU(), # 關鍵：讓模型學習非線性組合邏輯
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch, return_aux=False):
        # 0. Text Encoding
        h_text = self.text_encoder(
            batch.input_ids, batch.token_mask, batch.utterance_mask,
            batch.speaker_ids_padded, batch.emotion_ids_padded,
        )

        # 1. GNN Encoding
        h_graph = self.structural_tower(h_text, batch)

        # 2. Dense Batching
        h_final = torch.cat([h_graph, h_text], dim=-1) # [Nodes, 2H]
        h_dense, mask = to_dense_batch(h_final, batch.batch)
        
        B = h_dense.size(0)
        batch_idx = torch.arange(B, device=h_dense.device)
        cause_idx = batch.target_node_indices[:, 0]
        target_idx = batch.target_node_indices[:, 1]

        h_c = h_dense[batch_idx, cause_idx]   # [B, 2H]
        h_t = h_dense[batch_idx, target_idx]  # [B, 2H]

        # -------------------------------------------------
        # A. 計算 Student Representation (為了算 Loss)
        # -------------------------------------------------
        h_pair = torch.cat([h_c, h_t], dim=-1)
        z_student = self.student_expl_proj(h_pair)

        # -------------------------------------------------
        # B. 準備 Teacher Representation (z_teacher)
        # -------------------------------------------------
        z_teacher = None
        if self.use_expl_space and hasattr(batch, "expl_space_vec"):
            vec = batch.expl_space_vec.to(h_final.device)
            vec = vec.view(-1, self.expl_space_dim)
            z_teacher = self.expl_space_mlp(vec) # [B, H]

        # -------------------------------------------------
        # C. 準備 Distance Representation (h_dist)
        # -------------------------------------------------
        with torch.no_grad():
            edge_src = batch.edge_index[0]
            edge_tgt = batch.edge_index[1]
            batch_dists = []
            
            # 使用 CPU 或 GPU 向量化操作會更快，這裡用迴圈簡單示意
            c_nodes = batch.target_node_indices[:, 0]
            t_nodes = batch.target_node_indices[:, 1]
            
            # 這裡假設 CauseNode/TargetNode (super nodes) 各自連接著一個 Utterance Node
            # 我們要找出那個 Utterance Node 的 Index
            # 在 dataset.py 中，CauseNode -> CauseUtt 是 Type 0
            
            for i in range(len(c_nodes)):
                c_node = c_nodes[i]
                t_node = t_nodes[i]
                
                # 找出鄰居 (Type 0, src=SuperNode)
                mask_c = (edge_src == c_node)
                c_utt = edge_tgt[mask_c][0] if mask_c.any() else c_node
                
                mask_t = (edge_src == t_node)
                t_utt = edge_tgt[mask_t][0] if mask_t.any() else t_node
                
                dist = torch.abs(c_utt - t_utt)
                dist = torch.clamp(dist, max=31)
                batch_dists.append(dist)
                
            dist_tensor = torch.stack(batch_dists).to(h_final.device)
            
        h_dist = self.dist_embedder(dist_tensor) # [B, H]

        # -------------------------------------------------
        # D. 終極融合 (Feature Injection)
        # -------------------------------------------------
        # 將所有資訊注入給 Predictor
        feat_list = [h_c, h_t, h_dist]
        
        if z_teacher is not None:
            feat_list.append(z_teacher) # 這是救命稻草

        # (Optional: Old Edge Explain)
        if self.use_explain and hasattr(batch, "edge_features"):
            edge_batch = batch.batch[batch.edge_index[0]]
            e_feat, _ = scatter_max(batch.edge_features, edge_batch, dim=0)
            e_edge = self.edge_explain_mlp(e_feat)
            feat_list.append(e_edge)

        # Concat & Predict
        edge_repr = torch.cat(feat_list, dim=-1)
        logits = self.predictor(edge_repr).squeeze(-1)

        # -------------------------------------------------
        # Return
        # -------------------------------------------------
        if return_aux:
            return logits, {
                "z_student": z_student,
                "z_teacher": z_teacher,
                "dist": dist_tensor # 方便之後分析距離分佈
            }

        # 訓練模式：回傳 logits 和對齊用的 student/teacher
        if self.training and self.use_expl_space and (z_teacher is not None):
            return logits, z_student, z_teacher
            
        return logits