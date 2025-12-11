from .text_encoder import TextNodeEncoder
from .semantic_tower import SemanticTower
from .structural_tower import StructuralGraphTower

from torch_scatter import scatter_max
from torch_geometric.utils import to_dense_batch
import torch
import torch.nn as nn


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
        self.expl_dim = expl_dim

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

        # 3. Fusion
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 4. Edge Explain
        if self.use_explain:
            self.edge_explain_mlp = nn.Sequential(
                nn.Linear(expl_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.edge_explain_mlp = None

        # 5. Teacher & Gate
        if self.use_expl_space:
            self.expl_space_mlp = nn.Sequential(
                nn.Linear(expl_space_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),     # gate = [0,1]
            )
        else:
            self.expl_space_mlp = None
            self.gate_mlp = None

        # 6. Student Head
        self.student_expl_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 7. Predictor
        base_dim = hidden_dim * 4
        extra_dim = hidden_dim if self.use_explain else 0
        pred_in_dim = base_dim + extra_dim

        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # [修改]: 增加 return_aux 參數以便監測
    def forward(self, batch, return_aux=False):

        # 0. PLM encoding
        h_text = self.text_encoder(
            batch.input_ids,
            batch.token_mask,
            batch.utterance_mask,
            batch.speaker_ids_padded,
            batch.emotion_ids_padded,
        )

        # 1. GNN
        h_graph = self.structural_tower(h_text, batch)

        # 2. Fusion
        h_final = torch.cat([h_graph, h_text], dim=-1)

        # 3. dense batch
        h_dense, mask = to_dense_batch(h_final, batch.batch)

        B = h_dense.size(0)
        batch_idx = torch.arange(B, device=h_dense.device)

        cause_idx = batch.target_node_indices[:, 0]
        target_idx = batch.target_node_indices[:, 1]

        h_c = h_dense[batch_idx, cause_idx]   # [B, 2H]
        h_t = h_dense[batch_idx, target_idx]  # [B, 2H]

        # -------------------------------------------------
        # Student representation
        # -------------------------------------------------
        h_pair = torch.cat([h_c, h_t], dim=-1)  # [B, 4H]
        z_student = self.student_expl_proj(h_pair)  # [B, H]

        # -------------------------------------------------
        # Teacher representation + Gate
        # -------------------------------------------------
        z_teacher = None
        gate = None
        
        # 預設不變 (測試時若無 teacher)
        h_c_gated = h_c
        h_t_gated = h_t

        if self.use_expl_space and hasattr(batch, "expl_space_vec"):
            vec = batch.expl_space_vec.to(h_final.device)
            vec = vec.view(-1, self.expl_space_dim)
            
            # Teacher
            e_space = self.expl_space_mlp(vec)
            z_teacher = e_space

            # Gate
            gate = self.gate_mlp(e_space)  # [B, H]

            # Apply Gate (H -> 2H broadcast)
            gate_expanded = torch.cat([gate, gate], dim=-1)
            h_c_gated = h_c * gate_expanded
            h_t_gated = h_t * gate_expanded

        # -------------------------------------------------
        # Concat for prediction
        # -------------------------------------------------
        feat_list = [h_c_gated, h_t_gated]

        if self.use_explain and hasattr(batch, "edge_features"):
            edge_batch = batch.batch[batch.edge_index[0]]
            e_feat, _ = scatter_max(batch.edge_features, edge_batch, dim=0)
            e_edge = self.edge_explain_mlp(e_feat)
            feat_list.append(e_edge)

        edge_repr = torch.cat(feat_list, dim=-1)
        logits = self.predictor(edge_repr).squeeze(-1)

        # [新增]: 監測模式回傳內部數據
        if return_aux:
            return logits, {
                "z_student": z_student,
                "z_teacher": z_teacher,
                "gate": gate,  # 可能為 None
            }

        # 訓練模式回傳 tuple
        if self.training and self.use_expl_space and (z_teacher is not None):
            return logits, z_student, z_teacher
            
        return logits