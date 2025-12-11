# from .text_encoder import TextNodeEncoder
# from .semantic_tower import SemanticTower
# from .structural_tower import StructuralGraphTower

# from torch_scatter import scatter_max
# from torch_geometric.utils import to_dense_batch
# import torch
# import torch.nn as nn


# class CrossTowerCausalModel(nn.Module):
#     def __init__(
#         self,
#         text_model_name,
#         hidden_dim,
#         expl_dim,
#         num_speakers,
#         num_emotions,
#         dropout=0.1,
#         gnn_dropout=0.3,
#         num_gnn_layers=3,
#         freeze_text=True,
#         use_explain=True,       # edge-level explain (原本的)
#         expl_space_dim=0,       # explanation space 維度
#         use_expl_space=False,   # 是否使用 explanation space
#     ):
#         super().__init__()

#         # -------------------------------------------------
#         # 開關與維度
#         # -------------------------------------------------
#         self.hidden_dim = hidden_dim

#         # edge-level explain（原本機制）
#         self.use_explain = use_explain and (expl_dim is not None) and (expl_dim > 0)
#         self.expl_dim = expl_dim if expl_dim is not None else 0

#         # explanation space（新的 global explain 機制）
#         self.use_expl_space = (
#             use_expl_space and (expl_space_dim is not None) and (expl_space_dim > 0)
#         )
#         self.expl_space_dim = expl_space_dim if expl_space_dim is not None else 0

#         # -------------------------------------------------
#         # 0. Text Encoder
#         # -------------------------------------------------
#         self.text_encoder = TextNodeEncoder(
#             text_model_name=text_model_name,
#             freeze_text=freeze_text,
#             num_speakers=num_speakers,
#             num_emotions=num_emotions,
#             dropout=dropout,
#         )
#         d_sem = self.text_encoder.d_sem  # PLM 輸出維度

#         # -------------------------------------------------
#         # 1. Semantic Tower
#         # -------------------------------------------------
#         self.semantic_tower = SemanticTower(
#             in_dim=d_sem,
#             out_dim=hidden_dim,
#             expl_dim=(self.expl_dim if self.use_explain else 0),
#             dropout=dropout,
#         )

#         # -------------------------------------------------
#         # 2. Structural Tower (Graph)
#         # -------------------------------------------------
#         self.structural_tower = StructuralGraphTower(
#             hidden_dim=hidden_dim,
#             num_gnn_layers=num_gnn_layers,
#             dropout=gnn_dropout,
#         )

#         # -------------------------------------------------
#         # 3. Fusion (graph + text)
#         # -------------------------------------------------
#         # 目前直接 concat graph/text 做 skip connection
#         self.concat_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # -------------------------------------------------
#         # 4. edge-level explain MLP（原本的）
#         # -------------------------------------------------
#         if self.use_explain:
#             self.edge_explain_mlp = nn.Sequential(
#                 nn.Linear(self.expl_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.edge_explain_mlp = None

#         # -------------------------------------------------
#         # 5. explanation space MLP（新的）
#         # -------------------------------------------------
#         if self.use_expl_space:
#             self.expl_space_mlp = nn.Sequential(
#                 nn.Linear(self.expl_space_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.expl_space_mlp = None
#         # 最新的：5.5 Student explanation head：從 [h_c; h_t] 映射到 H 維
#         self.student_expl_proj = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim),  # h_c(2H) + h_t(2H) = 4H
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         # -------------------------------------------------
#         # 6. Predictor input 維度計算
#         # -------------------------------------------------
#         # === 正確計算 predictor 的輸入維度 ===
#         base_dim = hidden_dim * 4              # h_c(2H) + h_t(2H)
#         extra_dim = 0
#         if self.use_explain:
#             extra_dim += hidden_dim            # e_edge 是 H 維
#         if self.use_expl_space:
#             extra_dim += hidden_dim            # e_space 是 H 維

#         pred_in_dim = base_dim + extra_dim

#         # 6. Predictor
#         self.predictor = nn.Sequential(
#             nn.Linear(pred_in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     # -----------------------------------------------------
#     # Forward
#     # -----------------------------------------------------
#     def forward(self, batch):
#         """
#         batch:
#           - input_ids: [B, S, T]
#           - token_mask: [B, S, T]
#           - utterance_mask: [B, S]
#           - speaker_ids_padded: [B, S]
#           - emotion_ids_padded: [B, S]
#           - edge_index: [2, E]
#           - edge_features: [E, expl_dim]（只有部份邊有非零）
#           - edge_label: [B]
#           - target_node_indices: [B, 2] (cause_node, target_node)
#           - expl_space_vec: [B, expl_space_dim]（新的 explanation space 向量）
#         """

#         # 0. Text Encoder
#         h_text = self.text_encoder(
#             batch.input_ids,
#             batch.token_mask,
#             batch.utterance_mask,
#             batch.speaker_ids_padded,
#             batch.emotion_ids_padded,
#         )  # [Total_Nodes, hidden_dim]

#         # 1. Structural encoder (Graph)
#         h_graph = self.structural_tower(h_text, batch)  # [Total_Nodes, hidden_dim]

#         # 2. Skip connection fusion: [graph; text]
#         h_final = torch.cat([h_graph, h_text], dim=-1)  # [Total_Nodes, 2*hidden_dim]

#         # 3. 還原成 [B, Max_Nodes, 2*hidden_dim]
#         h_dense, mask = to_dense_batch(h_final, batch.batch)  # h_dense: [B, S_max, 2*hidden_dim]

#         cause_idx = batch.target_node_indices[:, 0]  # [B]
#         target_idx = batch.target_node_indices[:, 1]  # [B]
#         batch_indices = torch.arange(h_dense.size(0), device=h_dense.device)

#         # h_c, h_t: [B, 2*hidden_dim]
#         h_c = h_dense[batch_indices, cause_idx]
#         h_t = h_dense[batch_indices, target_idx]

#         feat_list = [h_c, h_t]  # 目前為 4 * hidden_dim
#         z_student = self.student_expl_proj(torch.cat([h_c, h_t], dim=-1))
#         z_teacher = None

#         # 4. edge-level explain（原本邊上的 explain 向量）
#         if self.use_explain and hasattr(batch, "edge_features"):
#             # edge_index[0] 是 source node，利用它對應回 batch id
#             edge_batch_id = batch.batch[batch.edge_index[0]]
#             # scatter_max: [E, expl_dim] -> [B, expl_dim]
#             e_feat, _ = scatter_max(batch.edge_features, edge_batch_id, dim=0)
#             e_edge = self.edge_explain_mlp(e_feat)  # [B, hidden_dim]
#             feat_list.append(e_edge)

#         # 5. explanation space（新的全域解釋向量）
#         if self.use_expl_space and hasattr(batch, "expl_space_vec"):
#             # PyG 會把每個 graph 的 1D 向量串成一條 1D：長度 = B * expl_space_dim
#             expl_space_vec = batch.expl_space_vec.to(h_final.device)  # [B * expl_space_dim]

#             # 依照我們在 __init__ 設的 expl_space_dim 還原成 [B, expl_space_dim]
#             expl_space_vec = expl_space_vec.view(-1, self.expl_space_dim)  # [B, expl_space_dim]

#             e_space = self.expl_space_mlp(expl_space_vec)  # [B, hidden_dim]
#             feat_list.append(e_space)
#             z_teacher = e_space  # [B, hidden_dim]


#         # 6. 串接所有特徵
#         edge_repr = torch.cat(feat_list, dim=-1)  # [B, pred_in_dim]

#         logits = self.predictor(edge_repr).squeeze(-1)  # [B]

#         # 最新的訓練時（且有 teacher explain）回傳 (logits, z_student, z_teacher)
#         if self.training and self.use_expl_space and (z_teacher is not None):
#             return logits, z_student, z_teacher
#         else:
#             # 評估 / 測試保持舊介面：只回 logits，evaluate() 不用改
#             return logits

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
# # Research/IMCEE/modules/crossTower/cross_tower_model.py

# from .text_encoder import TextNodeEncoder
# from .semantic_tower import SemanticTower
# from .structural_tower import StructuralGraphTower

# from torch_scatter import scatter_max
# from torch_geometric.utils import to_dense_batch
# import torch
# import torch.nn as nn


# class CrossTowerCausalModel(nn.Module):
#     def __init__(
#         self,
#         text_model_name,
#         hidden_dim,
#         expl_dim,
#         num_speakers,
#         num_emotions,
#         dropout=0.1,
#         gnn_dropout=0.3,
#         num_gnn_layers=3,
#         freeze_text=True,
#         use_explain=True,       # edge-level explain
#         expl_space_dim=0,       # explanation space 維度 (GPT explain space)
#         use_expl_space=False,   # 是否使用 explanation space
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim

#         # edge-level explain（原本機制，用在 e_edge）
#         self.use_explain = use_explain and (expl_dim is not None) and (expl_dim > 0)
#         self.expl_dim = expl_dim if expl_dim is not None else 0

#         # explanation space（LLM explain space，用在 teacher + residual）
#         self.use_expl_space = (
#             use_expl_space and (expl_space_dim is not None) and (expl_space_dim > 0)
#         )
#         self.expl_space_dim = expl_space_dim if expl_space_dim is not None else 0

#         # -------------------------------------------------
#         # 0. Text Encoder
#         # -------------------------------------------------
#         self.text_encoder = TextNodeEncoder(
#             text_model_name=text_model_name,
#             freeze_text=freeze_text,
#             num_speakers=num_speakers,
#             num_emotions=num_emotions,
#             dropout=dropout,
#         )
#         d_sem = self.text_encoder.d_sem  # PLM 輸出維度（通常 = hidden_dim）

#         # -------------------------------------------------
#         # 1. Semantic Tower（目前保留，不一定使用）
#         #    如果之後要在 encoder 後再做一層投影可以用
#         # -------------------------------------------------
#         self.semantic_tower = SemanticTower(
#             in_dim=d_sem,
#             out_dim=hidden_dim,
#             expl_dim=(self.expl_dim if self.use_explain else 0),
#             dropout=dropout,
#         )

#         # -------------------------------------------------
#         # 2. Structural Tower (Graph)
#         # -------------------------------------------------
#         self.structural_tower = StructuralGraphTower(
#             hidden_dim=hidden_dim,
#             num_gnn_layers=num_gnn_layers,
#             dropout=gnn_dropout,
#         )

#         # -------------------------------------------------
#         # 3. edge-level explain MLP（原本的 e_edge）
#         # -------------------------------------------------
#         if self.use_explain:
#             self.edge_explain_mlp = nn.Sequential(
#                 nn.Linear(self.expl_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.edge_explain_mlp = None

#         # -------------------------------------------------
#         # 4. explanation space MLP（Teacher：LLM explain space → H）
#         # -------------------------------------------------
#         if self.use_expl_space:
#             self.expl_space_mlp = nn.Sequential(
#                 nn.Linear(self.expl_space_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.expl_space_mlp = None

#         # -------------------------------------------------
#         # 5. Student explanation head：z_student (backbone 語意表示)
#         #    只看「原始 text 的 h_c_text, h_t_text」
#         #    Input: [h_c_text; h_t_text] ∈ R^{2H} → R^{H}
#         # -------------------------------------------------
#         self.student_expl_proj = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )


#         # -------------------------------------------------
#         # 6. Backbone：純 text-based pair 表徵
#         #    z_base = f_base([h_c_text; h_t_text]) ∈ R^{H}
#         # -------------------------------------------------
#         self.base_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # -------------------------------------------------
#         # 7. Structural residual：利用「GNN - Text」的差異當修正項
#         #    Δ_struct = f_struct([h_c_gnn - h_c_text; h_t_gnn - h_t_text]) ∈ R^{H}
#         # -------------------------------------------------
#         self.struct_delta_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # -------------------------------------------------
#         # 8. Semantic residual：student / edge explain / explain space
#         #    Δ_sem = f_sem([z_student; e_edge?; e_space?]) ∈ R^{H}
#         # -------------------------------------------------
#         sem_in_dim = hidden_dim  # z_student
#         # if self.use_explain:
#         #     sem_in_dim += hidden_dim  # e_edge
#         # if self.use_expl_space:
#         #     sem_in_dim += hidden_dim  # e_space

#         self.sem_delta_mlp = nn.Sequential(
#             nn.Linear(sem_in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # -------------------------------------------------
#         # 9. Residual scalar：學兩個 scalar α, β，控制兩個 residual 比重
#         #    α = σ(a), β = σ(b) ∈ (0,1)
#         # -------------------------------------------------
#         self.alpha_param = nn.Parameter(torch.zeros(1))  # for struct residual
#         self.beta_param = nn.Parameter(torch.zeros(1))   # for semantic residual

#         # -------------------------------------------------
#         # 10. Predictor：只吃 z_total ∈ R^{H}
#         # -------------------------------------------------
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     # -----------------------------------------------------
#     # Forward
#     # -----------------------------------------------------
#     def forward(self, batch):
#         """
#         batch:
#           - input_ids: [B, S, T]
#           - token_mask: [B, S, T]
#           - utterance_mask: [B, S]
#           - speaker_ids_padded: [B, S]
#           - emotion_ids_padded: [B, S]
#           - edge_index: [2, E]
#           - edge_features: [E, expl_dim]（只有部份邊有非零）
#           - edge_label: [B]
#           - target_node_indices: [B, 2] (cause_node, target_node)
#           - expl_space_vec: [B, expl_space_dim]（新的 explanation space 向量）
#         """

#         # 0. Text Encoder → node-level text representation
#         #    h_text: [Total_Nodes, H]
#         h_text = self.text_encoder(
#             batch.input_ids,
#             batch.token_mask,
#             batch.utterance_mask,
#             batch.speaker_ids_padded,
#             batch.emotion_ids_padded,
#         )

#         # 1. Structural encoder (Graph GNN) → h_graph
#         #    h_graph: [Total_Nodes, H]
#         h_graph = self.structural_tower(h_text, batch)

#         # 2. 還原成 [B, S_max, H]，分別對 text / graph
#         h_text_dense, _ = to_dense_batch(h_text, batch.batch)    # [B, S, H]
#         h_graph_dense, _ = to_dense_batch(h_graph, batch.batch)  # [B, S, H]

#         # 3. 取出 cause / target node 的 text & graph 向量
#         B = h_text_dense.size(0)
#         batch_indices = torch.arange(B, device=h_text_dense.device)

#         cause_idx = batch.target_node_indices[:, 0]  # [B]
#         target_idx = batch.target_node_indices[:, 1] # [B]

#         # 純 text
#         h_c_text = h_text_dense[batch_indices, cause_idx]   # [B, H]
#         h_t_text = h_text_dense[batch_indices, target_idx]  # [B, H]

#         # GNN 後的結構向量
#         h_c_gnn = h_graph_dense[batch_indices, cause_idx]   # [B, H]
#         h_t_gnn = h_graph_dense[batch_indices, target_idx]  # [B, H]

#         # -------------------------------------------------
#         # 4. Backbone：純 text pair 表徵
#         # -------------------------------------------------
#         # [h_c_text; h_t_text] ∈ R^{2H}
#         pair_text = torch.cat([h_c_text, h_t_text], dim=-1)  # [B, 2H]
#         z_base = self.base_mlp(pair_text)                    # [B, H]

#         # -------------------------------------------------
#         # 5. Structural residual：用 (GNN - Text) 的差當修正訊號
#         # -------------------------------------------------
#         struct_diff_c = h_c_gnn - h_c_text                  # [B, H]
#         struct_diff_t = h_t_gnn - h_t_text                  # [B, H]
#         struct_input = torch.cat([struct_diff_c, struct_diff_t], dim=-1)  # [B, 2H]
#         delta_struct = self.struct_delta_mlp(struct_input)               # [B, H]

#         # -------------------------------------------------
#         # 6. Semantic residual 的各個組件
#         #    - student: z_student（看 text pair）
#         #    - teacher: z_teacher = e_space（LLM explain space）
#         #    - e_edge:   edge-level explain
#         # -------------------------------------------------
#         # 6.1 學生表示：z_student
#         z_student = self.student_expl_proj(pair_text)  # [B, H]

#         # 6.2 老師表示：z_teacher (來自 explanation space)
#         z_teacher = None
#         e_space = None
#         if self.use_expl_space and hasattr(batch, "expl_space_vec"):
#             # batch.expl_space_vec 在建圖時就已經是 [B, D]，這裡保險起見用 view
#             expl_vec = batch.expl_space_vec.to(h_text.device)
#             expl_vec = expl_vec.view(-1, self.expl_space_dim)  # [B, D]
#             e_space = self.expl_space_mlp(expl_vec)            # [B, H]
#             z_teacher = e_space

#         # 6.3 edge-level explain: e_edge
#         e_edge = None
#         if self.use_explain and hasattr(batch, "edge_features"):
#             edge_batch_id = batch.batch[batch.edge_index[0]]     # [E]
#             e_feat, _ = scatter_max(batch.edge_features, edge_batch_id, dim=0)
#             e_edge = self.edge_explain_mlp(e_feat)               # [B, H]

#         # 6.4 組 semantic residual 的輸入
#         sem_feat_list = [z_student]  # 一定有 student
#         # if self.use_explain and (e_edge is not None):
#         #     sem_feat_list.append(e_edge)
#         # if self.use_expl_space and (e_space is not None):
#         #     sem_feat_list.append(e_space)

#         sem_input = torch.cat(sem_feat_list, dim=-1)  # [B, sem_in_dim]
#         delta_sem = self.sem_delta_mlp(sem_input)     # [B, H]

#         # -------------------------------------------------
#         # 7. 組合 residual：z_total = z_base + α * Δ_struct + β * Δ_sem
#         # -------------------------------------------------
#         alpha = torch.sigmoid(self.alpha_param)  # scalar in (0,1)
#         beta = torch.sigmoid(self.beta_param)    # scalar in (0,1)

#         z_total = z_base + alpha * delta_struct + beta * delta_sem  # [B, H]

#         # -------------------------------------------------
#         # 8. Predictor：只看 z_total
#         # -------------------------------------------------
#         logits = self.predictor(z_total).squeeze(-1)  # [B]

#         # -------------------------------------------------
#         # 9. 訓練時，如果有 teacher explain，就回傳 alignment 用的 z_student, z_teacher
#         # -------------------------------------------------
#         if self.training and self.use_expl_space and (z_teacher is not None):
#             return logits, z_student, z_teacher
#         else:
#             return logits

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Research/IMCEE/modules/crossTower/cross_tower_model.py

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch_scatter import scatter_max
# from torch_geometric.utils import to_dense_batch

# from .text_encoder import TextNodeEncoder
# from .semantic_tower import SemanticTower
# from .structural_tower import StructuralGraphTower


# class CrossTowerCausalModel(nn.Module):
#     """
#     CrossTowerCausalModel (Residual-Subtraction Style)
    
#     Academic Design Choices:
#     1. Fusion Strategy: LayerNorm + Subtraction. 
#        We hypothesize that the graph structure acts as a 'correction' signal.
#        LayerNorm ensures the text and graph manifolds are scale-aligned before subtraction.
#     2. Gating: Static Scalar (Alpha/Beta).
#        Parsimonious design to prevent overfitting on small datasets.
#     3. Distillation: Linear Projection (Student) aligned to LLM Space (Teacher).
#     """

#     def __init__(
#         self,
#         text_model_name: str,
#         hidden_dim: int,
#         expl_dim: int,
#         num_speakers: int,
#         num_emotions: int,
#         dropout: float = 0.1,
#         gnn_dropout: float = 0.3,
#         num_gnn_layers: int = 3,
#         freeze_text: bool = True,
#         use_explain: bool = True,        # edge-level explain features
#         expl_space_dim: int = 0,         # explanation space dim (teacher)
#         use_expl_space: bool = False,    # use explanation space (teacher alignment)
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.use_explain = bool(use_explain and (expl_dim is not None) and (expl_dim > 0))
#         self.expl_dim = int(expl_dim) if expl_dim is not None else 0
#         self.use_expl_space = bool(use_expl_space and (expl_space_dim is not None) and (expl_space_dim > 0))
#         self.expl_space_dim = int(expl_space_dim) if expl_space_dim is not None else 0

#         # 0. Text Encoder
#         self.text_encoder = TextNodeEncoder(
#             text_model_name=text_model_name,
#             freeze_text=freeze_text,
#             num_speakers=num_speakers,
#             num_emotions=num_emotions,
#             dropout=dropout,
#         )
#         d_sem = self.text_encoder.d_sem

#         # 1. Semantic Tower (Optional)
#         self.semantic_tower = SemanticTower(
#             in_dim=d_sem,
#             out_dim=hidden_dim,
#             expl_dim=(self.expl_dim if self.use_explain else 0),
#             dropout=dropout,
#         )

#         # 2. Structural Tower (Graph)
#         self.structural_tower = StructuralGraphTower(
#             hidden_dim=hidden_dim,
#             num_gnn_layers=num_gnn_layers,
#             dropout=gnn_dropout,
#         )

#         # 3. edge-level explain MLP
#         if self.use_explain:
#             self.edge_explain_mlp = nn.Sequential(
#                 nn.Linear(self.expl_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.edge_explain_mlp = None

#         # 4. Teacher Projection (LLM Space -> H)
#         if self.use_expl_space:
#             self.expl_space_mlp = nn.Sequential(
#                 nn.Linear(self.expl_space_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.expl_space_mlp = None

#         # 5. Student Projection (Backbone -> H)
#         # ACADEMIC NOTE: Using a single linear layer acts as a regularizer,
#         # forcing the student to learn generalizable features rather than overfitting teacher noise.
#         self.student_expl_proj = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 6. Backbone (Text Pair)
#         self.base_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 7. Structural Residual (LayerNorm + Subtraction)
#         # ACADEMIC NOTE: LayerNorm aligns the scales of graph and text representations
#         # to ensure the subtraction operation (Difference) is mathematically valid.
#         self.struct_norm = nn.LayerNorm(hidden_dim)
        
#         self.struct_delta_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), # Input is [Diff_Cause; Diff_Target]
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 8. Semantic Residual
#         self.sem_delta_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 9. Residual Scalars (Static Gating)
#         # Initialized to 0.0, so Sigmoid(0) = 0.5 (Neutral start)
#         # self.alpha_param = nn.Parameter(torch.zeros(1))  # for structural residual
#         # self.beta_param = nn.Parameter(torch.zeros(1))   # for semantic residual

#         # 9. Residual Scalars
#         # [修改] 初始化為 -2.0，讓 Sigmoid(-2.0) ≈ 0.12
#         # 這樣一開始結構訊號不會太強，避免 "喧賓奪主"
#         self.alpha_param = nn.Parameter(torch.tensor([-2.0])) 
#         self.beta_param = nn.Parameter(torch.tensor([0.0]))   # Beta 維持 0.5 或也可以降

#         # 10. Predictor
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, batch, return_aux: bool = False):
#         # 0) Text Encoder
#         h_text = self.text_encoder(
#             batch.input_ids,
#             batch.token_mask,
#             batch.utterance_mask,
#             batch.speaker_ids_padded,
#             batch.emotion_ids_padded,
#         )

#         # 1) Graph Encoder
#         h_graph = self.structural_tower(h_text, batch)

#         # 2) Dense Batching
#         h_text_dense, _ = to_dense_batch(h_text, batch.batch)
#         h_graph_dense, _ = to_dense_batch(h_graph, batch.batch)

#         # 3) Indices Extraction
#         B = h_text_dense.size(0)
#         batch_indices = torch.arange(B, device=h_text_dense.device)
#         cause_idx = batch.target_node_indices[:, 0]
#         target_idx = batch.target_node_indices[:, 1]

#         h_c_text = h_text_dense[batch_indices, cause_idx]
#         h_t_text = h_text_dense[batch_indices, target_idx]
#         h_c_gnn  = h_graph_dense[batch_indices, cause_idx]
#         h_t_gnn  = h_graph_dense[batch_indices, target_idx]

#         # 4) Backbone (z_base)
#         pair_text = torch.cat([h_c_text, h_t_text], dim=-1)
#         z_base = self.base_mlp(pair_text)

#         # 5) Structural Residual (LayerNorm + Subtraction)
#         # Apply LayerNorm first
#         h_c_gnn_norm = self.struct_norm(h_c_gnn)
#         h_c_text_norm = self.struct_norm(h_c_text)
#         h_t_gnn_norm = self.struct_norm(h_t_gnn)
#         h_t_text_norm = self.struct_norm(h_t_text)

#         # Calculate Difference (Structure Correction)
#         struct_diff_c = h_c_gnn_norm - h_c_text_norm
#         struct_diff_t = h_t_gnn_norm - h_t_text_norm
        
#         struct_input = torch.cat([struct_diff_c, struct_diff_t], dim=-1)
#         delta_struct = self.struct_delta_mlp(struct_input)

#         # 6) Semantic Residual (Distillation)
#         # Student Path
#         z_student = self.student_expl_proj(pair_text)
#         delta_sem = self.sem_delta_mlp(z_student)

#         # Teacher Path (Training / Debug only)
#         z_teacher = None
#         if self.use_expl_space and hasattr(batch, "expl_space_vec") and (batch.expl_space_vec is not None):
#             expl_vec = batch.expl_space_vec.to(h_text.device).view(-1, self.expl_space_dim)
#             z_teacher = self.expl_space_mlp(expl_vec)
        
#         # Edge Explain (Optional)
#         e_edge = None
#         if self.use_explain and hasattr(batch, "edge_features") and (batch.edge_features is not None):
#             edge_batch_id = batch.batch[batch.edge_index[0]]
#             e_feat, _ = scatter_max(batch.edge_features, edge_batch_id, dim=0)
#             e_edge = self.edge_explain_mlp(e_feat)

#         # 7) Fusion
#         alpha = torch.sigmoid(self.alpha_param)
#         beta  = torch.sigmoid(self.beta_param)
        
#         z_total = z_base + alpha * delta_struct + beta * delta_sem

#         # 8) Prediction
#         logits = self.predictor(z_total).squeeze(-1)

#         # 9) Return Logic
#         if return_aux:
#             # Return internal states for deep monitoring
#             return logits, {
#                 "z_base": z_base,
#                 "delta_struct": delta_struct,
#                 "delta_sem": delta_sem,
#                 "z_student": z_student,
#                 "z_teacher": z_teacher,
#                 "alpha": alpha,
#                 "beta": beta
#             }
        
#         if self.training and self.use_expl_space and (z_teacher is not None):
#             return logits, z_student, z_teacher
#         else:
#             return logits

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Research/IMCEE/modules/crossTower/cross_tower_model.py

# # from __future__ import annotations

# import torch
# import torch.nn as nn

# from torch_scatter import scatter_max
# from torch_geometric.utils import to_dense_batch

# from .text_encoder import TextNodeEncoder
# from .semantic_tower import SemanticTower
# from .structural_tower import StructuralGraphTower


# class CrossTowerCausalModel(nn.Module):
#     """
#     CrossTowerCausalModel (Residual-Subtraction Style)

#     Key Fix (Distillation correctness):
#     - Teacher branch is treated as a *fixed target* during distillation.
#       We stop gradients to teacher by returning z_teacher.detach() in forward(),
#       so expl_loss only updates the student side.

#     Optional:
#     - freeze_teacher=True will also freeze expl_space_mlp parameters.
#       This prevents teacher MLP from drifting (even if it could be updated by other losses).
#     """

#     def __init__(
#         self,
#         text_model_name: str,
#         hidden_dim: int,
#         expl_dim: int,
#         num_speakers: int,
#         num_emotions: int,
#         dropout: float = 0.1,
#         gnn_dropout: float = 0.3,
#         num_gnn_layers: int = 3,
#         freeze_text: bool = True,
#         use_explain: bool = True,        # edge-level explain features
#         expl_space_dim: int = 0,         # explanation space dim (teacher)
#         use_expl_space: bool = False,    # use explanation space (teacher alignment)
#         freeze_teacher: bool = True,     # NEW: freeze teacher MLP by default
#     ):
#         super().__init__()

#         self.hidden_dim = int(hidden_dim)

#         # edge-level explain
#         self.use_explain = bool(use_explain and (expl_dim is not None) and (expl_dim > 0))
#         self.expl_dim = int(expl_dim) if expl_dim is not None else 0

#         # explanation space (teacher)
#         self.use_expl_space = bool(use_expl_space and (expl_space_dim is not None) and (expl_space_dim > 0))
#         self.expl_space_dim = int(expl_space_dim) if expl_space_dim is not None else 0

#         # 0) Text Encoder
#         self.text_encoder = TextNodeEncoder(
#             text_model_name=text_model_name,
#             freeze_text=freeze_text,
#             num_speakers=num_speakers,
#             num_emotions=num_emotions,
#             dropout=dropout,
#         )
#         d_sem = self.text_encoder.d_sem

#         # 1) Semantic Tower (kept for compatibility; not necessarily used)
#         self.semantic_tower = SemanticTower(
#             in_dim=d_sem,
#             out_dim=self.hidden_dim,
#             expl_dim=(self.expl_dim if self.use_explain else 0),
#             dropout=dropout,
#         )

#         # 2) Structural Tower (Graph)
#         self.structural_tower = StructuralGraphTower(
#             hidden_dim=self.hidden_dim,
#             num_gnn_layers=num_gnn_layers,
#             dropout=gnn_dropout,
#         )

#         # 3) edge-level explain MLP (optional)
#         if self.use_explain:
#             self.edge_explain_mlp = nn.Sequential(
#                 nn.Linear(self.expl_dim, self.hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.edge_explain_mlp = None

#         # 4) Teacher projection (LLM space -> H)
#         if self.use_expl_space:
#             self.expl_space_mlp = nn.Sequential(
#                 nn.Linear(self.expl_space_dim, self.hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
#         else:
#             self.expl_space_mlp = None

#         # NEW: optionally freeze teacher MLP
#         self.freeze_teacher = bool(freeze_teacher)
#         if self.use_expl_space and self.expl_space_mlp is not None and self.freeze_teacher:
#             for p in self.expl_space_mlp.parameters():
#                 p.requires_grad = False

#         # 5) Student projection (text pair -> H)
#         self.student_expl_proj = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 6) Backbone (text pair)
#         self.base_mlp = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 7) Structural residual (LayerNorm + subtraction)
#         self.struct_norm = nn.LayerNorm(self.hidden_dim)
#         self.struct_delta_mlp = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 8) Semantic residual (from student)
#         self.sem_delta_mlp = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         # 9) Residual scalars
#         # init alpha ~ 0.12, beta ~ 0.5
#         self.alpha_param = nn.Parameter(torch.tensor([0.0]))
#         self.beta_param = nn.Parameter(torch.tensor([0.0]))

#         # 10) Predictor
#         self.predictor = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.hidden_dim, 1),
#         )

#     def forward(self, batch, return_aux: bool = False):
#         """
#         Returns:
#           - logits
#           - (optional) z_student, z_teacher_detached (during training & when teacher exists)
#           - (optional) aux dict (if return_aux=True)
#         """

#         # 0) Text Encoder -> node embeddings
#         h_text = self.text_encoder(
#             batch.input_ids,
#             batch.token_mask,
#             batch.utterance_mask,
#             batch.speaker_ids_padded,
#             batch.emotion_ids_padded,
#         )  # [TotalNodes, H]

#         # 1) Graph Encoder
#         h_graph = self.structural_tower(h_text, batch)  # [TotalNodes, H]

#         # 2) Dense batch
#         h_text_dense, _ = to_dense_batch(h_text, batch.batch)    # [B, S, H]
#         h_graph_dense, _ = to_dense_batch(h_graph, batch.batch)  # [B, S, H]

#         # 3) pick (cause_node, target_node)
#         B = h_text_dense.size(0)
#         bidx = torch.arange(B, device=h_text_dense.device)
#         cause_idx = batch.target_node_indices[:, 0]
#         target_idx = batch.target_node_indices[:, 1]

#         h_c_text = h_text_dense[bidx, cause_idx]
#         h_t_text = h_text_dense[bidx, target_idx]
#         h_c_gnn  = h_graph_dense[bidx, cause_idx]
#         h_t_gnn  = h_graph_dense[bidx, target_idx]

#         # 4) backbone
#         pair_text = torch.cat([h_c_text, h_t_text], dim=-1)  # [B, 2H]
#         z_base = self.base_mlp(pair_text)                    # [B, H]

#         # 5) structural residual (LN + subtraction)
#         h_c_gnn_n  = self.struct_norm(h_c_gnn)
#         h_c_text_n = self.struct_norm(h_c_text)
#         h_t_gnn_n  = self.struct_norm(h_t_gnn)
#         h_t_text_n = self.struct_norm(h_t_text)

#         struct_diff_c = h_c_gnn_n - h_c_text_n
#         struct_diff_t = h_t_gnn_n - h_t_text_n

#         struct_input = torch.cat([struct_diff_c, struct_diff_t], dim=-1)  # [B, 2H]
#         delta_struct = self.struct_delta_mlp(struct_input)                # [B, H]

#         # 6) semantic residual (student)
#         z_student = self.student_expl_proj(pair_text)  # [B, H]
#         delta_sem = self.sem_delta_mlp(z_student)      # [B, H]

#         # 7) teacher (for distillation target only)
#         z_teacher = None
#         if self.use_expl_space and hasattr(batch, "expl_space_vec") and (batch.expl_space_vec is not None):
#             expl_vec = batch.expl_space_vec.to(h_text.device).view(-1, self.expl_space_dim)  # [B, D]
#             # If teacher MLP is frozen, it's still okay to run forward
#             z_teacher = self.expl_space_mlp(expl_vec)  # [B, H]

#         # (optional) edge explain (currently not fused into prediction)
#         e_edge = None
#         if self.use_explain and hasattr(batch, "edge_features") and (batch.edge_features is not None):
#             edge_batch_id = batch.batch[batch.edge_index[0]]  # [E]
#             e_feat, _ = scatter_max(batch.edge_features, edge_batch_id, dim=0)  # [B, expl_dim]
#             e_edge = self.edge_explain_mlp(e_feat)  # [B, H]

#         # 8) fuse
#         alpha = torch.sigmoid(self.alpha_param)  # scalar
#         beta  = torch.sigmoid(self.beta_param)   # scalar

#         z_total = z_base + alpha * delta_struct + beta * delta_sem  # [B, H]

#         # 9) predict
#         logits = self.predictor(z_total).squeeze(-1)  # [B]

#         # 10) return aux for monitoring
#         if return_aux:
#             return logits, {
#                 "z_base": z_base,
#                 "delta_struct": delta_struct,
#                 "delta_sem": delta_sem,
#                 "z_student": z_student,
#                 "z_teacher": z_teacher,
#                 "e_edge": e_edge,
#                 "alpha": alpha.detach(),
#                 "beta": beta.detach(),
#             }

#         # 11) distill return (STOP-GRAD teacher)
#         if self.training and self.use_expl_space and (z_teacher is not None):
#             return logits, z_student, z_teacher.detach()
#         else:
#             return logits
