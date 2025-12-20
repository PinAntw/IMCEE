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
        self.expl_dim = expl_dim  # for edge_explain_mlp

        # 0. Text Encoder
        self.text_encoder = TextNodeEncoder(
            text_model_name=text_model_name,
            freeze_text=freeze_text,
            num_speakers=num_speakers,
            num_emotions=num_emotions,
            dropout=dropout,
        )
        d_sem = self.text_encoder.d_sem

        # 1. Semantic Tower (你目前 forward 沒用到，但先保留以相容舊結構)
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

        # 3. Explicit Distance Embedder
        self.dist_embedder = nn.Embedding(32, hidden_dim)

        # 4. Student projector (for distillation loss)
        self.student_expl_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5. Teacher projector
        if self.use_expl_space:
            self.expl_space_mlp = nn.Sequential(
                nn.Linear(expl_space_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.expl_space_mlp = None

        # 5.5 [FIX] edge_explain_mlp 在 forward 有用到，必須定義
        # batch.edge_features: [E, expl_dim] -> aggregate -> [B, expl_dim] -> map -> [B, H]
        if self.use_explain:
            self.edge_explain_mlp = nn.Sequential(
                nn.Linear(expl_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.edge_explain_mlp = None

        # 6. Predictor
        base_dim = hidden_dim * 4
        dist_dim = hidden_dim
        teacher_dim = hidden_dim if self.use_expl_space else 0
        edge_expl_dim = hidden_dim if self.use_explain else 0

        pred_in_dim = base_dim + dist_dim + teacher_dim + edge_expl_dim

        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch, return_aux=False):
        device = batch.edge_index.device

        # -------------------------------------------------
        # 0) Text Encoding (+ pass pair_utt_index for MPEG init)
        # -------------------------------------------------
        pair_idx = None
        if hasattr(batch, "pair_utt_index") and batch.pair_utt_index is not None:
            pair_idx = batch.pair_utt_index
        elif hasattr(batch, "pair_uttpos") and batch.pair_uttpos is not None:
            pos = batch.pair_uttpos
            pair_idx = pos.view(-1, 2) if pos.dim() == 3 else pos

        if pair_idx is not None:
            pair_idx = pair_idx.to(device=device, dtype=torch.long)

        h_text = self.text_encoder(
            batch.input_ids, batch.token_mask, batch.utterance_mask,
            batch.speaker_ids_padded, batch.emotion_ids_padded,
            pair_utt_index=pair_idx,
        )

        # -------------------------------------------------
        # 1) GNN Encoding
        # -------------------------------------------------
        h_graph = self.structural_tower(h_text, batch)

        # -------------------------------------------------
        # 2) Dense Batching
        # -------------------------------------------------
        h_final = torch.cat([h_graph, h_text], dim=-1)  # [Nodes, 2H]
        h_dense, _ = to_dense_batch(h_final, batch.batch)

        B = h_dense.size(0)
        batch_idx = torch.arange(B, device=h_dense.device)

        # target_node_indices 是 local index（每張圖內），to_dense_batch 後可直接用
        cause_idx = batch.target_node_indices[:, 0].to(device=h_dense.device)
        target_idx = batch.target_node_indices[:, 1].to(device=h_dense.device)

        h_c = h_dense[batch_idx, cause_idx]   # [B, 2H]
        h_t = h_dense[batch_idx, target_idx]  # [B, 2H]

        # -------------------------------------------------
        # A) Student repr
        # -------------------------------------------------
        h_pair = torch.cat([h_c, h_t], dim=-1)        # [B, 4H]
        z_student = self.student_expl_proj(h_pair)    # [B, H]

        # -------------------------------------------------
        # B) Teacher repr
        # -------------------------------------------------
        z_teacher = None
        if self.use_expl_space and hasattr(batch, "expl_space_vec") and batch.expl_space_vec is not None:
            vec = batch.expl_space_vec.to(h_final.device).view(-1, self.expl_space_dim)
            z_teacher = self.expl_space_mlp(vec)  # [B, H]

        # -------------------------------------------------
        # C) Distance repr (prefer pair_uttpos; fallback fixed for utt->super Type0)
        # -------------------------------------------------
        with torch.no_grad():
            if hasattr(batch, "pair_uttpos") and batch.pair_uttpos is not None:
                pos = batch.pair_uttpos
                pos = pos.view(-1, 2) if pos.dim() == 3 else pos
                pos = pos.to(device=h_final.device, dtype=torch.long)

                c_utt_local = pos[:, 0]
                t_utt_local = pos[:, 1]
                dist_tensor = (c_utt_local - t_utt_local).abs().clamp(max=31).long()

            else:
                # Fallback：用 Type0 邊找 anchor utterance
                # [FIX] 你目前 dataset Type0 是 utt -> super
                c_nodes_local = batch.target_node_indices[:, 0].to(h_final.device)  # [B]
                t_nodes_local = batch.target_node_indices[:, 1].to(h_final.device)  # [B]

                if not hasattr(batch, "ptr"):
                    raise RuntimeError("PyG Batch has no 'ptr'. Cannot compute global offsets for distance fallback.")

                offsets = batch.ptr[:-1].to(h_final.device)  # [B]
                c_nodes_global = c_nodes_local + offsets
                t_nodes_global = t_nodes_local + offsets

                edge_src = batch.edge_index[0].to(h_final.device)  # global
                edge_tgt = batch.edge_index[1].to(h_final.device)  # global
                edge_types = batch.edge_types.to(h_final.device) if hasattr(batch, "edge_types") else None

                batch_dists = []
                for i in range(B):
                    cg = c_nodes_global[i]
                    tg = t_nodes_global[i]
                    off = offsets[i]

                    if edge_types is not None:
                        # Type0: utt -> super，所以要用 edge_tgt == super 找 utt (edge_src)
                        mask_c = (edge_types == 0) & (edge_tgt == cg)
                        mask_t = (edge_types == 0) & (edge_tgt == tg)
                    else:
                        mask_c = (edge_tgt == cg)
                        mask_t = (edge_tgt == tg)

                    if mask_c.any():
                        c_utt_global = edge_src[mask_c][0]
                        c_utt_local = (c_utt_global - off).clamp(min=0)
                    else:
                        c_utt_local = c_nodes_local[i].clamp(min=0)

                    if mask_t.any():
                        t_utt_global = edge_src[mask_t][0]
                        t_utt_local = (t_utt_global - off).clamp(min=0)
                    else:
                        t_utt_local = t_nodes_local[i].clamp(min=0)

                    dist = (c_utt_local - t_utt_local).abs().clamp(max=31).long()
                    batch_dists.append(dist)

                dist_tensor = torch.stack(batch_dists).to(h_final.device)

        h_dist = self.dist_embedder(dist_tensor)  # [B, H]

        # -------------------------------------------------
        # D) Feature Injection
        # -------------------------------------------------
        feat_list = [h_c, h_t, h_dist]

        if z_teacher is not None:
            feat_list.append(z_teacher)

        # Old Edge Explain (aggregate edge_features -> B)
        if self.use_explain and hasattr(batch, "edge_features") and batch.edge_features is not None:
            # edge_batch: per-edge -> which graph in batch
            edge_batch = batch.batch[batch.edge_index[0]]
            e_feat, _ = scatter_max(batch.edge_features, edge_batch, dim=0)  # [B, expl_dim]
            e_edge = self.edge_explain_mlp(e_feat)                           # [B, H]
            feat_list.append(e_edge)

        edge_repr = torch.cat(feat_list, dim=-1)
        logits = self.predictor(edge_repr).squeeze(-1)

        # -------------------------------------------------
        # Return
        # -------------------------------------------------
        if return_aux:
            return logits, {
                "z_student": z_student,
                "z_teacher": z_teacher,
                "dist": dist_tensor,
            }

        if self.training and self.use_expl_space and (z_teacher is not None):
            return logits, z_student, z_teacher

        return logits
