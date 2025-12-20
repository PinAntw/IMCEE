# Research/IMCEE/modules/crossTower/text_encoder.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextNodeEncoder(nn.Module):
    """
    Early Fusion Document Encoder (CFIB-Style + Token Injection)
    + SuperNode Safe Init + (Optional) MPEG-style init for cause/target + conv mean init

    Node ordering per graph:
      0..num_utts-1 : utterance nodes
      num_utts      : conv node
      num_utts+1    : cause super node
      num_utts+2    : target super node

    Collate inputs:
      input_ids_padded      [B, S_nodes, T]
      token_mask_padded     [B, S_nodes, T]
      utterance_mask        [B, S_nodes]   (實際為 node_mask)
      speaker_ids_padded    [B, S_nodes]
      emotion_ids_padded    [B, S_nodes]
      pair_utt_index        [B, 2]         (c_idx, t_idx) in utterance-local indices

    Key behavior:
      - Only utterance nodes run PLM (with speaker/emotion token injection)
      - Super nodes do NOT run PLM
      - Super nodes default init: learnable embeddings (conv/cause/target)
      - If pair_utt_index provided: cause/target init from corresponding utterance vectors (MPEG-style)
      - conv init (recommended): mean of valid utterance vectors (if any), else fallback to learnable conv
      - Robust truncation: truncate to safe_len then pad back + sync mask to avoid pooling on padded zeros
      - Output aligned with PyG Batch node ordering (do not drop nodes)
    """

    def __init__(
        self,
        text_model_name,
        num_speakers,
        num_emotions,
        spk_dim=None,  # keep for compatibility (unused)
        emo_dim=None,  # keep for compatibility (unused)
        freeze_text=True,
        dropout=0.1,
        num_heads=8,
    ):
        super().__init__()

        print(f"[TextNodeEncoder] Loading PLM: {text_model_name}...")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.d_sem = int(self.text_encoder.config.hidden_size)
        print(f"[TextNodeEncoder] Detected hidden_size: {self.d_sem}")

        if freeze_text:
            print("[TextNodeEncoder] Freezing PLM parameters.")
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Speaker / Emotion token injection embedding (must match hidden_size)
        self.spk_embed = nn.Embedding(num_speakers, self.d_sem)
        self.emo_embed = nn.Embedding(num_emotions, self.d_sem)

        # CFIB-style attention pooling (token-level -> utterance vector)
        self.attn_score = nn.Linear(self.d_sem, 1)

        # utterance-level MHSA refine (only on utterances)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.d_sem,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.d_sem)
        self.dropout = nn.Dropout(dropout)

        # Learnable fallback init for 3 super nodes: conv/cause/target
        # 0=conv, 1=cause, 2=target
        self.super_node_embed = nn.Embedding(3, self.d_sem)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    @staticmethod
    def _ensure_bool_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        return mask > 0

    def _get_safe_len(self) -> int:
        max_pos = int(getattr(self.text_encoder.config, "max_position_embeddings", 512))
        model_type = str(getattr(self.text_encoder.config, "model_type", "")).lower()
        offset = 2 if model_type in {"roberta", "xlm-roberta"} else 0
        return max(1, max_pos - offset)

    def _attention_pool(self, hidden: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        """
        hidden:    [B, U, L, d]
        mask_bool: [B, U, L]  True=有效 token
        return:    [B, U, d]
        """
        attn_logits = self.attn_score(hidden)  # [B,U,L,1]
        attn_logits = attn_logits.masked_fill(~mask_bool.unsqueeze(-1), -1e9)
        alpha = F.softmax(attn_logits, dim=2)  # [B,U,L,1]
        pooled = torch.sum(alpha * hidden, dim=2)  # [B,U,d]
        return pooled

    def _plm_forward_with_trunc_and_mask_sync(
        self,
        flat_inputs_embeds: torch.Tensor,  # [B, total_len, d]
        flat_mask_bool: torch.Tensor,      # [B, total_len] bool
    ):
        """
        If total_len > safe_len: feed first safe_len to PLM.
        If padding back: pad outputs with zeros AND pad mask with False.
        """
        B, total_len, D = flat_inputs_embeds.shape
        safe_len = self._get_safe_len()
        kept_len = min(total_len, safe_len)

        embeds_trunc = flat_inputs_embeds[:, :kept_len, :]
        mask_trunc_bool = flat_mask_bool[:, :kept_len]
        attention_mask_trunc = mask_trunc_bool.to(dtype=torch.long)

        outputs_trunc = self.text_encoder(
            inputs_embeds=embeds_trunc,
            attention_mask=attention_mask_trunc,
        )[0]  # [B, kept_len, d]

        if kept_len == total_len:
            return outputs_trunc, mask_trunc_bool, kept_len

        pad_len = total_len - kept_len
        pad_hidden = torch.zeros(B, pad_len, D, device=outputs_trunc.device, dtype=outputs_trunc.dtype)
        outputs_padded = torch.cat([outputs_trunc, pad_hidden], dim=1)

        pad_mask = torch.zeros(B, pad_len, device=outputs_trunc.device, dtype=torch.bool)
        mask_padded = torch.cat([mask_trunc_bool, pad_mask], dim=1)

        return outputs_padded, mask_padded, kept_len

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        input_ids_padded,      # [B, S_nodes, T]
        token_mask_padded,     # [B, S_nodes, T]
        utterance_mask,        # [B, S_nodes]  (node_mask)
        speaker_ids_padded,    # [B, S_nodes]
        emotion_ids_padded,    # [B, S_nodes]
        pair_utt_index=None,   # [B, 2] (c_idx, t_idx) in utterance-local indices
    ):
        """
        return:
          h_text_nodes: [Total_Nodes, d_sem]
        """
        B, S_nodes, T = input_ids_padded.shape
        node_mask_bool = self._ensure_bool_mask(utterance_mask)      # [B,S_nodes] node exists mask
        token_mask_bool = self._ensure_bool_mask(token_mask_padded)  # [B,S_nodes,T]

        # ------------------------------------------------------------
        # 0) Infer per-graph utterance count (valid_nodes = num_utts + 3)
        # ------------------------------------------------------------
        valid_nodes = node_mask_bool.sum(dim=1)            # [B]
        num_utts = (valid_nodes - 3).clamp(min=0)          # [B]
        max_utts = int(num_utts.max().item()) if B > 0 else 0

        # utterance-slot mask (only first num_utts are utterances)
        utt_mask_full = torch.zeros(B, S_nodes, device=node_mask_bool.device, dtype=torch.bool)
        for i in range(B):
            u = int(num_utts[i].item())
            if u > 0:
                utt_mask_full[i, :u] = True

        # ------------------------------------------------------------
        # 1) Allocate node container (utter + super)
        # ------------------------------------------------------------
        h_nodes_padded = torch.zeros(B, S_nodes, self.d_sem, device=input_ids_padded.device, dtype=torch.float)

        # Keep references for MPEG init / conv mean
        h_final_utt = None
        utt_mask_effective = None  # [B, U] bool

        # ------------------------------------------------------------
        # 2) Utterance nodes: run PLM with injection + doc flatten
        # ------------------------------------------------------------
        if max_utts > 0:
            utt_input_ids = input_ids_padded[:, :max_utts, :]             # [B,U,T]
            utt_token_mask = token_mask_bool[:, :max_utts, :]             # [B,U,T]
            utt_speaker_ids = speaker_ids_padded[:, :max_utts]            # [B,U]
            utt_emotion_ids = emotion_ids_padded[:, :max_utts]            # [B,U]
            utt_mask_batch = utt_mask_full[:, :max_utts]                  # [B,U]

            # word embeddings
            word_embeddings_layer = self.text_encoder.get_input_embeddings()
            word_embs = word_embeddings_layer(utt_input_ids)              # [B,U,T,d]

            # speaker/emotion injection
            spk_embs = self.spk_embed(utt_speaker_ids).unsqueeze(2)       # [B,U,1,d]
            emo_embs = self.emo_embed(utt_emotion_ids).unsqueeze(2)       # [B,U,1,d]

            # [CLS] + [SPK] + [EMO] + rest...
            cls_token_emb = word_embs[:, :, 0:1, :]                       # [B,U,1,d]
            rest_tokens_emb = word_embs[:, :, 1:, :]                      # [B,U,T-1,d]
            mixed_embs = torch.cat([cls_token_emb, spk_embs, emo_embs, rest_tokens_emb], dim=2)  # [B,U,T+2,d]

            # mask sync
            extra_mask = torch.ones(B, max_utts, 2, device=utt_token_mask.device, dtype=torch.bool)
            cls_mask = utt_token_mask[:, :, 0:1]
            rest_mask = utt_token_mask[:, :, 1:]
            mixed_mask_bool = torch.cat([cls_mask, extra_mask, rest_mask], dim=2)  # [B,U,T+2]

            # slots not existing => all False
            mixed_mask_bool = mixed_mask_bool & utt_mask_batch.unsqueeze(-1)

            # flatten to long doc
            L = T + 2
            total_len = max_utts * L
            flat_inputs_embeds = mixed_embs.reshape(B, total_len, self.d_sem)
            flat_mask_bool = mixed_mask_bool.reshape(B, total_len)

            outputs_long, flat_mask_synced_bool, _ = self._plm_forward_with_trunc_and_mask_sync(
                flat_inputs_embeds, flat_mask_bool
            )  # [B,total_len,d], [B,total_len]

            # unflatten
            outputs_reshaped = outputs_long.view(B, max_utts, L, self.d_sem)      # [B,U,L,d]
            mask_reshaped_bool = flat_mask_synced_bool.view(B, max_utts, L)       # [B,U,L]

            # utterance effective mask: exist & has any token after truncation
            has_any_token = mask_reshaped_bool.any(dim=2)                         # [B,U]
            utt_mask_effective = utt_mask_batch & has_any_token                   # [B,U]

            # token-level attention pooling
            h_text_utt = self._attention_pool(outputs_reshaped, mask_reshaped_bool)  # [B,U,d]

            # utterance-level MHSA (mask invalid slots)
            key_padding_mask = ~utt_mask_effective  # True=mask out
            attn_out, _ = self.mhsa(
                query=h_text_utt,
                key=h_text_utt,
                value=h_text_utt,
                key_padding_mask=key_padding_mask,
            )
            h_final_utt = self.norm(h_text_utt + self.dropout(attn_out))          # [B,U,d]

            # write back utterance region (keep non-existing slots at zero)
            h_nodes_padded[:, :max_utts, :] = h_final_utt
            h_nodes_padded[:, :max_utts, :] = h_nodes_padded[:, :max_utts, :] * utt_mask_batch.unsqueeze(-1)

        # ------------------------------------------------------------
        # 3) Super nodes init
        #   - default: learnable embeddings
        #   - MPEG-style: cause/target init from utterance vectors (if pair_utt_index provided & valid)
        #   - conv init: mean of valid utterance vectors (if any), else fallback learnable conv
        # ------------------------------------------------------------
        for i in range(B):
            u = int(num_utts[i].item())
            conv_pos = u
            cause_pos = u + 1
            target_pos = u + 2

            # (a) default fallback init
            if 0 <= conv_pos < S_nodes and node_mask_bool[i, conv_pos]:
                h_nodes_padded[i, conv_pos] = self.super_node_embed.weight[0]   # conv
            if 0 <= cause_pos < S_nodes and node_mask_bool[i, cause_pos]:
                h_nodes_padded[i, cause_pos] = self.super_node_embed.weight[1]  # cause
            if 0 <= target_pos < S_nodes and node_mask_bool[i, target_pos]:
                h_nodes_padded[i, target_pos] = self.super_node_embed.weight[2] # target

            # nothing to do if no utterances encoded
            if h_final_utt is None or utt_mask_effective is None:
                continue

            # (b) MPEG-style cause/target init from utterances (requires pair_utt_index)
            if pair_utt_index is not None and pair_utt_index.numel() >= (B * 2):
                c_idx = int(pair_utt_index[i, 0].item())
                t_idx = int(pair_utt_index[i, 1].item())

                # cause node init
                if 0 <= c_idx < max_utts and utt_mask_effective[i, c_idx]:
                    if 0 <= cause_pos < S_nodes and node_mask_bool[i, cause_pos]:
                        h_nodes_padded[i, cause_pos] = h_final_utt[i, c_idx]

                # target node init
                if 0 <= t_idx < max_utts and utt_mask_effective[i, t_idx]:
                    if 0 <= target_pos < S_nodes and node_mask_bool[i, target_pos]:
                        h_nodes_padded[i, target_pos] = h_final_utt[i, t_idx]

            # (c) conv init = mean of valid utterances (recommended)
            if 0 <= conv_pos < S_nodes and node_mask_bool[i, conv_pos]:
                valid = utt_mask_effective[i]  # [U]
                if valid.any():
                    h_nodes_padded[i, conv_pos] = h_final_utt[i, valid].mean(dim=0)
                # else keep fallback learnable

        # ------------------------------------------------------------
        # 4) Flatten to [Total_Nodes, d] aligned with PyG Batch ordering
        # ------------------------------------------------------------
        h_text_nodes = h_nodes_padded[node_mask_bool]  # [Total_Nodes, d]
        return h_text_nodes
