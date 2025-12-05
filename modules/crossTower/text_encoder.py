# # Research/IMCEE/modules/crossTower/text_encoder.pyＴＣＮ
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel


# class TCNBlock(nn.Module):
#     """
#     一層簡單的 TCN block：
#     - Conv1d (d_model → d_model)
#     - ReLU
#     - Dropout
#     - 殘差 + LayerNorm
#     時間維度用 dilation 擴大 receptive field。
#     """
#     def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation // 2  # 保持長度不變

#         self.conv = nn.Conv1d(
#             in_channels=d_model,
#             out_channels=d_model,
#             kernel_size=kernel_size,
#             padding=padding,
#             dilation=dilation,
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         """
#         x: [B*S, T, d_model]
#         """
#         # 轉成 Conv1d 需求的 [N, C, T]
#         x_in = x.transpose(1, 2)        # [B*S, d_model, T]
#         out = self.conv(x_in)           # [B*S, d_model, T]
#         out = F.relu(out)
#         out = self.dropout(out)
#         out = out.transpose(1, 2)       # [B*S, T, d_model]

#         # 殘差 + LayerNorm（沿著最後一維）
#         return self.norm(out + x)       # [B*S, T, d_model]


# class TextNodeEncoder(nn.Module):
#     """
#     純文字編碼 (RoBERTa → TCN → utterance masked mean pooling)
#     不碰 explain、不管 speaker/emotion。
#     輸出 flatten node 表徵: [N, d_sem]
#     """
#     def __init__(self, text_model_name, freeze_text=True,
#                  tcn_layers=2, tcn_kernel_size=3, tcn_dropout=0.1):
#         super().__init__()
#         self.text_encoder = AutoModel.from_pretrained(text_model_name)
#         if freeze_text:
#             for p in self.text_encoder.parameters():
#                 p.requires_grad = False

#         self.d_sem = self.text_encoder.config.hidden_size

#         # 疊幾層 TCN block，d_model 不變（= d_sem）
#         self.tcn = nn.ModuleList()
#         for l in range(tcn_layers):
#             dilation = 2 ** l   # 逐層加大 receptive field
#             self.tcn.append(
#                 TCNBlock(
#                     d_model=self.d_sem,
#                     kernel_size=tcn_kernel_size,
#                     dilation=dilation,
#                     dropout=tcn_dropout,
#                 )
#             )

#     def _masked_mean(self, hidden, mask):
#         """
#         hidden: [B*S, T, d_sem]
#         mask:   [B*S, T]  (1=有效, 0=padding)
#         """
#         mask = mask.unsqueeze(-1)               # [B*S, T, 1]
#         hidden = hidden * mask                  # padding token 乘 0
#         summed = hidden.sum(1)                  # [B*S, d_sem]
#         count = mask.sum(1).clamp(min=1e-9)     # [B*S, 1]
#         return summed / count                   # [B*S, d_sem]

#     def forward(self, input_ids_padded, token_mask_padded, utterance_mask):
#         """
#         input_ids_padded:   [B, S, T]
#         token_mask_padded:  [B, S, T]
#         utterance_mask:     [B, S] (True=valid)
#         return: h_text_nodes: [N, d_sem]
#         """
#         B, S, T = input_ids_padded.shape

#         # 1. RoBERTa word-level encoding
#         outputs = self.text_encoder(
#             input_ids=input_ids_padded.view(B * S, T),
#             attention_mask=token_mask_padded.view(B * S, T),
#         )[0]  # [B*S, T, d_sem]

#         h_tok = outputs  # [B*S, T, d_sem]

#         # 2. TCN over token sequence
#         for block in self.tcn:
#             h_tok = block(h_tok)  # [B*S, T, d_sem]

#         # 3. masked mean pooling over time → utterance-level
#         h_utt = self._masked_mean(
#             h_tok,
#             token_mask_padded.view(B * S, T),
#         )  # [B*S, d_sem]

#         # 4. reshape 回 [B, S, d_sem]，再用 utterance_mask flatten
#         h_utt = h_utt.view(B, S, -1)    # [B, S, d_sem]
#         h_text_nodes = h_utt[utterance_mask]  # [N, d_sem]

#         return h_text_nodes


# Research/IMCEE/modules/crossTower/text_encoder.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Research/IMCEE/modules/crossTower/text_encoder.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModel

class TextNodeEncoder(nn.Module):
    """
    Early Fusion Text Encoder (With Projection)
    功能: RoBERTa + Spk + Emo -> Concat -> Linear(768) -> Contextual MHSA
    輸出: [N, d_text (768)]
    """
    def __init__(self, text_model_name, num_speakers, num_emotions, 
                 spk_dim=64, emo_dim=64, freeze_text=True, dropout=0.1):
        super().__init__()
        
        # 1. Text Model
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        self.d_sem = self.text_encoder.config.hidden_size # 768

        # 2. Embeddings
        self.spk_embed = nn.Embedding(num_speakers, spk_dim)
        self.emo_embed = nn.Embedding(num_emotions, emo_dim)

        # 3. Fusion Projection (896 -> 768)
        # 這裡將 Concat 後的特徵融合並降維，讓 MHSA 在標準維度上運作
        fusion_in_dim = self.d_sem + spk_dim + emo_dim
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_dim, self.d_sem),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Contextual MHSA (維度變回 768)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.d_sem, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.d_sem)

    def _mean_pool(self, hidden, mask):
        # hidden: [B*S, T, d], mask: [B*S, T]
        mask = mask.unsqueeze(-1)           # [B*S, T, 1]
        hidden = hidden * mask
        summed = hidden.sum(1)              # [B*S, d]
        count = mask.sum(1).clamp(min=1e-9)
        return summed / count               # [B*S, d]

    def forward(self, input_ids_padded, token_mask_padded, utterance_mask, 
                speaker_ids_padded, emotion_ids_padded):
        """
        Args:
            input_ids_padded:   [B, S, T]
            token_mask_padded:  [B, S, T]
            utterance_mask:     [B, S]
            speaker_ids_padded: [B, S]
            emotion_ids_padded: [B, S]
        Returns:
            h_text_nodes: [N, 768]
        """
        B, S, T = input_ids_padded.shape

        # 1. RoBERTa Encoding
        outputs = self.text_encoder(
            input_ids=input_ids_padded.view(B * S, T),
            attention_mask=token_mask_padded.view(B * S, T),
        )[0]

        h_text_utt = self._mean_pool(
            outputs,
            token_mask_padded.view(B * S, T)
        )  
        h_text_utt = h_text_utt.view(B, S, -1) # [B, S, 768]

        # 2. Embeddings
        h_spk = self.spk_embed(speaker_ids_padded) # [B, S, 64]
        h_emo = self.emo_embed(emotion_ids_padded) # [B, S, 64]

        # 3. Concat (768 + 64 + 64 = 896)
        h_cat = torch.cat([h_text_utt, h_spk, h_emo], dim=-1)

        # 4. Projection (896 -> 768)
        # [修改點] 在進 MHSA 之前，先降維
        h_fused = self.fusion_proj(h_cat) # [B, S, 768]

        # 5. Contextual MHSA (on 768 dim)
        key_padding_mask = ~utterance_mask 

        attn_out, _ = self.mhsa(
            query=h_fused, 
            key=h_fused, 
            value=h_fused, 
            key_padding_mask=key_padding_mask
        ) # [B, S, 768]

        # Residual + Norm
        h_final = self.norm(h_fused + attn_out)

        # 6. Flatten
        h_text_nodes = h_final[utterance_mask] # [N, 768]
        
        return h_text_nodes