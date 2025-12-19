# Research/IMCEE/modules/crossTower/text_encoder.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

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

        # Gatedmhsa
        # attn_out = self.mhsa(h_fused, key_padding_mask=~utterance_mask) 
        # h_final = self.norm(h_fused + attn_out)

        # 6. Flatten
        h_text_nodes = h_final[utterance_mask] # [N, 768]
        
        return h_text_nodes

# # Research/IMCEE/modules/crossTower/text_encoder.py
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# from transformers import AutoModel
# import torch.nn.functional as F

# class TextNodeEncoder(nn.Module):
#     """
#     CFIB-Style Document Encoder (Full Context Interaction)
    
#     [核心改動]
#     1. Input Formulation: 將 [Batch, Sent, Token] 攤平為 [Batch, Sent*Token] 餵入 RoBERTa。
#        讓 RoBERTa 能夠看到整段對話的上下文 (Document-Level Encoding)。
#     2. Pooling: 使用 CFIB 的 Attention Pooling 機制。
#     3. Windowing: 自動處理超過 512 Token 的長度問題。
    
#     輸出: [N, d_text (768)]
#     """
#     def __init__(self, text_model_name, num_speakers, num_emotions, 
#                  spk_dim=64, emo_dim=64, freeze_text=True, dropout=0.1):
#         super().__init__()
        
#         # 1. Text Model
#         self.text_encoder = AutoModel.from_pretrained(text_model_name)
#         if freeze_text:
#             for p in self.text_encoder.parameters():
#                 p.requires_grad = False
        
#         self.d_sem = self.text_encoder.config.hidden_size # 768

#         # [CFIB] Attention Pooling Score Calculation
#         self.attn_score = nn.Linear(self.d_sem, 1)

#         # 2. Embeddings
#         self.spk_embed = nn.Embedding(num_speakers, spk_dim)
#         self.emo_embed = nn.Embedding(num_emotions, emo_dim)

#         # 3. Fusion Projection
#         fusion_in_dim = self.d_sem + spk_dim + emo_dim
        
#         self.fusion_proj = nn.Sequential(
#             nn.Linear(fusion_in_dim, self.d_sem),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         # 4. Contextual MHSA (保留作為最後一層的特徵修飾)
#         self.mhsa = nn.MultiheadAttention(
#             embed_dim=self.d_sem, 
#             num_heads=8, 
#             dropout=dropout, 
#             batch_first=True
#         )

#         self.norm = nn.LayerNorm(self.d_sem)

#     def _attention_pool(self, hidden, mask):
#         """
#         CFIB Attention Pooling
#         hidden: [B, S, T, d]
#         mask:   [B, S, T]
#         """
#         # [B, S, T, 1]
#         attn_logits = self.attn_score(hidden) 
        
#         # Masking: (1-mask) * -9e5
#         extended_mask = (1.0 - mask.unsqueeze(-1)) * -9e5
#         attn_logits = attn_logits + extended_mask
        
#         # Softmax along Token dimension (dim=2)
#         alpha = F.softmax(attn_logits, dim=2) 
        
#         # Weighted Sum: [B, S, T, d] * [B, S, T, 1] -> sum(dim=2) -> [B, S, d]
#         weighted_sum = torch.sum(alpha * hidden, dim=2)
        
#         return weighted_sum

#     def _process_long_input(self, input_ids, attention_mask):
#         """
#         處理超過 512 的長度限制 (Sliding Window 策略)
#         input_ids: [B, Total_Len]
#         """
#         B, Total_Len = input_ids.shape
#         max_len = 512
#         stride = 256
        
#         if Total_Len <= max_len:
#             outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
#             return outputs

#         # 如果超過 512，切片處理再拼接 (簡易版 Sliding Window)
#         # 注意：這只是近似解，因為邊界處的 Context 會斷掉，但在 Graph Node Encoder 中是不得不做的妥協
#         outputs_list = []
#         for i in range(0, Total_Len, stride):
#             end = min(i + max_len, Total_Len)
#             chunk_ids = input_ids[:, i:end]
#             chunk_mask = attention_mask[:, i:end]
            
#             # 確保長度至少為1
#             if chunk_ids.shape[1] == 0: break

#             out = self.text_encoder(input_ids=chunk_ids, attention_mask=chunk_mask)[0]
            
#             # 如果不是第一塊，要扣掉重疊部分 (stride策略需更複雜，這裡採用簡單拼接非重疊部分)
#             # 這裡為了 "無痛" 且保持維度一致，我們簡單假設 Batch 內所有對話長度不一
#             # 實作上最保險的方式是: 既然是 Graph Node，我們其實可以退回 Sentence Encoding
#             # 但既然你要 CFIB 效果，我們這裡做一個簡單的 Padding 處理
            
#             # 修正策略：直接切斷 (Truncate) 或只取前 512
#             # 為了避免複雜的拼接 bug，若超過 512，我們這裡只取前 512 (這是大多數長文檔處理的標準妥協)
#             # 若要完整保留，建議在 Data Loader 層級就切好
#             if i == 0:
#                 outputs_list = out
#             else:
#                 # 這裡若要拼接會很複雜，因為 RoBERTa 的 Positional Embedding 是重置的
#                 pass 
        
#         # [妥協] 為了確保程式不崩潰，若超過 512，我們目前只能處理前 512 tokens
#         # 真正要解決這個需要 Longformer，或是改 Data Loader
#         final_out = self.text_encoder(input_ids=input_ids[:, :max_len], 
#                                       attention_mask=attention_mask[:, :max_len])[0]
        
#         # Pad back to original length (with zeros) so unflatten works
#         if Total_Len > max_len:
#             pad_len = Total_Len - max_len
#             pad_tensor = torch.zeros(B, pad_len, self.d_sem, device=input_ids.device)
#             final_out = torch.cat([final_out, pad_tensor], dim=1)
            
#         return final_out

#     def forward(self, input_ids_padded, token_mask_padded, utterance_mask, 
#                 speaker_ids_padded, emotion_ids_padded):
#         """
#         Args:
#             input_ids_padded: [B, S, T] (Batch, Sentences, Tokens)
#         """
#         B, S, T = input_ids_padded.shape

#         # ---------------------------------------------------------
#         # 1. Flatten Inputs: [B, S, T] -> [B, S*T]
#         # 這樣做讓 RoBERTa 能看到 "跨句子" 的上下文 (Document Level)
#         # ---------------------------------------------------------
#         flat_input_ids = input_ids_padded.view(B, S * T)
#         flat_token_mask = token_mask_padded.view(B, S * T)

#         # 2. RoBERTa Encoding (Document Level)
#         # 注意：這裡會遇到 512 限制。
#         # 如果你的對話很長，這一步只會吃到前 512 個 token (約 10-15 句話)
#         outputs_long = self._process_long_input(flat_input_ids, flat_token_mask) 
#         # outputs_long: [B, S*T, 768]

#         # 3. Unflatten: 切回 [B, S, T, 768]
#         # 這樣我們就拿到了 "看過整篇對話" 的每個 Token 的特徵
#         outputs_reshaped = outputs_long.view(B, S, T, -1)

#         # 4. Attention Pooling (CFIB Style)
#         # 從 [B, S, T, 768] -> [B, S, 768]
#         h_text_utt = self._attention_pool(
#             outputs_reshaped,
#             token_mask_padded # [B, S, T]
#         )

#         # 5. Embeddings Lookup & Fusion (不變)
#         h_spk = self.spk_embed(speaker_ids_padded) 
#         h_emo = self.emo_embed(emotion_ids_padded) 
#         h_cat = torch.cat([h_text_utt, h_spk, h_emo], dim=-1) 
#         h_fused = self.fusion_proj(h_cat) 

#         # 6. Contextual MHSA (不變)
#         key_padding_mask = ~utterance_mask 
#         attn_out, _ = self.mhsa(
#             query=h_fused, key=h_fused, value=h_fused, 
#             key_padding_mask=key_padding_mask
#         )
#         h_final = self.norm(h_fused + attn_out)

#         # 7. Flatten to Nodes
#         h_text_nodes = h_final[utterance_mask] 
        
#         return h_text_nodes