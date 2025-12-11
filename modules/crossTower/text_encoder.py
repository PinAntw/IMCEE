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

# Research/IMCEE/modules/crossTower/text_encoder.py 如果心encoder不好就換這個class
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# from transformers import AutoModel
# import torch.nn.functional as F

# class TextNodeEncoder(nn.Module):
#     """
#     Early Fusion Text Encoder (With Projection)
#     功能: RoBERTa + Spk + Emo -> Concat -> Linear(768) -> Contextual MHSA
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

#         # 2. Embeddings
#         self.spk_embed = nn.Embedding(num_speakers, spk_dim)
#         self.emo_embed = nn.Embedding(num_emotions, emo_dim)

#         # 3. Fusion Projection (896 -> 768)
#         # 這裡將 Concat 後的特徵融合並降維，讓 MHSA 在標準維度上運作
#         fusion_in_dim = self.d_sem + spk_dim + emo_dim
        
#         self.fusion_proj = nn.Sequential(
#             nn.Linear(fusion_in_dim, self.d_sem),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         # 4. Contextual MHSA (維度變回 768)
#         self.mhsa = nn.MultiheadAttention(
#             embed_dim=self.d_sem, 
#             num_heads=8, 
#             dropout=dropout, 
#             batch_first=True
#         )

#         # self.mhsa = GatedSDPAMHA(d_model=self.d_sem, num_heads=8, dropout=dropout)

#         self.norm = nn.LayerNorm(self.d_sem)

#     def _mean_pool(self, hidden, mask):
#         # hidden: [B*S, T, d], mask: [B*S, T]
#         mask = mask.unsqueeze(-1)           # [B*S, T, 1]
#         hidden = hidden * mask
#         summed = hidden.sum(1)              # [B*S, d]
#         count = mask.sum(1).clamp(min=1e-9)
#         return summed / count               # [B*S, d]

#     def forward(self, input_ids_padded, token_mask_padded, utterance_mask, 
#                 speaker_ids_padded, emotion_ids_padded):
#         """
#         Args:
#             input_ids_padded:   [B, S, T]
#             token_mask_padded:  [B, S, T]
#             utterance_mask:     [B, S]
#             speaker_ids_padded: [B, S]
#             emotion_ids_padded: [B, S]
#         Returns:
#             h_text_nodes: [N, 768]
#         """
#         B, S, T = input_ids_padded.shape

#         # 1. RoBERTa Encoding
#         outputs = self.text_encoder(
#             input_ids=input_ids_padded.view(B * S, T),
#             attention_mask=token_mask_padded.view(B * S, T),
#         )[0]

#         h_text_utt = self._mean_pool(
#             outputs,
#             token_mask_padded.view(B * S, T)
#         )  
#         h_text_utt = h_text_utt.view(B, S, -1) # [B, S, 768]

#         # 2. Embeddings
#         h_spk = self.spk_embed(speaker_ids_padded) # [B, S, 64]
#         h_emo = self.emo_embed(emotion_ids_padded) # [B, S, 64]

#         # 3. Concat (768 + 64 + 64 = 896)
#         h_cat = torch.cat([h_text_utt, h_spk, h_emo], dim=-1)

#         # 4. Projection (896 -> 768)
#         # [修改點] 在進 MHSA 之前，先降維
#         h_fused = self.fusion_proj(h_cat) # [B, S, 768]

#         # 5. Contextual MHSA (on 768 dim)
#         key_padding_mask = ~utterance_mask 
#         attn_out, _ = self.mhsa(
#             query=h_fused, 
#             key=h_fused, 
#             value=h_fused, 
#             key_padding_mask=key_padding_mask
#         ) # [B, S, 768]
#         # Residual + Norm
#         h_final = self.norm(h_fused + attn_out)

#         # Gatedmhsa
#         # attn_out = self.mhsa(h_fused, key_padding_mask=~utterance_mask) 
#         # h_final = self.norm(h_fused + attn_out)

#         # 6. Flatten
#         h_text_nodes = h_final[utterance_mask] # [N, 768]
        
#         return h_text_nodes
# Research/IMCEE/modules/crossTower/text_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class TextNodeEncoder(nn.Module):
    def __init__(self, text_model_name, num_speakers, num_emotions, 
                 spk_dim=768, emo_dim=768,  # [修改] 預設改為 768 以便對齊 RoBERTa
                 freeze_text=True, dropout=0.1):
        super().__init__()
        
        # 1. Text Model
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # [新增] 保存 tokenizer 以便查詢詞向量 (若是為了初始化)
        # 注意：通常 tokenizer 載入很快，或可在外部傳入
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        self.d_sem = self.text_encoder.config.hidden_size # 768

        # 2. Embeddings (原本是隨機初始化)
        # 我們保持它是 nn.Embedding，這樣它依然是 Learnable 的
        self.spk_embed = nn.Embedding(num_speakers, spk_dim)
        self.emo_embed = nn.Embedding(num_emotions, emo_dim)

        # 3. Fusion Projection 
        # 輸入維度變大了: 768(Text) + 768(Spk) + 768(Emo) = 2304
        fusion_in_dim = self.d_sem + spk_dim + emo_dim
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_dim, self.d_sem),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Contextual MHSA
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.d_sem, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.d_sem)

        # [新增] 自動執行語意初始化
        self._init_embeddings_with_semantics()

    def _init_embeddings_with_semantics(self):
        """
        利用 RoBERTa 的 Word Embeddings 來初始化 Speaker 和 Emotion 向量
        這樣模型一開始就懂得 "happiness" 的語意，而不是從隨機雜訊開始學。
        """
        print(">>> Initializing Speaker/Emotion Embeddings with RoBERTa semantics...")
        
        # 取得 RoBERTa 的 Word Embedding Table (不含 Positional/Token Type)
        # 這是 [Vocab_Size, 768] 的大表
        word_embeddings = self.text_encoder.embeddings.word_embeddings.weight.data
        
        # --- 1. Initialize Emotion Embeddings ---
        # 定義你的 Emotion 對應文字 (順序必須跟 dataset.py 的 self.emo_map 一致)
        # self.emo_map = {"neutral":0, "happiness":1, "anger":2, "surprise":3, "disgust":4, "sadness":5, "fear":6}
        emo_words = ["neutral", "happiness", "anger", "surprise", "disgust", "sadness", "fear"]
        
        for idx, word in enumerate(emo_words):
            # 取得該單字的 Token ID (取第一個 token，例如 "happiness" -> 2345)
            token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
            # 複製權重
            self.emo_embed.weight.data[idx] = word_embeddings[token_id].clone()
            
        # --- 2. Initialize Speaker Embeddings ---
        # self.spk_map = {"A":0, "B":1, "None":2}
        spk_words = ["A", "B", "person"] # None 用 "person" 或 "nobody" 代替
        
        for idx, word in enumerate(spk_words):
            token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
            self.spk_embed.weight.data[idx] = word_embeddings[token_id].clone()
            
        # 記得：雖然初始化了，但 requires_grad 預設是 True，所以它還是可以被微調 (Learnable)

    def _mean_pool(self, hidden, mask):
        mask = mask.unsqueeze(-1)
        hidden = hidden * mask
        summed = hidden.sum(1)
        count = mask.sum(1).clamp(min=1e-9)
        return summed / count

    def forward(self, input_ids_padded, token_mask_padded, utterance_mask, 
                speaker_ids_padded, emotion_ids_padded):
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

        # 2. Embeddings (Lookup)
        # 現在這些向量是帶有 RoBERTa 語意初始值的
        h_spk = self.spk_embed(speaker_ids_padded) # [B, S, 768]
        h_emo = self.emo_embed(emotion_ids_padded) # [B, S, 768]

        # 3. Concat (768 * 3 = 2304)
        h_cat = torch.cat([h_text_utt, h_spk, h_emo], dim=-1)

        # 4. Projection (2304 -> 768)
        h_fused = self.fusion_proj(h_cat) # [B, S, 768]

        # 5. MHSA
        key_padding_mask = ~utterance_mask 
        attn_out, _ = self.mhsa(
            query=h_fused, 
            key=h_fused, 
            value=h_fused, 
            key_padding_mask=key_padding_mask
        ) 
        h_final = self.norm(h_fused + attn_out)

        h_text_nodes = h_final[utterance_mask]
        
        return h_text_nodes
    
class GatedSDPAMHA(nn.Module):
    """
    G1: SDPA output gating (elementwise, head-specific, query-dependent)
    - gate score = sigmoid( LN(x) @ Wg + bg )  -> shape [B,S,H,dk]
    - gated_out = sdpa_out * gate
    """
    def __init__(self, d_model=768, num_heads=8, dropout=0.1, gate_bias_init=1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dropout = dropout

        self.pre_norm = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # elementwise gate: d_model -> H*dk (= d_model)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)  # sigmoid(1)=0.73，接近保留 baseline 訊息

    def forward(self, x, key_padding_mask=None):
        """
        x: [B,S,d_model]
        key_padding_mask: [B,S]  True=pad (和 nn.MultiheadAttention 一樣語意)
        """
        B, S, _ = x.shape
        x_ln = self.pre_norm(x)

        qkv = self.qkv(x_ln)  # [B,S,3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,H,S,dk]
        q = q.view(B, S, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.dk).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # scaled_dot_product_attention 的 bool mask：True 表示要 mask 掉
            attn_mask = key_padding_mask[:, None, None, :].expand(B, 1, S, S)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )  # [B,H,S,dk]

        # gate from current query hidden state (query-dependent)
        g = torch.sigmoid(self.gate_proj(x_ln))                 # [B,S,d_model]
        g = g.view(B, S, self.num_heads, self.dk).transpose(1, 2)  # [B,H,S,dk]

        y = y * g

        y = y.transpose(1, 2).contiguous().view(B, S, self.d_model)  # [B,S,d_model]
        y = self.out_proj(y)
        return y
