# #version 4 IMCEE/modules/dataset.py (Fixed)
# import json
# import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import Data, Batch
# from pathlib import Path
# from tqdm import tqdm
# from collections import defaultdict


# def load_conversations(path):
#     convs = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line.strip())
#             utt_map = {utt["utt_id"]: utt for utt in obj["utterances"]}
#             convs[obj["conv_id"]] = utt_map
#     return convs


# def load_pairs(path):
#     pairs = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             pairs.append(json.loads(line.strip()))
#     return pairs


# def load_explain_data(embed_path, index_path):
#     embeds = torch.load(embed_path, map_location="cpu")
#     index_map = {}
#     with open(index_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 index_map[parts[1]] = int(parts[0])
#     return embeds, index_map


# class CEEEnd2EndDataset(Dataset):
#     def __init__(self, conversations_path, pairs_path, explain_embed_path, explain_index_path, use_explain=True):
#         self.convs = load_conversations(conversations_path)
#         all_pairs = load_pairs(pairs_path)
#         self.use_explain = use_explain

#         self.expl_embeds, self.expl_index = load_explain_data(explain_embed_path, explain_index_path)
#         self.spk_map = {"A": 0, "B": 1, "None": 2}

#         self.emo_map = {
#             "neutral": 0, "happiness": 1, "anger": 2, "surprise": 3,
#             "disgust": 4, "sadness": 5, "fear": 6,
#         }

#         self.emotion_alias = {
#             "happy": "happiness", "happines": "happiness", "excited": "happiness",
#             "angry": "anger", "sad": "sadness", "surprised": "surprise",
#         }

#         self.num_speakers = len(self.spk_map)
#         self.num_emotions = len(self.emo_map) 

#         self.data_list = []
#         for conv_id, utterances_map in tqdm(self.convs.items(), desc="Building Graphs"):
#             utterances = sorted(list(utterances_map.values()), key=lambda u: u['turn'])
#             id_to_idx = {utt['utt_id']: i for i, utt in enumerate(utterances)}

#             texts = [utt['text'] for utt in utterances]
#             speaker_ids = []
#             emotion_ids = []

#             for utt in utterances:
#                 # ----- Speaker -----
#                 spk_raw = utt.get("speaker", "None")
#                 spk = spk_raw if spk_raw in self.spk_map else "None"
#                 speaker_ids.append(self.spk_map[spk])

#                 # ----- Emotion -----
#                 emo_raw = utt.get("emotion", "neutral")
#                 emo_norm = self.normalize_emotion(emo_raw)
#                 emo_final = self.emo_map.get(emo_norm, self.emo_map["neutral"])
#                 emotion_ids.append(emo_final)

#             real_num_nodes = len(texts)  # = num_utts，不多加 1


#             edge_src, edge_tgt, edge_labels, edge_features, edge_distance, edge_types = [], [], [], [], [], []

#             # === 1. 任務邊 (candidate → target) Type 0 ===
#             for p in all_pairs:
#                 if p['conv_id'] != conv_id: continue
#                 c_idx = id_to_idx.get(p['c_utt_id'])
#                 t_idx = id_to_idx.get(p['t_utt_id'])
#                 pid = f"{p['conv_id']}__{p['c_utt_id']}__{p['t_utt_id']}"
#                 if c_idx is not None and t_idx is not None and pid in self.expl_index:
#                     edge_src.append(c_idx)
#                     edge_tgt.append(t_idx)
#                     edge_labels.append(int(p['label']))
#                     if self.use_explain:
#                         edge_features.append(self.expl_embeds[self.expl_index[pid]])
#                     else:
#                         edge_features.append(torch.zeros_like(self.expl_embeds[0]))
#                     edge_distance.append(abs(c_idx - t_idx))
#                     edge_types.append(0)

#             # === 2. 時序邊 (U₁→U₂→...→Uₙ) Type 1 ===
#             num_utts = len(utterances) # 注意：只連真實的 utterance，不連 Conv Node
#             for i in range(num_utts - 1):
#                 edge_src.append(i)
#                 edge_tgt.append(i + 1)
#                 edge_labels.append(0.0)
#                 edge_features.append(torch.zeros_like(self.expl_embeds[0]))
#                 edge_distance.append(1)
#                 edge_types.append(1)

            

#             if not edge_src:
#                 continue

#             # === 去除重複邊 (邏輯不變) ===
#             edge_pairs = list(zip(edge_src, edge_tgt))
#             pair_to_first_idx = {}
#             for idx, pair in enumerate(edge_pairs):
#                 if pair not in pair_to_first_idx:
#                     pair_to_first_idx[pair] = idx

#             unique_indices = sorted(pair_to_first_idx.values())
            
#             edge_src = [edge_src[i] for i in unique_indices]
#             edge_tgt = [edge_tgt[i] for i in unique_indices]
#             edge_labels = [edge_labels[i] for i in unique_indices]
#             edge_distance = [edge_distance[i] for i in unique_indices]
#             edge_features = [edge_features[i] for i in unique_indices]
#             edge_types = [edge_types[i] for i in unique_indices]

#             edge_task_mask = torch.tensor(
#                 [t == 0 for t in edge_types], dtype=torch.bool
#             )

#             # === 建立圖 ===
#             graph_data = Data(
#                 texts=texts,
#                 speaker_ids=torch.tensor(speaker_ids, dtype=torch.long),
#                 emotion_ids=torch.tensor(emotion_ids, dtype=torch.long),
#                 edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
#                 edge_features=torch.stack(edge_features) if edge_features else torch.empty((0, self.get_explain_dim())),
#                 edge_distance=torch.tensor(edge_distance, dtype=torch.long),
#                 edge_labels=torch.tensor(edge_labels, dtype=torch.float),
#                 edge_types=torch.tensor(edge_types, dtype=torch.long),
#                 # === [關鍵修正] num_nodes 必須是包含 Conv Node 的數量 ===
#                 num_nodes=real_num_nodes, 
#                 edge_task_mask=edge_task_mask,
#             )
#             self.data_list.append(graph_data)

#     def __len__(self):
#         return len(self.data_list)

#     def get_explain_dim(self):
#         return self.expl_embeds.shape[1] if self.expl_embeds is not None else 1024

#     def __getitem__(self, idx):
#         return self.data_list[idx]
    
#     def normalize_emotion(self, emo_str: str) -> str:
#         if emo_str is None: return "neutral"
#         emo = emo_str.strip().lower()
#         emo = self.emotion_alias.get(emo, emo)
#         if emo not in self.emo_map: return "neutral"
#         return emo


# class End2EndCollate:
#     def __init__(self, tokenizer, max_len=128):
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __call__(self, batch_graphs):
#         batch = Batch.from_data_list(batch_graphs)
#         all_texts, max_seq_len_in_batch = [], 0
#         for g in batch_graphs:
#             all_texts.extend(g.texts)
#             max_seq_len_in_batch = max(max_seq_len_in_batch, len(g.texts))
        
#         batch_encodings = self.tokenizer(
#             all_texts, padding='longest', truncation=True,
#             max_length=self.max_len, return_tensors="pt"
#         )

#         B = len(batch_graphs)
#         S = max_seq_len_in_batch
#         T = batch_encodings['input_ids'].shape[-1]
        
#         padded_input_ids = torch.zeros((B, S, T), dtype=torch.long)
#         padded_token_mask = torch.zeros((B, S, T), dtype=torch.long)
#         padded_speaker_ids = torch.zeros((B, S), dtype=torch.long)
#         padded_emotion_ids = torch.zeros((B, S), dtype=torch.long)
#         utterance_mask = torch.zeros((B, S), dtype=torch.bool)

#         current_node_idx = 0
#         for i, g in enumerate(batch_graphs):
#             n = g.num_nodes # 這裡因為 Dataset 修正了，n 現在是正確的 N+1
#             padded_input_ids[i, :n] = batch_encodings['input_ids'][current_node_idx:current_node_idx + n]
#             padded_token_mask[i, :n] = batch_encodings['attention_mask'][current_node_idx:current_node_idx + n]
#             padded_speaker_ids[i, :n] = g.speaker_ids
#             padded_emotion_ids[i, :n] = g.emotion_ids
#             utterance_mask[i, :n] = True
#             current_node_idx += n

#         batch.input_ids = padded_input_ids
#         batch.token_mask = padded_token_mask
#         batch.speaker_ids_padded = padded_speaker_ids
#         batch.emotion_ids_padded = padded_emotion_ids
#         batch.utterance_mask = utterance_mask
#         return batch
#version 10 IMCEE/modules/dataset.py (Explain Restored)
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def load_conversations(path):
    convs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            utt_map = {utt["utt_id"]: utt for utt in obj["utterances"]}
            convs[obj["conv_id"]] = utt_map
    return convs

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs

def load_explain_data(embed_path, index_path):
    # 如果路徑不存在或為 None，回傳 None
    if not embed_path or not index_path or not Path(embed_path).exists():
        return None, {}
    
    embeds = torch.load(embed_path, map_location="cpu")
    index_map = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                index_map[parts[1]] = int(parts[0])
    return embeds, index_map

class CEEEnd2EndDataset(Dataset):
    def __init__(self, conversations_path, pairs_path, 
                 explain_embed_path=None, explain_index_path=None, 
                 use_explain=True):
        
        self.convs = load_conversations(conversations_path)
        all_pairs = load_pairs(pairs_path)
        
        self.use_explain = use_explain
        self.expl_embeds, self.expl_index = load_explain_data(explain_embed_path, explain_index_path)
        
        # 定義 Speaker / Emotion Map
        self.spk_map = {"A": 0, "B": 1, "None": 2}
        self.emo_map = {
            "neutral": 0, "happiness": 1, "anger": 2, "surprise": 3,
            "disgust": 4, "sadness": 5, "fear": 6,
        }
        self.emotion_alias = {
            "happy": "happiness", "happines": "happiness", "excited": "happiness",
            "angry": "anger", "sad": "sadness", "surprised": "surprise",
        }

        # [關鍵修復] 必須定義，因為 run_cross_tower.py 會讀取
        self.num_speakers = len(self.spk_map)
        self.num_emotions = len(self.emo_map)

        self.data_list = []
        
        # 依照 conv_id 分組
        pairs_by_conv = defaultdict(list)
        for p in all_pairs:
            pairs_by_conv[p['conv_id']].append(p)

        # 開始建構圖
        for conv_id, utterances_map in tqdm(self.convs.items(), desc="Building Graph (Explain Supported)"):
            utterances = sorted(list(utterances_map.values()), key=lambda u: u['turn'])
            id_to_idx = {utt['utt_id']: i for i, utt in enumerate(utterances)}

            base_texts = [utt['text'] for utt in utterances]
            base_speaker_ids = []
            base_emotion_ids = []
            
            for utt in utterances:
                spk = utt.get("speaker", "None")
                spk = spk if spk in self.spk_map else "None"
                base_speaker_ids.append(self.spk_map[spk])
                
                emo = utt.get("emotion", "neutral")
                emo = self.normalize_emotion(emo)
                base_emotion_ids.append(self.emo_map.get(emo, 0))

            num_utts = len(base_texts)
            conv_pairs = pairs_by_conv[conv_id]
            if not conv_pairs: continue

            for p in conv_pairs:
                c_idx = id_to_idx.get(p['c_utt_id'])
                t_idx = id_to_idx.get(p['t_utt_id'])
                pid = f"{conv_id}__{p['c_utt_id']}__{p['t_utt_id']}"
                
                # 基本檢查
                if c_idx is None or t_idx is None:
                    continue

                # ==========================================
                # Node Construction
                # ==========================================
                conv_node_idx = num_utts
                cause_node_idx = num_utts + 1
                target_node_idx = num_utts + 2
                total_nodes = num_utts + 3

                current_texts = base_texts + ["[CLS]", base_texts[c_idx], base_texts[t_idx]]
                current_spk_ids = base_speaker_ids + [self.spk_map["None"]] * 3
                current_emo_ids = base_emotion_ids + [self.emo_map["neutral"]] * 3

                # 準備 Edge Lists
                edge_src, edge_tgt, edge_types, edge_features = [], [], [], []
                edge_distances = [] 

                # 準備 Explain Feature
                # 如果 use_explain 為 True 且有資料 -> 讀取
                # 如果 use_explain 為 False 或 沒資料 -> 全 0 (維度依照 get_explain_dim)
                if self.use_explain and pid in self.expl_index and self.expl_embeds is not None:
                    expl_feat = self.expl_embeds[self.expl_index[pid]]
                else:
                    expl_feat = torch.zeros(self.get_explain_dim())
                zero_feat = torch.zeros(self.get_explain_dim())
                # -------------------------------------------------
                # 1. Classification Edges (Type 0)
                # -------------------------------------------------
                # Cause Node -> Cause Utterance (附帶 Explain Feature)
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=cause_node_idx, tgt=c_idx, etype=0, dist=0, feat=expl_feat)
                
                # Target Node -> Target Utterance (附帶 Explain Feature)
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=target_node_idx, tgt=t_idx, etype=0, dist=0, feat=expl_feat)
                
                # [Fix Here] Cause <-> CauseUtt (雙向補充)
                # 補上 list 參數 與 dist=0
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=cause_node_idx, tgt=c_idx, etype=0, dist=0, feat=expl_feat)
                
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=c_idx, tgt=cause_node_idx, etype=0, dist=0, feat=zero_feat) 

                # [Fix Here] Target <-> TargetUtt (雙向補充)
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=target_node_idx, tgt=t_idx, etype=0, dist=0, feat=expl_feat)
                
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                               src=t_idx, tgt=target_node_idx, etype=0, dist=0, feat=zero_feat)

                # -------------------------------------------------
                # 2. Temporal Edges (window <= 2)
                # -------------------------------------------------
                # zero_feat = torch.zeros(self.get_explain_dim())

                for i in range(num_utts):
                    for dist in (1, 2):
                        j = i + dist
                        if j < num_utts:
                            t_type = 1 if dist == 1 else 2
                            self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                        src=i, tgt=j, etype=t_type, dist=dist, feat=zero_feat)
                # -------------------------------------------------
                # 3. Speaker Edges
                # -------------------------------------------------
                spk_to_indices = defaultdict(list)
                for i in range(num_utts):
                    spk_to_indices[base_speaker_ids[i]].append(i)
                
                for spk, indices in spk_to_indices.items():
                    if spk == self.spk_map["None"]: continue
                    for k in range(len(indices) - 1):
                        u_curr = indices[k]
                        u_next = indices[k+1]
                        dist = u_next - u_curr
                        self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                       src=u_curr, tgt=u_next, etype=4, dist=dist, feat=zero_feat)

                # -------------------------------------------------
                # 4. Global Edges
                # -------------------------------------------------
                for i in range(num_utts):
                    self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                   src=conv_node_idx, tgt=i, etype=5, dist=0, feat=zero_feat)
                    self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                   src=i, tgt=conv_node_idx, etype=5, dist=0, feat=zero_feat)

                # ==========================================
                # Build Data Object
                # ==========================================
                target_indices_tensor = torch.tensor([[cause_node_idx, target_node_idx]], dtype=torch.long)

                graph_data = Data(
                    texts=current_texts,
                    speaker_ids=torch.tensor(current_spk_ids, dtype=torch.long),
                    emotion_ids=torch.tensor(current_emo_ids, dtype=torch.long),
                    edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
                    edge_types=torch.tensor(edge_types, dtype=torch.long),
                    edge_features=torch.stack(edge_features), # 這裡一定要 stack
                    edge_distance=torch.tensor(edge_distances, dtype=torch.long),
                    edge_label=torch.tensor([float(p['label'])], dtype=torch.float),
                    num_nodes=total_nodes,
                    target_node_indices=target_indices_tensor 
                )
                self.data_list.append(graph_data)

    def _add_edge(self, src_list, tgt_list, type_list, dist_list, feat_list, src, tgt, etype, dist, feat):
        src_list.append(src)
        tgt_list.append(tgt)
        type_list.append(etype)
        dist_list.append(dist)
        feat_list.append(feat)

    def get_explain_dim(self):
        if self.expl_embeds is not None:
            return self.expl_embeds.shape[1]
        return 1024 # 預設 fallback

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def normalize_emotion(self, emo_str: str) -> str:
        if emo_str is None: return "neutral"
        emo = emo_str.strip().lower()
        emo = self.emotion_alias.get(emo, emo)
        if emo not in self.emo_map: return "neutral"
        return emo

class End2EndCollate:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch_graphs):
        batch = Batch.from_data_list(batch_graphs)
        all_texts, max_seq_len_in_batch = [], 0
        
        for g in batch_graphs:
            all_texts.extend(g.texts)
            max_seq_len_in_batch = max(max_seq_len_in_batch, len(g.texts))
        
        batch_encodings = self.tokenizer(
            all_texts, padding='longest', truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )

        B = len(batch_graphs)
        S = max_seq_len_in_batch
        T = batch_encodings['input_ids'].shape[-1]
        
        padded_input_ids = torch.zeros((B, S, T), dtype=torch.long)
        padded_token_mask = torch.zeros((B, S, T), dtype=torch.long)
        padded_speaker_ids = torch.zeros((B, S), dtype=torch.long)
        padded_emotion_ids = torch.zeros((B, S), dtype=torch.long)
        utterance_mask = torch.zeros((B, S), dtype=torch.bool)

        current_node_idx = 0
        for i, g in enumerate(batch_graphs):
            n = g.num_nodes
            padded_input_ids[i, :n] = batch_encodings['input_ids'][current_node_idx:current_node_idx + n]
            padded_token_mask[i, :n] = batch_encodings['attention_mask'][current_node_idx:current_node_idx + n]
            padded_speaker_ids[i, :n] = g.speaker_ids
            padded_emotion_ids[i, :n] = g.emotion_ids
            utterance_mask[i, :n] = True
            current_node_idx += n

        batch.input_ids = padded_input_ids
        batch.token_mask = padded_token_mask
        batch.speaker_ids_padded = padded_speaker_ids
        batch.emotion_ids_padded = padded_emotion_ids
        batch.utterance_mask = utterance_mask
        
        return batch