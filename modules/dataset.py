# IMCEE/modules/dataset.py

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
    """
    通用的 (embeddings.pt + index.tsv) loader：
    回傳 (embeddings_tensor 或 None, id -> index 的 dict)
    """
    if not embed_path or not index_path:
        return None, {}

    embed_path = Path(embed_path)
    index_path = Path(index_path)

    if not embed_path.exists() or not index_path.exists():
        return None, {}

    embeds = torch.load(embed_path, map_location="cpu")
    index_map = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            idx_str, edge_id = parts
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            index_map[edge_id] = idx
    return embeds, index_map

def load_llm_shortcuts(path):
    """
    讀取 LLM 預測結果。
    格式: {"id": "te_354__u1__u2", "label": "cause", ...}
    解析 id -> (conv_id, c_utt_id, t_utt_id)
    回傳: set 包含所有合法的 (conv_id, c_utt_id, t_utt_id) tuple
    """
    shortcuts = set()
    if not path:
        return shortcuts
    
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Warning: Shortcut file not found at {path}")
        return shortcuts
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                # 過濾條件：LLM 認為是 cause
                if obj.get("label") == "cause":
                    # 解析 ID: "te_354__u1__u2"
                    # 注意：conv_id 可能包含底線，所以建議 split 之後取頭尾，或者根據資料集特性處理
                    # 假設格式固定為 {conv_id}__{c_id}__{t_id}
                    raw_id = obj["id"]
                    parts = raw_id.split("__")
                    
                    if len(parts) >= 3:
                        t_id = parts[-1]
                        c_id = parts[-2]
                        # 剩下的前面部分組合成 conv_id (以防 conv_id 本身有底線)
                        conv_id = "__".join(parts[:-2])
                        
                        shortcuts.add((conv_id, c_id, t_id))
            except Exception as e:
                continue
                
    print(f"[Dataset] Loaded {len(shortcuts)} valid semantic shortcuts from LLM.")
    return shortcuts

class CEEEnd2EndDataset(Dataset):
    def __init__(
        self,
        conversations_path,
        pairs_path,
        llm_shortcut_path=None, # [接收路徑]
        explain_embed_path=None,
        explain_index_path=None,
        use_explain=True,
        expl_space_pt=None,
        expl_space_tsv=None,
        use_expl_space=False,
    ):
        # 讀對話 & pair
        self.convs = load_conversations(conversations_path)
        all_pairs = load_pairs(pairs_path)

        self.llm_shortcuts = load_llm_shortcuts(llm_shortcut_path)

        # 開關
        self.use_explain = use_explain
        self.use_expl_space = use_expl_space

        # 原本的 edge-level explain（edge_features 用）
        self.expl_embeds, self.expl_index = load_explain_data(
            explain_embed_path, explain_index_path
        )

        # 新的 explanation space（graph/pair-level 向量）
        self.expl_space_emb, self.expl_space_index = load_explain_data(
            expl_space_pt, expl_space_tsv
        )

        # Speaker / Emotion map
        self.spk_map = {"A": 0, "B": 1, "None": 2}
        self.emo_map = {
            "neutral": 0,
            "happiness": 1,
            "anger": 2,
            "surprise": 3,
            "disgust": 4,
            "sadness": 5,
            "fear": 6,
        }
        self.emotion_alias = {
            "happy": "happiness",
            "happines": "happiness",
            "excited": "happiness",
            "angry": "anger",
            "sad": "sadness",
            "surprised": "surprise",
        }

        # run_cross_tower 會讀這兩個
        self.num_speakers = len(self.spk_map)
        self.num_emotions = len(self.emo_map)

        self.data_list = []

        # 依 conv 分組 pairs
        pairs_by_conv = defaultdict(list)
        for p in all_pairs:
            pairs_by_conv[p["conv_id"]].append(p)

        # 建圖
        for conv_id, utt_map in tqdm(
            self.convs.items(), desc="Building Graph"
        ):
            # 排序 utterances
            utterances = sorted(list(utt_map.values()), key=lambda u: u["turn"])
            id_to_idx = {utt["utt_id"]: i for i, utt in enumerate(utterances)}

            base_texts = [utt["text"] for utt in utterances]
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
            if not conv_pairs:
                continue

            for p in conv_pairs:
                c_idx = id_to_idx.get(p["c_utt_id"])
                t_idx = id_to_idx.get(p["t_utt_id"])
                pid = f"{conv_id}__{p['c_utt_id']}__{p['t_utt_id']}"

                if c_idx is None or t_idx is None:
                    continue

                # ================================
                # node construction
                # ================================
                conv_node_idx = num_utts
                cause_node_idx = num_utts + 1
                target_node_idx = num_utts + 2
                total_nodes = num_utts + 3

                current_texts = base_texts + [
                    "[CLS]",
                    base_texts[c_idx],
                    base_texts[t_idx],
                ]
                current_spk_ids = base_speaker_ids + [self.spk_map["None"]] * 3
                current_emo_ids = base_emotion_ids + [self.emo_map["neutral"]] * 3

                edge_src, edge_tgt, edge_types, edge_features = [], [], [], []
                edge_distances = []

                # === edge-level explain feature ===
                expl_dim = self.get_explain_dim()
                if self.use_explain and self.expl_embeds is not None and pid in self.expl_index:
                    expl_feat = self.expl_embeds[self.expl_index[pid]]
                else:
                    expl_feat = torch.zeros(expl_dim)
                zero_feat = torch.zeros(expl_dim)

                # === explanation space (graph-level) 向量 ===
                expl_space_dim = self.get_expl_space_dim()
                if (
                    self.use_expl_space
                    and self.expl_space_emb is not None
                    and pid in self.expl_space_index
                ):
                    expl_space_vec = self.expl_space_emb[self.expl_space_index[pid]]
                else:
                    expl_space_vec = torch.zeros(expl_space_dim)

                # -------------------------------------------------
                # 1. Classification Edges (Type 0)
                # -------------------------------------------------
                # Cause Node -> Cause Utterance (帶 explain)
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=cause_node_idx,
                    tgt=c_idx,
                    etype=0,
                    dist=0,
                    feat=expl_feat,
                )

                # Target Node -> Target Utterance (帶 explain)
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=target_node_idx,
                    tgt=t_idx,
                    etype=0,
                    dist=0,
                    feat=expl_feat,
                )

                # Cause <-> CauseUtt (雙向補充)
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=cause_node_idx,
                    tgt=c_idx,
                    etype=0,
                    dist=0,
                    feat=expl_feat,
                )
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=c_idx,
                    tgt=cause_node_idx,
                    etype=0,
                    dist=0,
                    feat=zero_feat,
                )

                # Target <-> TargetUtt (雙向補充)
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=target_node_idx,
                    tgt=t_idx,
                    etype=0,
                    dist=0,
                    feat=expl_feat,
                )
                self._add_edge(
                    edge_src,
                    edge_tgt,
                    edge_types,
                    edge_distances,
                    edge_features,
                    src=t_idx,
                    tgt=target_node_idx,
                    etype=0,
                    dist=0,
                    feat=zero_feat,
                )

                # -------------------------------------------------
                # 2. Temporal Edges (window <= 2, 單向過去→未來)
                # -------------------------------------------------
                for i in range(num_utts):
                    for dist in (1, 2,3,4):
                        j = i + dist
                        if j < num_utts:
                            t_type = 1 if dist == 1 else 2
                            self._add_edge(
                                edge_src,
                                edge_tgt,
                                edge_types,
                                edge_distances,
                                edge_features,
                                src=i,
                                tgt=j,
                                etype=t_type,
                                dist=dist,
                                feat=zero_feat,
                            )

                # -------------------------------------------------
                # 3. Speaker Edges
                # -------------------------------------------------
                spk_to_indices = defaultdict(list)
                for i in range(num_utts):
                    spk_to_indices[base_speaker_ids[i]].append(i)

                for spk, indices in spk_to_indices.items():
                    if spk == self.spk_map["None"]:
                        continue
                    for k in range(len(indices) - 1):
                        u_curr = indices[k]
                        u_next = indices[k + 1]
                        dist = u_next - u_curr
                        self._add_edge(
                            edge_src,
                            edge_tgt,
                            edge_types,
                            edge_distances,
                            edge_features,
                            src=u_curr,
                            tgt=u_next,
                            etype=4,
                            dist=dist,
                            feat=zero_feat,
                        )

                # -------------------------------------------------
                # 4. Global Edges
                # -------------------------------------------------
                for i in range(num_utts):
                    self._add_edge(
                        edge_src,
                        edge_tgt,
                        edge_types,
                        edge_distances,
                        edge_features,
                        src=conv_node_idx,
                        tgt=i,
                        etype=5,
                        dist=0,
                        feat=zero_feat,
                    )
                    self._add_edge(
                        edge_src,
                        edge_tgt,
                        edge_types,
                        edge_distances,
                        edge_features,
                        src=i,
                        tgt=conv_node_idx,
                        etype=5,
                        dist=0,
                        feat=zero_feat,
                    )

                # ==========================================
                # Build Data Object
                # ==========================================
                target_indices_tensor = torch.tensor(
                    [[cause_node_idx, target_node_idx]], dtype=torch.long
                )

                graph_data = Data(
                    texts=current_texts,
                    speaker_ids=torch.tensor(current_spk_ids, dtype=torch.long),
                    emotion_ids=torch.tensor(current_emo_ids, dtype=torch.long),
                    edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
                    edge_types=torch.tensor(edge_types, dtype=torch.long),
                    edge_features=torch.stack(edge_features),
                    edge_distance=torch.tensor(edge_distances, dtype=torch.long),
                    edge_label=torch.tensor([float(p["label"])], dtype=torch.float),
                    num_nodes=total_nodes,
                    target_node_indices=target_indices_tensor,
                    # 新增：explanation space 向量
                    expl_space_vec=expl_space_vec,
                    pair_uttpos=torch.tensor([[c_idx, t_idx]], dtype=torch.long),
                )
                self.data_list.append(graph_data)

    def _add_edge(
        self,
        src_list,
        tgt_list,
        type_list,
        dist_list,
        feat_list,
        src,
        tgt,
        etype,
        dist,
        feat,
    ):
        src_list.append(src)
        tgt_list.append(tgt)
        type_list.append(etype)
        dist_list.append(dist)
        feat_list.append(feat)

    def get_explain_dim(self):
        if self.expl_embeds is not None:
            return self.expl_embeds.shape[1]
        # 原本的預設 fallback
        return 1024

    def get_expl_space_dim(self):
        if self.expl_space_emb is not None:
            return self.expl_space_emb.shape[1]
        return 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def normalize_emotion(self, emo_str: str) -> str:
        if emo_str is None:
            return "neutral"
        emo = emo_str.strip().lower()
        emo = self.emotion_alias.get(emo, emo)
        if emo not in self.emo_map:
            return "neutral"
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
            all_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        B = len(batch_graphs)
        S = max_seq_len_in_batch
        T = batch_encodings["input_ids"].shape[-1]

        padded_input_ids = torch.zeros((B, S, T), dtype=torch.long)
        padded_token_mask = torch.zeros((B, S, T), dtype=torch.long)
        padded_speaker_ids = torch.zeros((B, S), dtype=torch.long)
        padded_emotion_ids = torch.zeros((B, S), dtype=torch.long)
        utterance_mask = torch.zeros((B, S), dtype=torch.bool)

        current_node_idx = 0
        for i, g in enumerate(batch_graphs):
            n = g.num_nodes
            padded_input_ids[i, :n] = batch_encodings["input_ids"][
                current_node_idx : current_node_idx + n
            ]
            padded_token_mask[i, :n] = batch_encodings["attention_mask"][
                current_node_idx : current_node_idx + n
            ]
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


# # IMCEE/modules/dataset.py

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
#     """
#     通用的 (embeddings.pt + index.tsv) loader：
#     回傳 (embeddings_tensor 或 None, id -> index 的 dict)
#     """
#     if not embed_path or not index_path:
#         return None, {}

#     embed_path = Path(embed_path)
#     index_path = Path(index_path)

#     if not embed_path.exists() or not index_path.exists():
#         return None, {}

#     embeds = torch.load(embed_path, map_location="cpu")
#     index_map = {}
#     with open(index_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split("\t")
#             if len(parts) != 2:
#                 continue
#             idx_str, edge_id = parts
#             try:
#                 idx = int(idx_str)
#             except ValueError:
#                 continue
#             index_map[edge_id] = idx
#     return embeds, index_map


# def load_llm_shortcuts(path):
#     """
#     讀取 LLM 預測結果。
#     格式: {"id": "te_354__u1__u2", "label": "cause", ...}
#     解析 id -> (conv_id, c_utt_id, t_utt_id)
#     回傳: set 包含所有合法的 (conv_id, c_utt_id, t_utt_id) tuple
#     """
#     shortcuts = set()
#     if not path:
#         return shortcuts
    
#     path_obj = Path(path)
#     if not path_obj.exists():
#         print(f"Warning: Shortcut file not found at {path}")
#         return shortcuts
    
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             try:
#                 obj = json.loads(line)
#                 # 過濾條件：LLM 認為是 cause
#                 if obj.get("label") == "cause":
#                     # 解析 ID: "te_354__u1__u2"
#                     # 假設格式固定為 {conv_id}__{c_id}__{t_id}
#                     raw_id = obj["id"]
#                     parts = raw_id.split("__")
                    
#                     if len(parts) >= 3:
#                         t_id = parts[-1]
#                         c_id = parts[-2]
#                         # 剩下的前面部分組合成 conv_id (以防 conv_id 本身有底線)
#                         conv_id = "__".join(parts[:-2])
                        
#                         shortcuts.add((conv_id, c_id, t_id))
#             except Exception as e:
#                 continue
                
#     print(f"[Dataset] Loaded {len(shortcuts)} valid semantic shortcuts from LLM.")
#     return shortcuts


# class CEEEnd2EndDataset(Dataset):
#     def __init__(
#         self,
#         conversations_path,
#         pairs_path,
#         llm_shortcut_path=None, # [接收路徑]
#         explain_embed_path=None,
#         explain_index_path=None,
#         use_explain=True,
#         expl_space_pt=None,
#         expl_space_tsv=None,
#         use_expl_space=False,
#     ):
#         # 讀對話 & pair
#         self.convs = load_conversations(conversations_path)
#         all_pairs = load_pairs(pairs_path)

#         # 載入 shortcuts
#         self.llm_shortcuts = load_llm_shortcuts(llm_shortcut_path)

#         # 開關
#         self.use_explain = use_explain
#         self.use_expl_space = use_expl_space

#         # 原本的 edge-level explain（edge_features 用）
#         self.expl_embeds, self.expl_index = load_explain_data(
#             explain_embed_path, explain_index_path
#         )

#         # 新的 explanation space（graph/pair-level 向量）
#         self.expl_space_emb, self.expl_space_index = load_explain_data(
#             expl_space_pt, expl_space_tsv
#         )

#         # Speaker / Emotion map
#         self.spk_map = {"A": 0, "B": 1, "None": 2}
#         self.emo_map = {
#             "neutral": 0,
#             "happiness": 1,
#             "anger": 2,
#             "surprise": 3,
#             "disgust": 4,
#             "sadness": 5,
#             "fear": 6,
#         }
#         self.emotion_alias = {
#             "happy": "happiness",
#             "happines": "happiness",
#             "excited": "happiness",
#             "angry": "anger",
#             "sad": "sadness",
#             "surprised": "surprise",
#         }

#         # run_cross_tower 會讀這兩個
#         self.num_speakers = len(self.spk_map)
#         self.num_emotions = len(self.emo_map)

#         self.data_list = []

#         # 依 conv 分組 pairs
#         pairs_by_conv = defaultdict(list)
#         for p in all_pairs:
#             pairs_by_conv[p["conv_id"]].append(p)

#         # 建圖
#         for conv_id, utt_map in tqdm(
#             self.convs.items(), desc="Building Graph"
#         ):
#             # 排序 utterances
#             utterances = sorted(list(utt_map.values()), key=lambda u: u["turn"])
#             id_to_idx = {utt["utt_id"]: i for i, utt in enumerate(utterances)}

#             base_texts = [utt["text"] for utt in utterances]
#             base_speaker_ids = []
#             base_emotion_ids = []

#             for utt in utterances:
#                 spk = utt.get("speaker", "None")
#                 spk = spk if spk in self.spk_map else "None"
#                 base_speaker_ids.append(self.spk_map[spk])

#                 emo = utt.get("emotion", "neutral")
#                 emo = self.normalize_emotion(emo)
#                 base_emotion_ids.append(self.emo_map.get(emo, 0))

#             num_utts = len(base_texts)
#             conv_pairs = pairs_by_conv[conv_id]
#             if not conv_pairs:
#                 continue

#             for p in conv_pairs:
#                 c_idx = id_to_idx.get(p["c_utt_id"])
#                 t_idx = id_to_idx.get(p["t_utt_id"])
#                 pid = f"{conv_id}__{p['c_utt_id']}__{p['t_utt_id']}"

#                 if c_idx is None or t_idx is None:
#                     continue

#                 # ================================
#                 # node construction
#                 # ================================
#                 conv_node_idx = num_utts
#                 cause_node_idx = num_utts + 1
#                 target_node_idx = num_utts + 2
#                 total_nodes = num_utts + 3

#                 current_texts = base_texts + [
#                     "[CLS]",
#                     base_texts[c_idx],
#                     base_texts[t_idx],
#                 ]
#                 current_spk_ids = base_speaker_ids + [self.spk_map["None"]] * 3
#                 current_emo_ids = base_emotion_ids + [self.emo_map["neutral"]] * 3

#                 edge_src, edge_tgt, edge_types, edge_features = [], [], [], []
#                 edge_distances = []

#                 # === edge-level explain feature ===
#                 expl_dim = self.get_explain_dim()
#                 if self.use_explain and self.expl_embeds is not None and pid in self.expl_index:
#                     expl_feat = self.expl_embeds[self.expl_index[pid]]
#                 else:
#                     expl_feat = torch.zeros(expl_dim)
#                 zero_feat = torch.zeros(expl_dim)

#                 # === explanation space (graph-level) 向量 ===
#                 expl_space_dim = self.get_expl_space_dim()
#                 if (
#                     self.use_expl_space
#                     and self.expl_space_emb is not None
#                     and pid in self.expl_space_index
#                 ):
#                     expl_space_vec = self.expl_space_emb[self.expl_space_index[pid]]
#                 else:
#                     expl_space_vec = torch.zeros(expl_space_dim)

#                 # -------------------------------------------------
#                 # 1. Classification Edges (Type 0)
#                 # -------------------------------------------------
#                 # Cause Node -> Cause Utterance (帶 explain)
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=cause_node_idx,
#                     tgt=c_idx,
#                     etype=0,
#                     dist=0,
#                     feat=expl_feat,
#                 )

#                 # Target Node -> Target Utterance (帶 explain)
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=target_node_idx,
#                     tgt=t_idx,
#                     etype=0,
#                     dist=0,
#                     feat=expl_feat,
#                 )

#                 # Cause <-> CauseUtt (雙向補充)
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=cause_node_idx,
#                     tgt=c_idx,
#                     etype=0,
#                     dist=0,
#                     feat=expl_feat,
#                 )
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=c_idx,
#                     tgt=cause_node_idx,
#                     etype=0,
#                     dist=0,
#                     feat=zero_feat,
#                 )

#                 # Target <-> TargetUtt (雙向補充)
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=target_node_idx,
#                     tgt=t_idx,
#                     etype=0,
#                     dist=0,
#                     feat=expl_feat,
#                 )
#                 self._add_edge(
#                     edge_src,
#                     edge_tgt,
#                     edge_types,
#                     edge_distances,
#                     edge_features,
#                     src=t_idx,
#                     tgt=target_node_idx,
#                     etype=0,
#                     dist=0,
#                     feat=zero_feat,
#                 )

#                 # -------------------------------------------------
#                 # 2. Temporal Edges (window <= 2, 單向過去→未來)
#                 # -------------------------------------------------
#                 for i in range(num_utts):
#                     for dist in (1, 2):
#                         j = i + dist
#                         if j < num_utts:
#                             t_type = 1 if dist == 1 else 2
#                             self._add_edge(
#                                 edge_src,
#                                 edge_tgt,
#                                 edge_types,
#                                 edge_distances,
#                                 edge_features,
#                                 src=i,
#                                 tgt=j,
#                                 etype=t_type,
#                                 dist=dist,
#                                 feat=zero_feat,
#                             )

#                 # -------------------------------------------------
#                 # 3. Speaker Edges
#                 # -------------------------------------------------
#                 spk_to_indices = defaultdict(list)
#                 for i in range(num_utts):
#                     spk_to_indices[base_speaker_ids[i]].append(i)

#                 for spk, indices in spk_to_indices.items():
#                     if spk == self.spk_map["None"]:
#                         continue
#                     for k in range(len(indices) - 1):
#                         u_curr = indices[k]
#                         u_next = indices[k + 1]
#                         dist = u_next - u_curr
#                         self._add_edge(
#                             edge_src,
#                             edge_tgt,
#                             edge_types,
#                             edge_distances,
#                             edge_features,
#                             src=u_curr,
#                             tgt=u_next,
#                             etype=4,
#                             dist=dist,
#                             feat=zero_feat,
#                         )

#                 # -------------------------------------------------
#                 # 4. Global Edges
#                 # -------------------------------------------------
#                 for i in range(num_utts):
#                     self._add_edge(
#                         edge_src,
#                         edge_tgt,
#                         edge_types,
#                         edge_distances,
#                         edge_features,
#                         src=conv_node_idx,
#                         tgt=i,
#                         etype=5,
#                         dist=0,
#                         feat=zero_feat,
#                     )
#                     self._add_edge(
#                         edge_src,
#                         edge_tgt,
#                         edge_types,
#                         edge_distances,
#                         edge_features,
#                         src=i,
#                         tgt=conv_node_idx,
#                         etype=5,
#                         dist=0,
#                         feat=zero_feat,
#                     )

#                 # -------------------------------------------------
#                 # 5. LLM Semantic Shortcuts (Type 6) [新增區塊]
#                 # -------------------------------------------------
#                 if self.llm_shortcuts:
#                     # 遍歷所有可能的節點對 (i -> j)
#                     # 因為捷徑可能有向 (Cause -> Target)
#                     for i in range(num_utts):
#                         for j in range(num_utts):
#                             if i == j: continue
                            
#                             dist = abs(j - i)
#                             # 條件：距離 >= 3 (避免與局部 Temporal Edge 重疊)
#                             if dist >= 3:
#                                 u_c_id = utterances[i]["utt_id"]
#                                 u_t_id = utterances[j]["utt_id"]
                                
#                                 # 檢查是否存在於 LLM 的 cause 列表中
#                                 if (conv_id, u_c_id, u_t_id) in self.llm_shortcuts:
#                                     self._add_edge(
#                                         edge_src,
#                                         edge_tgt,
#                                         edge_types,
#                                         edge_distances,
#                                         edge_features,
#                                         src=i,
#                                         tgt=j,
#                                         etype=6, # Type 6: Semantic Shortcut
#                                         dist=dist,
#                                         feat=zero_feat,
#                                     )

#                 # ==========================================
#                 # Build Data Object
#                 # ==========================================
#                 target_indices_tensor = torch.tensor(
#                     [[cause_node_idx, target_node_idx]], dtype=torch.long
#                 )

#                 graph_data = Data(
#                     texts=current_texts,
#                     speaker_ids=torch.tensor(current_spk_ids, dtype=torch.long),
#                     emotion_ids=torch.tensor(current_emo_ids, dtype=torch.long),
#                     edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
#                     edge_types=torch.tensor(edge_types, dtype=torch.long),
#                     edge_features=torch.stack(edge_features),
#                     edge_distance=torch.tensor(edge_distances, dtype=torch.long),
#                     edge_label=torch.tensor([float(p["label"])], dtype=torch.float),
#                     num_nodes=total_nodes,
#                     target_node_indices=target_indices_tensor,
#                     # 新增：explanation space 向量
#                     expl_space_vec=expl_space_vec,
#                     pair_uttpos=torch.tensor([[c_idx, t_idx]], dtype=torch.long),
#                 )
#                 self.data_list.append(graph_data)

#     def _add_edge(
#         self,
#         src_list,
#         tgt_list,
#         type_list,
#         dist_list,
#         feat_list,
#         src,
#         tgt,
#         etype,
#         dist,
#         feat,
#     ):
#         src_list.append(src)
#         tgt_list.append(tgt)
#         type_list.append(etype)
#         dist_list.append(dist)
#         feat_list.append(feat)

#     def get_explain_dim(self):
#         if self.expl_embeds is not None:
#             return self.expl_embeds.shape[1]
#         # 原本的預設 fallback
#         return 1024

#     def get_expl_space_dim(self):
#         if self.expl_space_emb is not None:
#             return self.expl_space_emb.shape[1]
#         return 0

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         return self.data_list[idx]

#     def normalize_emotion(self, emo_str: str) -> str:
#         if emo_str is None:
#             return "neutral"
#         emo = emo_str.strip().lower()
#         emo = self.emotion_alias.get(emo, emo)
#         if emo not in self.emo_map:
#             return "neutral"
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
#             all_texts,
#             padding="longest",
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         )

#         B = len(batch_graphs)
#         S = max_seq_len_in_batch
#         T = batch_encodings["input_ids"].shape[-1]

#         padded_input_ids = torch.zeros((B, S, T), dtype=torch.long)
#         padded_token_mask = torch.zeros((B, S, T), dtype=torch.long)
#         padded_speaker_ids = torch.zeros((B, S), dtype=torch.long)
#         padded_emotion_ids = torch.zeros((B, S), dtype=torch.long)
#         utterance_mask = torch.zeros((B, S), dtype=torch.bool)

#         current_node_idx = 0
#         for i, g in enumerate(batch_graphs):
#             n = g.num_nodes
#             padded_input_ids[i, :n] = batch_encodings["input_ids"][
#                 current_node_idx : current_node_idx + n
#             ]
#             padded_token_mask[i, :n] = batch_encodings["attention_mask"][
#                 current_node_idx : current_node_idx + n
#             ]
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