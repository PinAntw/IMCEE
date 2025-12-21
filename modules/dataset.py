# IMCEE/modules/dataset.py
# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ============================================================
# IO helpers
# ============================================================

def load_conversations(path):
    """
    讀 conversations jsonl：
    每行一個 conv，內含 utterances list
    回傳：convs[conv_id] = {utt_id: utt_obj}
    """
    convs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            utt_map = {utt["utt_id"]: utt for utt in obj["utterances"]}
            convs[obj["conv_id"]] = utt_map
    return convs


def load_pairs(path):
    """
    讀 pairs jsonl：
    每行一個 pair 樣本（含 conv_id, c_utt_id, t_utt_id, label）
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def load_explain_data(embed_path, index_path):
    """
    通用 (embeddings.pt + index.tsv) loader

    - embeddings.pt: tensor [N, D]
    - index.tsv: "idx \t edge_id"
      其中 edge_id 你目前用 pid = f"{conv_id}__{c_utt_id}__{t_utt_id}"
    回傳：
      embeds (tensor or None), index_map (dict edge_id -> idx)
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
    讀 LLM shortcut（你現在有載入，但 dataset 端沒有真正用進建圖）
    - 這裡保留以維持你現有 pipeline；未來要用 shortcut 再接入 edge construction。

    回傳：set((conv_id, c_id, t_id))
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
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("label") == "cause":
                    raw_id = obj["id"]
                    parts = raw_id.split("__")
                    if len(parts) >= 3:
                        t_id = parts[-1]
                        c_id = parts[-2]
                        conv_id = "__".join(parts[:-2])
                        shortcuts.add((conv_id, c_id, t_id))
            except Exception:
                continue

    print(f"[Dataset] Loaded {len(shortcuts)} valid semantic shortcuts from LLM.")
    return shortcuts


# ============================================================
# Dataset
# ============================================================

class CEEEnd2EndDataset(Dataset):
    """
    Pair-level graph dataset:
    - 每一個 pair 生成一張圖
    - Node ordering 固定：
        0..num_utts-1 : utterance nodes (依 turn 排序)
        num_utts      : conv node
        num_utts + 1  : cause super node
        num_utts + 2  : target super node
      total_nodes = num_utts + 3

    Edge types：
      Type 0 : utterance -> super (anchoring / binding)   [只保留 utt->super，移除 super->utt]
      Type 1 : temporal distance = 1
      Type 2 : temporal distance in {2,3,4}
      Type 3 : context window edges (preceding utterances -> super), w=4
      Type 4 : speaker chain edges (同 speaker 只連相鄰出現)
      Type 5 : global edges (conv <-> utterances)
    """

    def __init__(
        self,
        conversations_path,
        pairs_path,
        llm_shortcut_path=None,
        explain_embed_path=None,
        explain_index_path=None,
        use_explain=True,
        expl_space_pt=None,
        expl_space_tsv=None,
        use_expl_space=False,
    ):
        # 讀對話 & pairs
        self.convs = load_conversations(conversations_path)
        all_pairs = load_pairs(pairs_path)

        # （目前未使用到建圖，但保留）
        self.llm_shortcuts = load_llm_shortcuts(llm_shortcut_path)

        # 開關
        self.use_explain = use_explain
        self.use_expl_space = use_expl_space

        # explain feature（edge-level / pair-level）
        self.expl_embeds, self.expl_index = load_explain_data(explain_embed_path, explain_index_path)
        self.expl_space_emb, self.expl_space_index = load_explain_data(expl_space_pt, expl_space_tsv)

        # ------------------------------------------------------------
        # ID mapping
        # ------------------------------------------------------------
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

        # Padding ID：放在最後一個位置
        self.spk_pad_id = len(self.spk_map)  # 3
        self.emo_pad_id = len(self.emo_map)  # 7

        # Embedding vocab size 必須包含 padding id
        self.num_speakers = self.spk_pad_id + 1  # 4
        self.num_emotions = self.emo_pad_id + 1  # 8

        print(f"[Dataset] Config: num_speakers={self.num_speakers} (pad_id={self.spk_pad_id})")
        print(f"[Dataset] Config: num_emotions={self.num_emotions} (pad_id={self.emo_pad_id})")

        # ------------------------------------------------------------
        # group pairs by conv_id
        # ------------------------------------------------------------
        pairs_by_conv = defaultdict(list)
        for p in all_pairs:
            pairs_by_conv[p["conv_id"]].append(p)

        self.data_list = []

        # ------------------------------------------------------------
        # Build graphs
        # ------------------------------------------------------------
        for conv_id, utt_map in tqdm(self.convs.items(), desc="Building Graph"):
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
            conv_pairs = pairs_by_conv.get(conv_id, [])
            if not conv_pairs:
                continue

            # 固定 explain 維度（沒有 explain 時也要能 stack）
            expl_dim = self.get_explain_dim()
            expl_space_dim = self.get_expl_space_dim()

            for p in conv_pairs:
                c_idx = id_to_idx.get(p["c_utt_id"])
                t_idx = id_to_idx.get(p["t_utt_id"])
                if c_idx is None or t_idx is None:
                    continue

                # pair id: 用於查 explain 向量
                pid = f"{conv_id}__{p['c_utt_id']}__{p['t_utt_id']}"

                # ---------------------------
                # Node indexing
                # ---------------------------
                conv_node_idx = num_utts
                cause_node_idx = num_utts + 1
                target_node_idx = num_utts + 2
                total_nodes = num_utts + 3

                # ---------------------------
                # Node texts
                # ---------------------------
                # super nodes 用固定字串，避免 encoder 直接看到原句內容形成捷徑
                current_texts = base_texts + [
                    "<CONV_NODE>",
                    "<CAUSE_NODE>",
                    "<TARGET_NODE>",
                ]

                # ---------------------------
                # Node speaker/emotion ids
                # ---------------------------
                current_spk_ids = base_speaker_ids + [self.spk_map["None"]] * 3
                current_emo_ids = base_emotion_ids + [self.emo_map["neutral"]] * 3

                spk_tensor = torch.tensor(current_spk_ids, dtype=torch.long)
                emo_tensor = torch.tensor(current_emo_ids, dtype=torch.long)

                # ---------------------------
                # Node type (for debugging / future masking)
                # 0=utt, 1=conv, 2=cause, 3=target
                # ---------------------------
                node_type = torch.zeros(total_nodes, dtype=torch.long)
                node_type[conv_node_idx] = 1
                node_type[cause_node_idx] = 2
                node_type[target_node_idx] = 3

                # ---------------------------
                # Explain features
                # ---------------------------
                if self.use_explain and (self.expl_embeds is not None) and (pid in self.expl_index):
                    expl_feat = self.expl_embeds[self.expl_index[pid]].to(dtype=torch.float)
                else:
                    expl_feat = torch.zeros(expl_dim, dtype=torch.float)

                zero_feat = torch.zeros(expl_dim, dtype=torch.float)

                if self.use_expl_space and (self.expl_space_emb is not None) and (pid in self.expl_space_index):
                    expl_space_vec = self.expl_space_emb[self.expl_space_index[pid]].to(dtype=torch.float)
                else:
                    expl_space_vec = torch.zeros(expl_space_dim, dtype=torch.float)

                # ---------------------------
                # Edge construction
                # ---------------------------
                edge_src, edge_tgt, edge_types, edge_features, edge_distances = [], [], [], [], []

                # # 刪掉了(1) Type 0: utterance -> super anchoring edges
                # # 只保留 utt -> super（移除 super -> utt）
                # self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                #                c_idx, cause_node_idx, 0, 0, expl_feat)
                # self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                #                t_idx, target_node_idx, 0, 0, expl_feat)

                # (1.5) Type 3: MPEG-style context window edges (preceding utterances -> super), w=4
                # 注意：不包含 c_idx/t_idx 本身，避免與 Type 0 重複
                # (1) Type 3: context window edges (including self) -> super, w=4
                w = 4

                c_start = max(0, c_idx - w)
                for u in range(c_start, c_idx + 1):  # include self
                    feat = expl_feat if (u == c_idx) else zero_feat
                    self._add_edge(
                        edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                        u, cause_node_idx, 3, c_idx - u, feat
                    )

                t_start = max(0, t_idx - w)
                for u in range(t_start, t_idx + 1):  # include self
                    feat = expl_feat if (u == t_idx) else zero_feat
                    self._add_edge(
                        edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                        u, target_node_idx, 3, t_idx - u, feat
                    )


                # (2) Temporal edges (Type 1/2): utterance -> utterance, distance up to 4
                for i in range(num_utts):
                    for dist in (1, 2, 3, 4):
                        j = i + dist
                        if j < num_utts:
                            t_type = 1 if dist == 1 else 2
                            self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                           i, j, t_type, dist, zero_feat)

                # (3) Speaker edges (Type 4) - Chain only (同 speaker 只連相鄰出現)
                spk_to_indices = defaultdict(list)
                for i in range(num_utts):
                    spk_to_indices[base_speaker_ids[i]].append(i)

                for spk, indices in spk_to_indices.items():
                    if spk == self.spk_map["None"]:
                        continue
                    for k in range(len(indices) - 1):
                        u = indices[k]
                        v = indices[k + 1]
                        self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                       u, v, 4, v - u, zero_feat)

                # (4) Global edges (Type 5): conv <-> (utterances + super nodes)
                # 讓 conv 成為全局 hub：連 utterances + cause/target super nodes（雙向）
                for i in range(num_utts):
                    self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                conv_node_idx, i, 5, 0, zero_feat)
                    self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                                i, conv_node_idx, 5, 0, zero_feat)

                # 新增：conv <-> cause/target super nodes
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                            conv_node_idx, cause_node_idx, 5, 0, zero_feat)
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                            cause_node_idx, conv_node_idx, 5, 0, zero_feat)

                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                            conv_node_idx, target_node_idx, 5, 0, zero_feat)
                self._add_edge(edge_src, edge_tgt, edge_types, edge_distances, edge_features,
                            target_node_idx, conv_node_idx, 5, 0, zero_feat)

                # target_node_indices：你後續 model 取這兩個 index 做分類
                target_indices_tensor = torch.tensor([[cause_node_idx, target_node_idx]], dtype=torch.long)

                # pair 位置（local indices）
                pair_uttpos = torch.tensor([[c_idx, t_idx]], dtype=torch.long)
                pair_utt_index = pair_uttpos.clone()

                graph_data = Data(
                    # raw texts, for tokenizer in collate
                    texts=current_texts,

                    # node attributes
                    speaker_ids=spk_tensor,
                    emotion_ids=emo_tensor,
                    node_type=node_type,
                    num_utts=torch.tensor([num_utts], dtype=torch.long),

                    # graph structure
                    edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
                    edge_types=torch.tensor(edge_types, dtype=torch.long),
                    edge_distance=torch.tensor(edge_distances, dtype=torch.long),
                    edge_features=torch.stack(edge_features, dim=0).to(dtype=torch.float),

                    # labels / indices
                    edge_label=torch.tensor([float(p["label"])], dtype=torch.float),
                    num_nodes=total_nodes,
                    target_node_indices=target_indices_tensor,
                    pair_uttpos=pair_uttpos,
                    pair_utt_index=pair_utt_index,

                    # optional pair-level vector (teacher space)
                    expl_space_vec=expl_space_vec,
                )

                self.data_list.append(graph_data)

    # ---------------------------
    # helpers
    # ---------------------------
    @staticmethod
    def _add_edge(src_list, tgt_list, type_list, dist_list, feat_list, src, tgt, etype, dist, feat):
        """
        統一 append edge 的工具函數，避免各處手寫漏欄位。
        feat: torch.Tensor [expl_dim]
        """
        src_list.append(int(src))
        tgt_list.append(int(tgt))
        type_list.append(int(etype))
        dist_list.append(int(dist))
        feat_list.append(feat)

    def get_explain_dim(self):
        if self.expl_embeds is not None:
            return int(self.expl_embeds.shape[1])
        return 1024

    def get_expl_space_dim(self):
        if self.expl_space_emb is not None:
            return int(self.expl_space_emb.shape[1])
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


# ============================================================
# Collate
# ============================================================

class End2EndCollate:
    """
    將一批 Data graphs 打包成 Batch，並做 tokenizer encoding。

    關鍵修正：
    1) padded_input_ids 使用 tokenizer.pad_token_id，而不是 0
    2) 拆分 mask 語意：
       - node_mask：哪些位置真的有節點（含 super nodes）
       - utt_mask：哪些位置是 utterance 節點
       - 為相容你現有 code：batch.utterance_mask 仍指向 node_mask（歷史遺留）
    3) 另外把 pair_utt_index 整理成 batch.pair_utt_index: [B,2]（供 MPEG-style init 使用）
    """

    def __init__(self, tokenizer, max_len=128, spk_pad_id=3, emo_pad_id=7):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.spk_pad_id = spk_pad_id
        self.emo_pad_id = emo_pad_id

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is None. Please set pad_token for tokenizer.")
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def __call__(self, batch_graphs):
        # 先用 PyG 組 Batch（會處理 edge_index offset 等）
        batch = Batch.from_data_list(batch_graphs)

        # ------------------------------------------------------------
        # 1) Collect all texts (node-level) for tokenization
        # ------------------------------------------------------------
        all_texts = []
        max_nodes_in_batch = 0

        for g in batch_graphs:
            all_texts.extend(g.texts)
            max_nodes_in_batch = max(max_nodes_in_batch, len(g.texts))

        batch_encodings = self.tokenizer(
            all_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        B = len(batch_graphs)
        S = max_nodes_in_batch
        T = batch_encodings["input_ids"].shape[-1]

        # ------------------------------------------------------------
        # 2) Allocate padded tensors
        # ------------------------------------------------------------
        padded_input_ids = torch.full((B, S, T), self.pad_token_id, dtype=torch.long)
        padded_token_mask = torch.zeros((B, S, T), dtype=torch.long)

        padded_speaker_ids = torch.full((B, S), self.spk_pad_id, dtype=torch.long)
        padded_emotion_ids = torch.full((B, S), self.emo_pad_id, dtype=torch.long)

        node_mask = torch.zeros((B, S), dtype=torch.bool)
        utt_mask = torch.zeros((B, S), dtype=torch.bool)

        node_type_padded = torch.full((B, S), -1, dtype=torch.long)

        # MPEG-style init 需要：每張圖的 (c_idx, t_idx)
        pair_utt_index_batch = torch.full((B, 2), -1, dtype=torch.long)

        # ------------------------------------------------------------
        # 3) Fill per-graph
        # ------------------------------------------------------------
        current_node_idx = 0
        for i, g in enumerate(batch_graphs):
            n = int(g.num_nodes)

            # texts -> token
            padded_input_ids[i, :n] = batch_encodings["input_ids"][current_node_idx: current_node_idx + n]
            padded_token_mask[i, :n] = batch_encodings["attention_mask"][current_node_idx: current_node_idx + n]

            padded_speaker_ids[i, :n] = g.speaker_ids
            padded_emotion_ids[i, :n] = g.emotion_ids

            node_mask[i, :n] = True

            num_utts = int(g.num_utts.item()) if hasattr(g, "num_utts") else (n - 3)
            utt_mask[i, :num_utts] = True

            if hasattr(g, "node_type"):
                node_type_padded[i, :n] = g.node_type

            if hasattr(g, "pair_utt_index"):
                pair_utt_index_batch[i] = g.pair_utt_index[0].to(dtype=torch.long)
            elif hasattr(g, "pair_uttpos"):
                pair_utt_index_batch[i] = g.pair_uttpos[0].to(dtype=torch.long)

            current_node_idx += n

        # ------------------------------------------------------------
        # 4) Attach to batch
        # ------------------------------------------------------------
        batch.input_ids = padded_input_ids
        batch.token_mask = padded_token_mask
        batch.speaker_ids_padded = padded_speaker_ids
        batch.emotion_ids_padded = padded_emotion_ids

        # 舊欄位名（歷史遺留）= node exists mask
        batch.utterance_mask = node_mask

        # 新增乾淨欄位
        batch.node_mask = node_mask
        batch.utt_mask = utt_mask
        batch.node_type_padded = node_type_padded

        # MPEG-style init 需要
        batch.pair_utt_index = pair_utt_index_batch

        return batch
