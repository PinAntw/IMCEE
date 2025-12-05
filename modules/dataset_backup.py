#version 3 IMCEE/modules/dataset.py
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
    embeds = torch.load(embed_path, map_location="cpu")
    index_map = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                index_map[parts[1]] = int(parts[0])
    return embeds, index_map


class CEEEnd2EndDataset(Dataset):
    def __init__(self, conversations_path, pairs_path, explain_embed_path, explain_index_path, use_explain=True):
        self.convs = load_conversations(conversations_path)
        all_pairs = load_pairs(pairs_path)
        self.use_explain = use_explain

        self.expl_embeds, self.expl_index = load_explain_data(explain_embed_path, explain_index_path)
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

        # alias
        self.emotion_alias = {
            "happy": "happiness",
            "happines": "happiness",
            "excited": "happiness",
            "angry": "anger",
            "sad": "sadness",
            "surprised": "surprise",
        }

        self.num_speakers = len(self.spk_map)
        self.num_emotions = len(self.emo_map) 

        self.data_list = []
        for conv_id, utterances_map in tqdm(self.convs.items(), desc="Building Graphs"):
            utterances = sorted(list(utterances_map.values()), key=lambda u: u['turn'])
            id_to_idx = {utt['utt_id']: i for i, utt in enumerate(utterances)}

            texts = [utt['text'] for utt in utterances]
            
            speaker_ids = []
            emotion_ids = []

            for utt in utterances:
                # ----- Speaker -----
                spk_raw = utt.get("speaker", "None")
                spk = spk_raw if spk_raw in self.spk_map else "None"
                speaker_ids.append(self.spk_map[spk])

                # ----- Emotion -----
                emo_raw = utt.get("emotion", "neutral")
                emo_norm = self.normalize_emotion(emo_raw)
                emo_final = self.emo_map.get(emo_norm, self.emo_map["neutral"])
                emotion_ids.append(emo_final)


            edge_src, edge_tgt, edge_labels, edge_features, edge_distance, edge_types = [], [], [], [], [], []

            # === 任務邊 (candidate → target) ===
            for p in all_pairs:
                if p['conv_id'] != conv_id:
                    continue
                c_idx = id_to_idx.get(p['c_utt_id'])
                t_idx = id_to_idx.get(p['t_utt_id'])
                pid = f"{p['conv_id']}__{p['c_utt_id']}__{p['t_utt_id']}"
                if c_idx is not None and t_idx is not None and pid in self.expl_index:
                    edge_src.append(c_idx)
                    edge_tgt.append(t_idx)
                    edge_labels.append(int(p['label']))
                    if self.use_explain:
                        edge_features.append(self.expl_embeds[self.expl_index[pid]])
                    else:
                        edge_features.append(torch.zeros_like(self.expl_embeds[0]))
                    edge_distance.append(abs(c_idx - t_idx))
                    edge_types.append(0)  # 0 = 任務邊

            # === 時序邊 (U₁→U₂→...→Uₙ) ===
            num_utts = len(utterances)
            for i in range(num_utts - 1):
                edge_src.append(i)
                edge_tgt.append(i + 1)
                edge_labels.append(0.0)
                edge_features.append(torch.zeros_like(self.expl_embeds[0]))
                edge_distance.append(1)
                edge_types.append(1)  # 1 = 時序邊
            if not edge_src:
                continue

            # === 去除重複邊 ===
            edge_pairs = list(zip(edge_src, edge_tgt))
            pair_to_first_idx = {}
            for idx, pair in enumerate(edge_pairs):
                if pair not in pair_to_first_idx:
                    pair_to_first_idx[pair] = idx

            unique_indices = sorted(pair_to_first_idx.values())
            
            # num_before = len(edge_pairs)
            # num_after = len(unique_indices)
            # num_removed = num_before - num_after
            # if num_removed > 0:
            #     print(f"[去重] conv_id={conv_id} 刪除 {num_removed} 條重複邊 (原本 {num_before} → 現在 {num_after})")

            edge_src = [edge_src[i] for i in unique_indices]
            edge_tgt = [edge_tgt[i] for i in unique_indices]
            edge_labels = [edge_labels[i] for i in unique_indices]
            edge_distance = [edge_distance[i] for i in unique_indices]
            edge_features = [edge_features[i] for i in unique_indices]
            edge_types = [edge_types[i] for i in unique_indices]

            # === 建立任務邊遮罩 (只保留 edge_type == 0) ===
            edge_task_mask = torch.tensor(
                [t == 0 for t in edge_types], dtype=torch.bool
            )

            # === 建立圖 ===
            graph_data = Data(
                texts=texts,
                speaker_ids=torch.tensor(speaker_ids, dtype=torch.long),
                emotion_ids=torch.tensor(emotion_ids, dtype=torch.long),
                edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
                edge_features=torch.stack(edge_features),
                edge_distance=torch.tensor(edge_distance, dtype=torch.long),
                edge_labels=torch.tensor(edge_labels, dtype=torch.float),
                edge_types=torch.tensor(edge_types, dtype=torch.long),
                num_nodes=len(utterances),
                edge_task_mask=edge_task_mask,
            )
            self.data_list.append(graph_data)

    def __len__(self):
        return len(self.data_list)

    def get_explain_dim(self):
        return self.expl_embeds.shape[1] if self.expl_embeds is not None else 1024

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    # ====================================
    # Emotion Normalizer（最小增量）
    # ====================================
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
