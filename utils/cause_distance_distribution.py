#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統計情緒的 cause 在 target 前的第幾句最常出現
distance = t_turn - c_turn
例如：
target turn=10、cause turn=7 → distance=3

輸出：
emotion → 前 K 名距離（含出現次數與比例）
"""

import json
from collections import defaultdict, Counter
import argparse


def load_turn_map(conversations_path):
    """
    建立 (conv_id, utt_id) → turn 編號的查詢表
    """
    turn_map = {}

    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            conv_id = obj["conv_id"]
            for utt in obj["utterances"]:
                utt_id = utt["utt_id"]
                turn = utt["turn"]
                turn_map[(conv_id, utt_id)] = turn

    return turn_map


def compute_distance_distribution(pairs_path, turn_map):
    # emotion → list of distances
    emo_distances = defaultdict(list)

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] != 1:
                continue

            emo = obj["meta"]["t_emotion"]

            conv_id = obj["conv_id"]
            t_id = obj["t_utt_id"]
            c_id = obj["c_utt_id"]

            t_turn = turn_map[(conv_id, t_id)]
            c_turn = turn_map[(conv_id, c_id)]
            dist = t_turn - c_turn

            # 只統計前面的句子（正距離）
            if dist > 0:
                emo_distances[emo].append(dist)

    return emo_distances


def top_k_distribution(emo_distances, k=3):
    """
    回傳每種 emotion 最常見的 top-k cause 距離
    """
    emo_topk = {}

    for emo, dists in emo_distances.items():
        counter = Counter(dists)
        total = sum(counter.values())

        # 取前 k 名（依照 count 多到少）
        top_items = counter.most_common(k)

        emo_topk[emo] = {
            "top": [
                {
                    "distance": dist,
                    "count": cnt,
                    "ratio": cnt / total
                }
                for dist, cnt in top_items
            ],
            "total_causal": total
        }

    return emo_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True, help="conversations.jsonl")
    ap.add_argument("--pairs", required=True, help="pairs_xxx.jsonl")
    ap.add_argument("--k", type=int, default=3, help="Top-K distances")
    args = ap.parse_args()

    turn_map = load_turn_map(args.conversations)
    emo_distances = compute_distance_distribution(args.pairs, turn_map)
    results = top_k_distribution(emo_distances, args.k)

    print("\n=== 各情緒的 Top-{} cause 距離分佈 ===".format(args.k))
    for emo in sorted(results.keys()):
        print(f"\nEmotion: {emo} (total causal = {results[emo]['total_causal']})")
        for item in results[emo]["top"]:
            print(f"  distance = {item['distance']}, "
                  f"count = {item['count']}, "
                  f"ratio = {item['ratio']*100:.2f}%")
    print("\n=========================================\n")


if __name__ == "__main__":
    main()
