#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統計每種 target emotion 平均有多少前置因果句 (label=1)
依據 (conv_id, t_utt_id) 分組，計算每個 target 有多少 causal
最後再對 emotion 做平均
"""

import json
from collections import defaultdict
import argparse


def compute_avg_causal(pairs_path):
    # key: (conv_id, t_utt_id) → count of causal
    target_causal_counts = defaultdict(int)
    target_emotions = {}  # (conv_id, t_utt_id) → emotion

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())

            key = (obj["conv_id"], obj["t_utt_id"])
            emo = obj["meta"]["t_emotion"]
            label = obj["label"]

            # 記錄 target emotion
            target_emotions[key] = emo

            # causal pair
            if label == 1:
                target_causal_counts[key] += 1

    # emotion → [list of causal-counts]
    emo_to_counts = defaultdict(list)
    for key, count in target_causal_counts.items():
        emo = target_emotions[key]
        emo_to_counts[emo].append(count)

    # compute mean
    emo_to_avg = {}
    for emo, counts in emo_to_counts.items():
        if len(counts) == 0:
            emo_to_avg[emo] = 0.0
        else:
            emo_to_avg[emo] = sum(counts) / len(counts)

    return emo_to_avg, emo_to_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        required=True,
        help="Path to pairs_xxx.jsonl"
    )
    args = ap.parse_args()

    emo_to_avg, emo_to_counts = compute_avg_causal(args.pairs)

    print("\n=== 每種情緒平均因果句數量 ===")
    for emo in sorted(emo_to_avg):
        avg = emo_to_avg[emo]
        print(f"{emo}: {avg:.4f}  (samples={len(emo_to_counts[emo])})")
    print("================================\n")


if __name__ == "__main__":
    main()
