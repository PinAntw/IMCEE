#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統計：
1. 整體：cause 是情緒句 vs 非情緒句 的比例
2. 個別情緒：每種 target emotion 的 cause 裡，情緒 cause 比率

需要資料格式：
{
  "conv_id": "...",
  "t_utt_id": "...",
  "meta": {
      "t_emotion": "...",
      "c_emotion": "..."    # cause 的情緒（必要）
  },
  "label": 1 or 0
}
"""

import json
from collections import defaultdict
import argparse

# 定義哪些是情緒
EMOTIONS = {"anger", "sadness", "happiness", "disgust", "surprise",
            "fear", "excited", "frustration"}  # 根據你的資料集調整


def is_emotional(e):
    return e.lower() in EMOTIONS


def compute_stats(pairs_path):
    # overall counters
    total_causal = 0
    emotional_causal = 0

    # for each target emotion
    emo_stats = defaultdict(lambda: {"total": 0, "emotional": 0})

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            label = obj["label"]

            # not a causal pair → skip
            if label != 1:
                continue

            t_emo = obj["meta"]["t_emotion"]
            c_emo = obj["meta"]["c_emotion"]  # cause emotion

            # overall
            total_causal += 1
            if is_emotional(c_emo):
                emotional_causal += 1

            # per-emotion stats
            emo_stats[t_emo]["total"] += 1
            if is_emotional(c_emo):
                emo_stats[t_emo]["emotional"] += 1

    # final ratio
    overall_ratio = emotional_causal / total_causal if total_causal > 0 else 0.0

    per_emo_ratio = {}
    for emo, d in emo_stats.items():
        if d["total"] == 0:
            per_emo_ratio[emo] = 0.0
        else:
            per_emo_ratio[emo] = d["emotional"] / d["total"]

    return overall_ratio, per_emo_ratio, emo_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Path to pairs_xxx.jsonl")
    args = ap.parse_args()

    overall_ratio, per_emo_ratio, raw = compute_stats(args.pairs)

    print("\n=== 整體：情緒 cause 比率 ===")
    print(f"{overall_ratio:.4f}")
    print("=============================\n")

    print("=== 個別 target emotion 的情緒 cause 比率 ===")
    for emo in sorted(per_emo_ratio):
        r = per_emo_ratio[emo]
        print(f"{emo}: {r:.4f}  (causal={raw[emo]['total']}, emotional={raw[emo]['emotional']})")
    print("===========================================\n")


if __name__ == "__main__":
    main()
