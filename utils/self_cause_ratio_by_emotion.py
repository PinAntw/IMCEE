#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統計各情緒中 target utterance 本身
是否經常出現在 causal 中（self-cause）

輸出：
emotion → 自己就是原因的比例 (%)
"""

import json
from collections import defaultdict
import argparse


def compute_self_cause_ratio(pairs_path):
    # emotion → total causal count
    emo_total = defaultdict(int)

    # emotion → self-cause count
    emo_self = defaultdict(int)

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())

            emo = obj["meta"]["t_emotion"]
            label = obj["label"]

            if label != 1:
                continue

            t_id = obj["t_utt_id"]
            c_id = obj["c_utt_id"]

            # 所有 causal
            emo_total[emo] += 1

            # 自己就是 cause
            if t_id == c_id:
                emo_self[emo] += 1

    # 計算比例
    emo_ratio = {}
    for emo in emo_total:
        total = emo_total[emo]
        self_cnt = emo_self.get(emo, 0)
        emo_ratio[emo] = self_cnt / total if total > 0 else 0.0

    return emo_ratio, emo_total, emo_self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Path to pairs_xxx.jsonl")
    args = ap.parse_args()

    ratio, total, self_cnt = compute_self_cause_ratio(args.pairs)

    print("\n=== 情緒的 self-cause 比例（target 自己就是原因） ===")
    for emo in sorted(ratio.keys()):
        print(f"{emo}: {ratio[emo]*100:.2f}% "
              f"(self={self_cnt[emo]}, total={total[emo]})")
    print("====================================================\n")


if __name__ == "__main__":
    main()
