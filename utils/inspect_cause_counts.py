#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect cause count distribution (correct version):
- Correctly treats each target as (conv_id, t_utt_id)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, required=True,
                    choices=["train", "valid", "test"],
                    help="Which split to analyze")
    args = ap.parse_args()

    data_root = Path("data/preprocess")
    pairs_path = data_root / f"pairs_{args.split}.jsonl"

    print(f"[CauseCount] pairs = {pairs_path}")

    pairs = load_pairs(pairs_path)

    # === 正確 target key： (conv_id, t_utt_id) ===
    tgt_to_count = defaultdict(int)

    for p in pairs:
        tgt_key = f"{p['conv_id']}__{p['t_utt_id']}"
        if p["label"] == 1:
            tgt_to_count[tgt_key] += 1

    # 所有 target（包含 k=0）
    all_targets = sorted(list(set([
        f"{p['conv_id']}__{p['t_utt_id']}" for p in pairs
    ])))

    print(f"\nTotal targets appearing in {args.split} pairs: {len(all_targets)}")

    # === Histogram ===
    hist = defaultdict(int)
    for tgt in all_targets:
        k = tgt_to_count[tgt]
        hist[k] += 1

    print("\n===== Cause Count Histogram (targets in pairs only) =====")
    for k in sorted(hist.keys()):
        print(f"  k = {k:2d} : {hist[k]:6d} nodes ({hist[k] / len(all_targets) * 100:6.2f}%)")

    # Summary
    k0 = hist.get(0, 0)
    k1 = hist.get(1, 0)
    k2p = sum(cnt for kk, cnt in hist.items() if kk >= 2)
    k3p = sum(cnt for kk, cnt in hist.items() if kk >= 3)

    print("\n===== Summary =====")
    print(f"  k = 0      : {k0:6d} nodes ({k0 / len(all_targets) * 100:6.2f}%)")
    print(f"  k = 1      : {k1:6d} nodes ({k1 / len(all_targets) * 100:6.2f}%)")
    print(f"  k >= 2     : {k2p:6d} nodes ({k2p / len(all_targets) * 100:6.2f}%)")
    print(f"  k >= 3     : {k3p:6d} nodes ({k3p / len(all_targets) * 100:6.2f}%)")


if __name__ == "__main__":
    main()
