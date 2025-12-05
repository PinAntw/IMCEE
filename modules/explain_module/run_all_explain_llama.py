#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_explain_llama.py

一鍵呼叫 generator_llama.py，依序對 train / valid / test 產生 LLM explanation。

預設路徑對應：
- conversations: data/preprocess/conversations.jsonl
- pairs_train:  data/preprocess/pairs_train.jsonl
- pairs_valid:  data/preprocess/pairs_valid.jsonl
- pairs_test:   data/preprocess/pairs_test.jsonl
- outputs:      outputs/explain_{split}_results.{jsonl,txt}

使用範例（在專案根目錄）:
    python3 run_all_explain_llama.py
或指定 model / batch:
    python3 run_all_explain_llama.py --model meta-llama/Llama-3.1-8B-Instruct --train_batch 32
"""

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generator_script",
        default="modules/explain_module/generator_explain_llama.py",
        help="Path to generator_llama.py (相對或絕對路徑).",
    )
    ap.add_argument(
        "--conversations",
        default="data/preprocess/conversations.jsonl",
        help="Path to conversations.jsonl",
    )
    ap.add_argument(
        "--data_dir",
        default="data/preprocess",
        help="Directory containing pairs_train/valid/test.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs",
        help="Directory to save explain_* outputs",
    )
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name for Llama",
    )
    ap.add_argument(
        "--train_batch",
        type=int,
        default=8,
        help="Batch size for train split",
    )
    ap.add_argument(
        "--valid_batch",
        type=int,
        default=8,
        help="Batch size for valid split",
    )
    ap.add_argument(
        "--test_batch",
        type=int,
        default=8,
        help="Batch size for test split",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="max_new_tokens passed to generator_llama.py",
    )
    ap.add_argument(
        "--max_input_len",
        type=int,
        default=256,
        help="max_input_len passed to generator_llama.py",
    )
    args = ap.parse_args()

    generator_script = Path(args.generator_script)
    if not generator_script.exists():
        raise FileNotFoundError(f"generator_script not found: {generator_script}")

    conversations = str(Path(args.conversations))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 三個 split 的設定
    splits = [
        {
            "name": "train",
            "pairs": data_dir / "pairs_train.jsonl",
            "out_jsonl": out_dir / "explain_train.jsonl",
            "out_text": out_dir / "explain_train.txt",
            "batch": args.train_batch,
        },
        {
            "name": "valid",
            "pairs": data_dir / "pairs_valid.jsonl",
            "out_jsonl": out_dir / "explain_valid.jsonl",
            "out_text": out_dir / "explain_valid.txt",
            "batch": args.valid_batch,
        },
        {
            "name": "test",
            "pairs": data_dir / "pairs_test.jsonl",
            "out_jsonl": out_dir / "explain_test.jsonl",
            "out_text": out_dir / "explain_test.txt",
            "batch": args.test_batch,
        },
    ]

    for cfg in splits:
        pairs_path = cfg["pairs"]
        if not pairs_path.exists():
            print(f"[Skip] {cfg['name']} pairs file not found: {pairs_path}")
            continue

        print(f"\n=== Running {cfg['name']} split ===")
        print(f"pairs_in   : {pairs_path}")
        print(f"out_jsonl  : {cfg['out_jsonl']}")
        print(f"out_text   : {cfg['out_text']}")
        print(f"batch      : {cfg['batch']}")

        cmd = [
            "python",
            str(generator_script),
            "--conversations",
            conversations,
            "--pairs_in",
            str(pairs_path),
            "--out_jsonl",
            str(cfg["out_jsonl"]),
            "--out_text",
            str(cfg["out_text"]),
            "--model",
            args.model,
            "--batch",
            str(cfg["batch"]),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--max_input_len",
            str(args.max_input_len),
        ]

        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
