#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計對話中 utterance 數量與 token 數量的基本資訊。

使用方式：
    python3 utils/stats_utterances_tokens.py \
        --conversations /home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl \
        --model_name meta-llama/Llama-3.1-8B-Instruct
"""

import json
import argparse
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(
        description="統計 conversations.jsonl 中對話的 utterance / token 統計量"
    )
    ap.add_argument(
        "--conversations",
        type=str,
        required=True,
        help="conversations.jsonl 路徑",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="用來計算 token 數量的 HF tokenizer 名稱",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    conv_path = Path(args.conversations)

    if not conv_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {conv_path}")

    print(f"[Info] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 統計量累積器
    num_convs = 0

    # utterance 數量
    total_utterances_per_conv = 0
    max_utterances_in_conv = 0

    # 每個 utterance 的 token 數
    total_tokens_per_utt = 0
    max_tokens_in_utt = 0
    total_num_utts = 0

    # 每個對話「總 token 數」
    total_tokens_per_conv = 0
    max_tokens_in_conv = 0

    print(f"[Info] Reading conversations from: {conv_path}")
    with conv_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing conversations"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            utterances = obj.get("utterances", [])
            num_convs += 1

            # --- utterance 數量 ---
            num_utts = len(utterances)
            total_utterances_per_conv += num_utts
            if num_utts > max_utterances_in_conv:
                max_utterances_in_conv = num_utts

            # --- token 統計 ---
            conv_token_sum = 0

            for utt in utterances:
                text = utt.get("text", "")
                # 只計算非空白
                if text is None:
                    text = ""

                enc = tokenizer(
                    text,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_tensors=None,
                )
                # enc["input_ids"] 是一個 list[int]
                tok_len = len(enc["input_ids"])

                total_num_utts += 1
                total_tokens_per_utt += tok_len
                conv_token_sum += tok_len

                if tok_len > max_tokens_in_utt:
                    max_tokens_in_utt = tok_len

            # 對話總 token
            total_tokens_per_conv += conv_token_sum
            if conv_token_sum > max_tokens_in_conv:
                max_tokens_in_conv = conv_token_sum

    # ====== 統計結果 ======
    print("\n========== 統計結果 ==========")
    print(f"對話數量 (num_convs): {num_convs}")

    if num_convs > 0:
        avg_utts_per_conv = total_utterances_per_conv / num_convs
        avg_tokens_per_conv = total_tokens_per_conv / num_convs
    else:
        avg_utts_per_conv = 0.0
        avg_tokens_per_conv = 0.0

    if total_num_utts > 0:
        avg_tokens_per_utt = total_tokens_per_utt / total_num_utts
    else:
        avg_tokens_per_utt = 0.0

    print(f"\n[Utterance 數量]")
    print(f"  每個對話的 utterance 平均數量: {avg_utts_per_conv:.2f}")
    print(f"  單一對話最大 utterance 數量: {max_utterances_in_conv}")

    print(f"\n[Token 數量（以 {args.model_name} 的 tokenizer 為準）]")
    print(f"  每個 utterance 平均 token 數: {avg_tokens_per_utt:.2f}")
    print(f"  單一句子最大 token 數: {max_tokens_in_utt}")

    print(f"\n[對話層級的 token 數]")
    print(f"  每個對話總 token 平均數: {avg_tokens_per_conv:.2f}")
    print(f"  單一對話最大總 token 數: {max_tokens_in_conv}")
    print("================================")


if __name__ == "__main__":
    main()
