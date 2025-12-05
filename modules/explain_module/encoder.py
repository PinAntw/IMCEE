#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-to-LM Interpreter (TAPE 概念 [cite: 1417, 1436-1437, 1453])

此腳本負責將 `generator.py` 產生的「解釋文字」(.txt)
透過預訓練的 SentenceTransformer (SBERT) 轉換為
GNN 可以讀取的「特徵向量」(.pt)。

這是一個離線 (offline)、一次性的步驟。
"""

import argparse
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser(description="Encode explanation texts into embeddings.")
    ap.add_argument("--input_text", required=True, 
                    help="Path to the .txt file (e.g., outputs/explain_train_results.txt)")
    ap.add_argument("--output_pt", required=True, 
                    help="Path to save the output .pt file (e.g., outputs/explain_train_embeddings.pt)")
    ap.add_argument("--model", 
                    default="roberta-large", 
                    help="SentenceTransformer model to use for encoding.")
    ap.add_argument("--batch_size", type=int, default=128, 
                    help="Batch size for SBERT encoding.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 載入預訓練的 SBERT 模型
    print(f"Loading model: {args.model}")
    # 這是 TAPE 論文中的 "LM Interpreter"
    model = SentenceTransformer(args.model, device=device)

    # 2. 準備路徑並讀取句子
    in_path = Path(args.input_text)
    out_path = Path(args.output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: Input file not found at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading sentences from: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        # 讀取所有行，並去除前後空白
        # 過濾掉空行，以防萬一
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Found {len(sentences)} sentences to encode.")

    # 3. 執行編碼 (Encode)
    # SBERT 的 encode 速度很快，batch size 可以設高一點
    print("Encoding sentences...")
    embeddings = model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,  # 顯示進度條
        convert_to_tensor=True,  # 直接轉換為 PyTorch Tensor
        device=device
    )

    # 4. 儲存結果
    # 儲存前移至 CPU 是標準作法
    embeddings_cpu = embeddings.cpu()
    torch.save(embeddings_cpu, out_path)

    print(f"\nSuccessfully saved {embeddings.shape} embeddings to: {out_path}")

if __name__ == "__main__":
    main()