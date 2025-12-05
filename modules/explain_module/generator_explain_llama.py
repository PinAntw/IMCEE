#!/usr/bin/env python3
#modules/explain_module/generator_llama.py 
#python3 modules/explain_module/generator_llama.py --conversations /home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl --pairs_in /home/joung/r13725060/Research/IMCEE/data/preprocess/pairs_test.jsonl --out_jsonl outputs/explain_test_results.jsonl --out_text outputs/explain_test_results.txt --batch 16 --model "meta-llama/Llama-3.1-8B-Instruct"
#python3 modules/explain_module/generator_llama.py --conversations /home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl --pairs_in /home/joung/r13725060/Research/IMCEE/data/preprocess/pairs_valid.jsonl --out_jsonl outputs/explain_valid_results.jsonl --out_text outputs/explain_valid_results.txt --batch 16 --model "meta-llama/Llama-3.1-8B-Instruct"
#python3 modules/explain_module/generator_llama.py --conversations /home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl --pairs_in /home/joung/r13725060/Research/IMCEE/data/preprocess/pairs_train.jsonl --out_jsonl outputs/explain_train_results.jsonl --out_text outputs/explain_train_results.txt --batch 32 --model "meta-llama/Llama-3.1-8B-Instruct" 
# -*- coding: utf-8 -*-
import os, json, argparse, re
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# [刪除舊的 PROMPT_POS]

# [加入這兩個新的]
SYSTEM_PROMPT = (
    "You are an expert in causal reasoning. Your task is to write ONE concise, neutral sentence "
    "describing the plausible causal mechanism (e.g., blame, threat, empathy, apology) "
    "of how the CANDIDATE utterance influences the TARGET utterance.\n"
    "RULES:\n"
    "1. Be neutral. Do NOT judge truth or labels.\n"
    "2. Do NOT correct spelling. Focus ONLY on the mechanism.\n"
    "3. If no strong mechanism is plausible, describe the neutral interaction (e.g., 'The candidate asks a follow-up question').\n"
    "4. NEVER output an empty sentence.\n"
    "5. ONLY output the single explanation sentence. Do NOT add headers or repeat the input."
)

USER_TEMPLATE = (
    "CANDIDATE: {cand}\n"
    "TARGET: {targ}"
)
def load_conversations(path):
    convs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # 建立一個巢狀字典： conv_id -> utt_id -> text
            utt_map = {utt["utt_id"]: utt["text"] for utt in obj["utterances"]}
            convs[obj["conv_id"]] = utt_map
    return convs

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append(obj)  # need: id, conv_id, c_utt_id, t_utt_id, label
    return pairs

def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf:
        yield buf

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # 可選：去除模型常見前綴 (雖然 apply_chat_template 後不太需要了)
    s = re.sub(r"^(Explanation\s*:\s*)", "", s, flags=re.IGNORECASE)
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--out_jsonl", required=True)          # e.g., outputs/explanations.jsonl
    ap.add_argument("--out_text", default=None)            # e.g., outputs/explain_text.txt
    ap.add_argument("--out_index", default=None)           # e.g., outputs/explain_index.tsv
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)        # 4B 建議小一點 batch
    ap.add_argument("--max_input_len", type=int, default=384)
    ap.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    args = ap.parse_args()

    # dtype 設定
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Qwen 是 causal LM；使用 AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

    convs = load_conversations(args.conversations)
    pairs = load_pairs(args.pairs_in)

    # 處理所有 pairs
    pos_pairs = pairs

    # --- [斷點續傳邏輯 START] ---
    out_path = Path(args.out_jsonl); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text_path = Path(args.out_text) if args.out_text else out_path.with_suffix(".txt")
    out_index_path = Path(args.out_index) if args.out_index else out_path.with_name(out_path.stem + "_index.tsv")

    existing_ids = set()
    file_mode = "w" # 預設為 write (覆蓋)
    
    if out_path.exists():
        print(f"Resuming from existing file: {out_path}")
        file_mode = "a" # 改為 append (附加)
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass # 跳過損壞的行
    
    print(f"Loaded {len(existing_ids)} existing explanation IDs.")

    line_no = 0
    if file_mode == "a" and out_index_path.exists():
        try:
            with open(out_index_path, "r", encoding="utf-8") as f:
                line_no = sum(1 for _ in f)
            print(f"Starting index file from line {line_no}")
        except FileNotFoundError:
            line_no = len(existing_ids) # 嘗試同步

    print("Filtering already processed pairs...")
    pairs_to_process = []
    for p in pairs:
        pid = f'{p["conv_id"]}__{p["c_utt_id"]}__{p["t_utt_id"]}'
        if pid not in existing_ids:
            pairs_to_process.append(p)

    print(f"Total pairs: {len(pairs)}. Pairs to process: {len(pairs_to_process)}")
    if not pairs_to_process:
        print("All pairs already processed. Exiting.")
        return
    
    fw_jsonl = open(out_path, file_mode, encoding="utf-8")
    fw_text = open(out_text_path, file_mode, encoding="utf-8")
    fw_index = open(out_index_path, file_mode, encoding="utf-8")
    # --- [斷點續傳邏輯 END] ---
    

    total_batches = (len(pairs_to_process) + args.batch - 1) // args.batch

    with torch.no_grad():
        for batch in tqdm(batched(pairs_to_process, args.batch), total=total_batches):
            
            # 1. 建立 message 列表
            batch_messages = []
            pids = []
            for p in batch:
                cand = convs[p["conv_id"]][p["c_utt_id"]]
                targ = convs[p["conv_id"]][p["t_utt_id"]]
                user_content = USER_TEMPLATE.format(cand=cand, targ=targ)
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                batch_messages.append(messages)
                
                pid = f'{p["conv_id"]}__{p["c_utt_id"]}__{p["t_utt_id"]}'
                pids.append(pid)

            # 2. 使用 apply_chat_template 進行 Tokenize
            tok.padding_side = "left" # 對 Causal LM 生成很重要
            enc = tok.apply_chat_template(
                batch_messages,
                add_generation_prompt=True, # 告訴模型準備回答
                padding=True,
                truncation=True,
                max_length=args.max_input_len,
                return_tensors="pt"
            )

            # --- [AttributeError 修正 START] ---
            # 3. 手動建立 attention_mask 並傳送
            input_ids = enc
            attention_mask = (input_ids != tok.pad_token_id).long()

            if device == "cuda":
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
            
            enc_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            # --- [AttributeError 修正 END] ---

            # 4. 生成 (Generate)
            input_token_len = enc_dict["input_ids"].shape[1]

            gen_out = model.generate(
                **enc_dict, # <-- 使用修正後的字典
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id
            )

            # 5. 解碼 (Decode) - 僅解碼新產生的部分
            new_tokens = gen_out[:, input_token_len:]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

            results = [normalize_text(ex) for ex in texts] # normalize_text 仍然有用

            # 6. 寫入 (Write)
            for pid, ex in zip(pids, results):
                if not ex: # 如果模型還是產生空值
                    ex = "No specific causal mechanism identified." # 給一個預設值

                fw_jsonl.write(json.dumps({"id": pid, "explain": ex}, ensure_ascii=False) + "\n")
                fw_text.write(ex + "\n")
                fw_index.write(f"{line_no}\t{pid}\n")
                line_no += 1

    fw_jsonl.close()
    fw_text.close()
    fw_index.close()

# --- [IndentationError 修正] ---
# 取消以下兩行的縮排
if __name__ == "__main__":
    main()