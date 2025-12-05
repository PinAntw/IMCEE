#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IMCEE/modules/explain_module/generator_llama.py
#
# Example runs:
# python3 modules/explain_module/generator_explain_llama.py  --conversations data/preprocess/conversations.jsonl  --pairs_in data/preprocess/pairs_sample.jsonl  --out_jsonl outputs/explain_sample.jsonl  --out_text outputs/explain_sample.txt  --batch 16  --model "meta-llama/Llama-3.1-8B-Instruct"

import re
import json
import argparse
from pathlib import Path

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = (
    "You are an expert in causal emotion reasoning.\n"
    "\n"
    "You will be given a dialogue, a candidate utterance Ui, and a target emotional utterance Uj.\n"
    "Your task is to determine whether the content of Ui is likely to cause, trigger, or meaningfully contribute to the emotion expressed in Uj, based on the dialogue context.\n"
    "\n"
    "Think briefly:\n"
    "1) Infer what event, meaning, or situation in the dialogue explains the emotion in Uj.\n"
    "2) Evaluate whether Ui would lead a reasonable person to feel the emotion in Uj (not just match the tone or atmosphere).\n"
    "3) Output a causal influence score between 0 and 1.\n"
    "   - High (0.7–1.0): Ui clearly explains the emotion.\n"
    "   - Medium (0.4–0.6): Ui partially contributes but is not the main cause.\n"
    "   - Low (0.0–0.3): Ui is unrelated, weakly related, or the emotion is caused by something else.\n"
    "\n"
    "Ui and Uj may be the same utterance; if the content naturally evokes that emotion by itself, this counts as a valid self-cause.\n"
    "\n"
    "Output ONLY valid JSON:\n"
    "{\n"
    "  \"score\": <0–1 number>,\n"
    "  \"label\": \"cause\" or \"not_cause\",\n"
    "  \"reason\": \"<ONE short sentence explaining your reasoning>\"\n"
    "}\n"
)


# 整段對話 + Ui/Uj + Uj emotion
USER_TEMPLATE = (
    "DIALOGUE:\n"
    "{dialog}\n\n"
    "Candidate cause utterance (Ui = {c_id}): {cand}\n"
    "Target emotional utterance (Uj = {t_id}, emotion = {t_emotion}): {targ}\n"
)


def load_conversations(path):
    """
    conversations.jsonl → {conv_id: {utt_id: utterance_dict}}
    utterance_dict 至少含有: text, emotion
    """
    convs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            utt_map = {utt["utt_id"]: utt for utt in obj["utterances"]}
            convs[obj["conv_id"]] = utt_map
    return convs


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append(obj)
    return pairs


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_score_label_reason(raw: str):
    """
    不假設 raw 是合法 JSON，只用 regex / 字串把 score / label / reason 撈出來。
    raw 可能像：
      { "score": 0.8, "label": "cause", "reason": "...." }
    也可能被截斷、缺 }、缺最後的 "。
    """
    score = 0.0
    label = "not_cause"
    reason = ""

    # 1. score
    m_score = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if m_score:
        try:
            score = float(m_score.group(1))
        except Exception:
            score = 0.0

    # 2. label
    m_label = re.search(r'"label"\s*:\s*"(cause|not_cause)"', raw)
    if m_label:
        label = m_label.group(1)

    # 3. reason: 從 "reason": 開始，取後面整段，去掉開頭的引號，最後砍到最後一個雙引號（如果有）
    idx = raw.find('"reason"')
    if idx != -1:
        after = raw[idx:]
        colon_idx = after.find(":")
        if colon_idx != -1:
            reason_part = after[colon_idx + 1 :].strip()

            # 去掉開頭的 { 或 , 之類
            while reason_part and reason_part[0] in ["{", ",", " "]:
                reason_part = reason_part[1:]
                reason_part = reason_part.lstrip()

            # 去掉開頭的引號
            if reason_part.startswith('"'):
                reason_part = reason_part[1:]

            # 如果有對應的結尾引號，就砍到最後一個 "
            last_quote = reason_part.rfind('"')
            if last_quote != -1:
                reason_text = reason_part[:last_quote]
            else:
                reason_text = reason_part

            reason = normalize_text(reason_text)

    return score, label, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_text", default=None)
    ap.add_argument("--out_index", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max_input_len", type=int, default=512)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()

    # dtype
    if args.dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()

    convs = load_conversations(args.conversations)
    pairs = load_pairs(args.pairs_in)

    # 斷點續傳
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text_path = Path(args.out_text) if args.out_text else out_path.with_suffix(".txt")
    out_index_path = Path(args.out_index) if args.out_index else out_path.with_name(out_path.stem + "_index.tsv")

    existing_ids = set()
    file_mode = "w"

    if out_path.exists():
        print(f"Resuming from existing file: {out_path}")
        file_mode = "a"
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(existing_ids)} existing explanation IDs.")

    line_no = 0
    if file_mode == "a" and out_index_path.exists():
        try:
            with open(out_index_path, "r", encoding="utf-8") as f:
                line_no = sum(1 for _ in f)
            print(f"Starting index file from line {line_no}")
        except FileNotFoundError:
            line_no = len(existing_ids)

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

    total_batches = (len(pairs_to_process) + args.batch - 1) // args.batch

    with torch.no_grad():
        for batch in tqdm(batched(pairs_to_process, args.batch), total=total_batches):

            batch_messages = []
            pids = []

            for p in batch:
                conv_id = p["conv_id"]
                c_id = p["c_utt_id"]
                t_id = p["t_utt_id"]

                utt_map = convs[conv_id]

                ordered_utts = sorted(
                    utt_map.items(),
                    key=lambda x: int(re.sub(r"\D", "", x[0]) or 0),
                )
                dialog_lines = []
                for uid, utt in ordered_utts:
                    text = utt.get("text", "")
                    emo = utt.get("emotion", None)
                    if emo is not None:
                        dialog_lines.append(f"{uid} [{emo}]: {text}")
                    else:
                        dialog_lines.append(f"{uid}: {text}")
                dialog_str = "\n".join(dialog_lines)

                cand_text = utt_map[c_id].get("text", "")
                targ_text = utt_map[t_id].get("text", "")
                t_emotion = utt_map[t_id].get("emotion", "unknown")

                user_content = USER_TEMPLATE.format(
                    dialog=dialog_str,
                    c_id=c_id,
                    t_id=t_id,
                    cand=cand_text,
                    targ=targ_text,
                    t_emotion=t_emotion,
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                batch_messages.append(messages)

                pid = f"{conv_id}__{c_id}__{t_id}"
                pids.append(pid)

            tok.padding_side = "left"
            enc = tok.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=args.max_input_len,
                return_tensors="pt",
            )

            input_ids = enc
            attention_mask = (input_ids != tok.pad_token_id).long()

            if device == "cuda":
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

            enc_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

            input_token_len = enc_dict["input_ids"].shape[1]

            gen_out = model.generate(
                **enc_dict,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

            new_tokens = gen_out[:, input_token_len:]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

            # 這裡不再用 json.loads，而是 parse_score_label_reason
            for pid, raw in zip(pids, texts):
                raw = raw.strip()

                score, label, reason = parse_score_label_reason(raw)

                if not reason:
                    reason = "No specific causal mechanism identified."

                # clamp score 在 [0,1]
                try:
                    score = max(0.0, min(1.0, float(score)))
                except Exception:
                    score = 0.0

                fw_jsonl.write(
                    json.dumps(
                        {"id": pid, "score": score, "label": label, "reason": reason},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fw_text.write(reason + "\n")
                fw_index.write(f"{line_no}\t{pid}\n")
                line_no += 1

    fw_jsonl.close()
    fw_text.close()
    fw_index.close()


if __name__ == "__main__":
    main()
