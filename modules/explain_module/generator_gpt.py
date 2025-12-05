# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # IMCEE/modules/explain_module/generator_gpt.py
# #
# # Example:
# # python3 modules/explain_module/generator_gpt.py --conversations data/preprocess/conversations.jsonl --pairs_in data/preprocess/pairs_valid.jsonl --out_jsonl outputs/explain_valid_gpt4omini.jsonl --out_text outputs/explain_valid_gpt4omini.txt

# import re
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict

# from tqdm import tqdm

# import openai  # 需要先安裝: pip install openai

# # 你可以改成 gpt-4.1, gpt-4.1-mini, gpt-4o 等
# DEFAULT_MODEL_NAME = "gpt-4o-mini"

# SYSTEM_PROMPT = (
#     "You are an expert in causal emotion reasoning.\n"
#     "\n"
#     "You will be given a dialogue, a candidate utterance Ui, and a target emotional utterance Uj.\n"
#     "Your task is to determine whether the content of Ui is likely to cause, trigger, or meaningfully contribute to the emotion expressed in Uj, based on the dialogue context.\n"
#     "\n"
#     "Think briefly:\n"
#     "1) Infer what event, meaning, or situation in the dialogue explains the emotion in Uj.\n"
#     "2) Evaluate whether Ui would lead a reasonable person to feel the emotion in Uj (not just match the tone or atmosphere).\n"
#     "3) Output a causal influence score between 0 and 1.\n"
#     "   - High (0.7–1.0): Ui clearly explains the emotion.\n"
#     "   - Medium (0.4–0.6): Ui partially contributes but is not the main cause.\n"
#     "   - Low (0.0–0.3): Ui is unrelated, weakly related, or the emotion is caused by something else.\n"
#     "\n"
#     "Ui and Uj may be the same utterance; if the content naturally evokes that emotion by itself, this counts as a valid self-cause.\n"
#     "\n"
#     "Output ONLY valid JSON:\n"
#     "{\n"
#     "  \"score\": <0–1 number>,\n"
#     "  \"label\": \"cause\" or \"not_cause\",\n"
#     "  \"reason\": \"<ONE short sentence explaining your reasoning>\"\n"
#     "}\n"
# )

# USER_TEMPLATE = (
#     "DIALOGUE:\n"
#     "{dialog}\n\n"
#     "Candidate cause utterance (Ui = {c_id}): {cand}\n"
#     "Target emotional utterance (Uj = {t_id}, emotion = {t_emotion}): {targ}\n"
# )


# def load_conversations(path: str) -> Dict[str, Dict[str, dict]]:
#     """
#     conversations.jsonl → {conv_id: {utt_id: utterance_dict}}
#     utterance_dict 至少含有: text, emotion
#     """
#     convs = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             utt_map = {utt["utt_id"]: utt for utt in obj["utterances"]}
#             convs[obj["conv_id"]] = utt_map
#     return convs


# def load_pairs(path: str) -> List[dict]:
#     pairs = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             pairs.append(obj)
#     return pairs


# def batched(it, n):
#     buf = []
#     for x in it:
#         buf.append(x)
#         if len(buf) == n:
#             yield buf
#             buf = []
#     if buf:
#         yield buf


# def normalize_text(s: str) -> str:
#     s = s.strip()
#     s = re.sub(r"\s+", " ", s)
#     return s.strip()


# def parse_score_label_reason(raw: str):
#     """
#     raw 是 GPT 回傳的文字（理論上是 JSON，但保險起見仍用 regex 抽 score/label/reason）
#     """
#     score = 0.0
#     label = "not_cause"
#     reason = ""

#     # 1. score
#     m_score = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
#     if m_score:
#         try:
#             score = float(m_score.group(1))
#         except Exception:
#             score = 0.0

#     # 2. label
#     m_label = re.search(r'"label"\s*:\s*"(cause|not_cause)"', raw)
#     if m_label:
#         label = m_label.group(1)

#     # 3. reason
#     idx = raw.find('"reason"')
#     if idx != -1:
#         after = raw[idx:]
#         colon_idx = after.find(":")
#         if colon_idx != -1:
#             reason_part = after[colon_idx + 1 :].strip()

#             while reason_part and reason_part[0] in ["{", ",", " "]:
#                 reason_part = reason_part[1:]
#                 reason_part = reason_part.lstrip()

#             if reason_part.startswith('"'):
#                 reason_part = reason_part[1:]

#             last_quote = reason_part.rfind('"')
#             if last_quote != -1:
#                 reason_text = reason_part[:last_quote]
#             else:
#                 reason_text = reason_part

#             reason = normalize_text(reason_text)

#     return score, label, reason


# def call_gpt_api(
#     model_name: str,
#     system_prompt: str,
#     user_content: str,
#     temperature: float = 0.0,
# ) -> str:
#     """
#     呼叫 OpenAI Chat Completions，回傳 assistant 的文字（string）。
#     """
#     client = openai.OpenAI()  # 使用 OPENAI_API_KEY 環境變數

#     resp = client.chat.completions.create(
#         model=model_name,
#         temperature=temperature,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_content},
#         ],
#     )
#     return resp.choices[0].message.content


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--conversations", required=True)
#     ap.add_argument("--pairs_in", required=True)
#     ap.add_argument("--out_jsonl", required=True)
#     ap.add_argument("--out_text", default=None)
#     ap.add_argument("--out_index", default=None)
#     ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="OpenAI GPT model name")
#     ap.add_argument(
#         "--batch",
#         type=int,
#         default=1,
#         help="Logical batch for迭代顯示用，實際 GPT API 是逐條呼叫。",
#     )
#     args = ap.parse_args()

#     convs = load_conversations(args.conversations)
#     pairs = load_pairs(args.pairs_in)

#     out_path = Path(args.out_jsonl)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_text_path = Path(args.out_text) if args.out_text else out_path.with_suffix(".txt")
#     out_index_path = Path(args.out_index) if args.out_index else out_path.with_name(
#         out_path.stem + "_index.tsv"
#     )

#     existing_ids = set()
#     file_mode = "w"
#     if out_path.exists():
#         print(f"Resuming from existing file: {out_path}")
#         file_mode = "a"
#         with open(out_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     existing_ids.add(json.loads(line)["id"])
#                 except json.JSONDecodeError:
#                     pass

#     print(f"Loaded {len(existing_ids)} existing explanation IDs.")

#     line_no = 0
#     if file_mode == "a" and out_index_path.exists():
#         try:
#             with open(out_index_path, "r", encoding="utf-8") as f:
#                 line_no = sum(1 for _ in f)
#             print(f"Starting index file from line {line_no}")
#         except FileNotFoundError:
#             line_no = len(existing_ids)

#     print("Filtering already processed pairs...")
#     pairs_to_process = []
#     for p in pairs:
#         pid = f'{p["conv_id"]}__{p["c_utt_id"]}__{p["t_utt_id"]}'
#         if pid not in existing_ids:
#             pairs_to_process.append(p)

#     print(f"Total pairs: {len(pairs)}. Pairs to process: {len(pairs_to_process)}")
#     if not pairs_to_process:
#         print("All pairs already processed. Exiting.")
#         return

#     fw_jsonl = open(out_path, file_mode, encoding="utf-8")
#     fw_text = open(out_text_path, file_mode, encoding="utf-8")
#     fw_index = open(out_index_path, file_mode, encoding="utf-8")

#     total_batches = (len(pairs_to_process) + args.batch - 1) // args.batch

#     with tqdm(total=len(pairs_to_process)) as pbar:
#         for batch in batched(pairs_to_process, args.batch):
#             for p in batch:
#                 conv_id = p["conv_id"]
#                 c_id = p["c_utt_id"]
#                 t_id = p["t_utt_id"]

#                 utt_map = convs[conv_id]

#                 ordered_utts = sorted(
#                     utt_map.items(),
#                     key=lambda x: int(re.sub(r"\D", "", x[0]) or 0),
#                 )
#                 dialog_lines = []
#                 for uid, utt in ordered_utts:
#                     text = utt.get("text", "")
#                     emo = utt.get("emotion", None)
#                     if emo is not None:
#                         dialog_lines.append(f"{uid} [{emo}]: {text}")
#                     else:
#                         dialog_lines.append(f"{uid}: {text}")
#                 dialog_str = "\n".join(dialog_lines)

#                 cand_text = utt_map[c_id].get("text", "")
#                 targ_text = utt_map[t_id].get("text", "")
#                 t_emotion = utt_map[t_id].get("emotion", "unknown")

#                 user_content = USER_TEMPLATE.format(
#                     dialog=dialog_str,
#                     c_id=c_id,
#                     t_id=t_id,
#                     cand=cand_text,
#                     targ=targ_text,
#                     t_emotion=t_emotion,
#                 )

#                 try:
#                     raw = call_gpt_api(
#                         model_name=args.model,
#                         system_prompt=SYSTEM_PROMPT,
#                         user_content=user_content,
#                         temperature=0.0,
#                     )
#                 except Exception as e:
#                     print(f"[Error] GPT API failed for {conv_id}__{c_id}__{t_id}: {e}")
#                     raw = ""

#                 raw = (raw or "").strip()
#                 score, label, reason = parse_score_label_reason(raw)

#                 if not reason:
#                     reason = "No specific causal mechanism identified."

#                 try:
#                     score = max(0.0, min(1.0, float(score)))
#                 except Exception:
#                     score = 0.0

#                 pid = f"{conv_id}__{c_id}__{t_id}"

#                 fw_jsonl.write(
#                     json.dumps(
#                         {"id": pid, "score": score, "label": label, "reason": reason},
#                         ensure_ascii=False,
#                     )
#                     + "\n"
#                 )
#                 fw_text.write(reason + "\n")
#                 fw_index.write(f"{line_no}\t{pid}\n")
#                 line_no += 1

#                 pbar.update(1)

#     fw_jsonl.close()
#     fw_text.close()
#     fw_index.close()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
import asyncio
import httpx  # 新增: 用於解除連線限制
from pathlib import Path
from typing import List, Dict

from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_CONCURRENCY = 20  # 預設值 (執行時請用 --concurrency 參數覆蓋)

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

USER_TEMPLATE = (
    "DIALOGUE:\n"
    "{dialog}\n\n"
    "Candidate cause utterance (Ui = {c_id}): {cand}\n"
    "Target emotional utterance (Uj = {t_id}, emotion = {t_emotion}): {targ}\n"
)


def load_conversations(path: str) -> Dict[str, Dict[str, dict]]:
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


def load_pairs(path: str) -> List[dict]:
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
    raw 是 GPT 回傳的文字（理論上是 JSON，但保險起見仍用 regex 抽 score/label/reason）
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

    # 3. reason
    idx = raw.find('"reason"')
    if idx != -1:
        after = raw[idx:]
        colon_idx = after.find(":")
        if colon_idx != -1:
            reason_part = after[colon_idx + 1 :].strip()

            while reason_part and reason_part[0] in ["{", ",", " "]:
                reason_part = reason_part[1:]
                reason_part = reason_part.lstrip()

            if reason_part.startswith('"'):
                reason_part = reason_part[1:]

            last_quote = reason_part.rfind('"')
            if last_quote != -1:
                reason_text = reason_part[:last_quote]
            else:
                reason_text = reason_part

            reason = normalize_text(reason_text)

    return score, label, reason

async def call_gpt_api_async(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_content: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
) -> str:
    """
    非同步呼叫 OpenAI API，並使用 Semaphore 控制併發量
    """
    async with sem:  # 限制同時只有 N 個請求在執行
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            # 簡單的錯誤處理，實際使用建議加上重試機制 (如 tenancy)
            print(f"\n[Error] API failed: {e}")
            return ""

async def process_single_pair(
    client, sem, p, convs, model_name
):
    """處理單一 Pair 的邏輯封裝"""
    conv_id = p["conv_id"]
    c_id = p["c_utt_id"]
    t_id = p["t_utt_id"]
    
    # 這裡複製你原本的組裝邏輯
    utt_map = convs[conv_id]
    ordered_utts = sorted(utt_map.items(), key=lambda x: int(re.sub(r"\D", "", x[0]) or 0))
    
    dialog_lines = []
    for uid, utt in ordered_utts:
        text = utt.get("text", "")
        emo = utt.get("emotion", None)
        dialog_lines.append(f"{uid} [{emo}]: {text}" if emo else f"{uid}: {text}")
    dialog_str = "\n".join(dialog_lines)

    cand_text = utt_map[c_id].get("text", "")
    targ_text = utt_map[t_id].get("text", "")
    t_emotion = utt_map[t_id].get("emotion", "unknown")

    user_content = USER_TEMPLATE.format(
        dialog=dialog_str, c_id=c_id, t_id=t_id, cand=cand_text, targ=targ_text, t_emotion=t_emotion
    )

    # 非同步呼叫
    raw = await call_gpt_api_async(client, model_name, SYSTEM_PROMPT, user_content, sem)
    
    # 解析結果
    score, label, reason = parse_score_label_reason(raw or "")
    if not reason:
        reason = "No specific causal mechanism identified."
    try:
        score = max(0.0, min(1.0, float(score)))
    except:
        score = 0.0

    pid = f"{conv_id}__{c_id}__{t_id}"
    result_obj = {"id": pid, "score": score, "label": label, "reason": reason}
    
    return result_obj, reason, pid

async def main_async():
    ap = argparse.ArgumentParser()
    # ... (你的參數設定保持不變) ...
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--pairs_in", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_text", default=None)
    ap.add_argument("--out_index", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--concurrency", type=int, default=20, help="Concurrent requests")
    args = ap.parse_args()

    # 載入資料
    convs = load_conversations(args.conversations)
    pairs = load_pairs(args.pairs_in)
    
    # 設定路徑與讀取已存在 ID (略過重複)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text_path = Path(args.out_text) if args.out_text else out_path.with_suffix(".txt")
    out_index_path = Path(args.out_index) if args.out_index else out_path.with_name(out_path.stem + "_index.tsv")

    existing_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try: existing_ids.add(json.loads(line)["id"])
                except: pass

    pairs_to_process = [
        p for p in pairs 
        if f'{p["conv_id"]}__{p["c_utt_id"]}__{p["t_utt_id"]}' not in existing_ids
    ]
    print(f"Total: {len(pairs)}, To Process: {len(pairs_to_process)}")

    if not pairs_to_process:
        return

    # --- 針對 Tier 5 優化的 Client 設定 ---
    # 增加 httpx 的連線池限制 (預設為 100，改為 1000 以支援高併發)
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=1000)
    async_http_client = httpx.AsyncClient(limits=limits, timeout=60.0)
    
    client = AsyncOpenAI(http_client=async_http_client)
    sem = asyncio.Semaphore(args.concurrency) # 控制併發數

    # 建立 Tasks
    tasks = [
        process_single_pair(client, sem, p, convs, args.model) 
        for p in pairs_to_process
    ]

    # 開啟檔案準備寫入
    mode = "a" if existing_ids else "w"
    
    # 計算起始行號
    line_no = len(existing_ids)
    
    try:
        with open(out_path, mode, encoding="utf-8") as fw_jsonl, \
             open(out_text_path, mode, encoding="utf-8") as fw_text, \
             open(out_index_path, mode, encoding="utf-8") as fw_index:

            # 使用 tqdm 顯示進度，並依完成順序寫入
            for coro in tqdm_asyncio.as_completed(tasks):
                result_obj, reason_text, pid = await coro
                
                # 寫入結果
                fw_jsonl.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                fw_text.write(reason_text + "\n")
                fw_index.write(f"{line_no}\t{pid}\n")
                
                # 強制刷新緩衝區，避免程式崩潰時資料遺失
                fw_jsonl.flush()
                fw_text.flush()
                fw_index.flush()
                
                line_no += 1
    finally:
        await client.close()  # 確保關閉連線

if __name__ == "__main__":
    asyncio.run(main_async())