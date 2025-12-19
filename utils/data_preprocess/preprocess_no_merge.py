import json
import pandas as pd
import re
import os

# ================= 設定區 =================

# 1. Conversations 檔案路徑
CONVERSATIONS_PATH = '/home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl'

# 2. 設定 CSV 輸入與輸出目錄
BASE_CSV_DIR = '/home/joung/r13725060/Research/RECCON/data/subtask2/fold1'
BASE_OUT_DIR = '/home/joung/r13725060/Research/IMCEE/data/preprocess'

# 為了避免覆蓋原本已經處理好的合併版檔案，建議輸出檔名加上後綴 (e.g., _spans)
TASKS = [
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_train_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'pairs_train_spans.jsonl'),
        "split_name": "train"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_valid_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'pairs_valid_spans.jsonl'),
        "split_name": "valid"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_test_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'pairs_test_spans.jsonl'),
        "split_name": "test"
    }
]

# Regex 解析 ID
id_pattern = re.compile(r'dailydialog_(?P<split>\w+)_(?P<conv_num>\d+)_utt_(?P<t_idx>\d+)_(?:.*)_cause_utt_(?P<c_idx>\d+)(?:_span_\d+)?')

# ================= 1. 讀取 Conversations 資料 =================
print(f"Loading conversations from {CONVERSATIONS_PATH}...")
conv_db = {}

if not os.path.exists(CONVERSATIONS_PATH):
    raise FileNotFoundError(f"Conversations file not found: {CONVERSATIONS_PATH}")

with open(CONVERSATIONS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        conv_id = data['conv_id']
        utt_map = {u['utt_id']: u for u in data['utterances']}
        conv_db[conv_id] = {"utterances": utt_map}

print(f"Loaded {len(conv_db)} conversations.")

# ================= 2. 定義處理函式 (不合併版) =================

def process_single_file_no_merge(csv_path, output_path, target_split_name):
    print(f"\nProcessing split: [{target_split_name}] (NO MERGE)")
    print(f"  - Input: {csv_path}")
    print(f"  - Output: {output_path}")

    if not os.path.exists(csv_path):
        print(f"  [Error] Input CSV not found: {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    pairs_data = []
    
    count_found = 0
    count_missing = 0

    for index, row in df.iterrows():
        raw_id = row['id']
        label = row['labels']
        # 嘗試讀取 span 欄位，如果沒有則為空字串
        span_text = row.get('span', "")
        
        match = id_pattern.search(raw_id)
        if not match:
            continue
            
        split_prefix = match.group('split') 
        conv_num = match.group('conv_num')
        t_idx = match.group('t_idx')
        c_idx = match.group('c_idx')
        
        conv_id = f"{split_prefix}_{conv_num}"
        t_utt_id = f"u{t_idx}"
        c_utt_id = f"u{c_idx}"
        
        # 檢查對話與句子是否存在
        if conv_id not in conv_db:
            count_missing += 1
            continue
            
        utterances = conv_db[conv_id]['utterances']
        if t_utt_id not in utterances or c_utt_id not in utterances:
            count_missing += 1
            continue
            
        t_utt = utterances[t_utt_id]
        c_utt = utterances[c_utt_id]
        
        t_speaker = t_utt['speaker']
        c_speaker = c_utt['speaker']
        
        meta = {
            "t_emotion": t_utt['emotion'],
            "c_emotion": c_utt['emotion'],
            "t_speaker": t_speaker,
            "c_speaker": c_speaker,
            "same_speaker": (t_speaker == c_speaker),
            "self_cause": (t_utt_id == c_utt_id),
            "split": target_split_name
        }
        
        pair_entry = {
            "conv_id": conv_id,
            "t_utt_id": t_utt_id,
            "c_utt_id": c_utt_id,
            "label": int(label),
            "span": span_text,  # [新增] 保留 span 文字以便區分
            "orig_id": raw_id,  # [新增] 保留原始 CSV ID 方便追蹤
            "meta": meta
        }
        
        # [修改關鍵] 直接 Append，完全不檢查重複
        pairs_data.append(pair_entry)
        count_found += 1

    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in pairs_data:
            f.write(json.dumps(entry) + '\n')

    print(f"  - Done. Original CSV rows: {len(df)}")
    print(f"  - Written Pairs: {len(pairs_data)}")
    if count_missing > 0:
        print(f"  - [Warning] {count_missing} entries skipped due to missing ID.")

# ================= 3. 執行批次任務 =================
print("Starting batch processing (Retaining Spans)...")

for task in TASKS:
    process_single_file_no_merge(
        task['csv'], 
        task['out'], 
        task['split_name']
    )

print("\nAll tasks completed.")