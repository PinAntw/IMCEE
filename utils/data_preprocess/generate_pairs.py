import json
import pandas as pd
import re
import os

# ================= 設定區 =================

# 1. Conversations 檔案路徑 (共用的)
CONVERSATIONS_PATH = '/home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl'

# 2. 設定要處理的任務清單
# 格式: (CSV輸入路徑, JSONL輸出路徑, Split標記)
# 請確認 validation 的檔名是 'dev' 還是 'validation'
BASE_CSV_DIR = '/home/joung/r13725060/Research/RECCON/data/subtask2/fold1'
BASE_OUT_DIR = '/home/joung/r13725060/Research/IMCEE/data/preprocess'

TASKS = [
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_train_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_train.jsonl'),
        "split_name": "train"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_valid_without_context.csv'), # 注意：如果你的檔名是 validation 請修改這裡
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_valid.jsonl'),
        "split_name": "valid"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_test_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_test.jsonl'),
        "split_name": "test"
    }
]

# Regex 用來解析 ID
# 支援 tr, te, dev, val 等縮寫
id_pattern = re.compile(r'dailydialog_(?P<split>\w+)_(?P<conv_num>\d+)_utt_(?P<t_idx>\d+)_(?:.*)_cause_utt_(?P<c_idx>\d+)(?:_span_\d+)?')

# ================= 1. 讀取 Conversations 資料 (只讀一次) =================
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

# ================= 2. 定義處理函式 =================

def process_single_file(csv_path, output_path, target_split_name):
    print(f"\nProcessing split: [{target_split_name}]")
    print(f"  - Input: {csv_path}")
    print(f"  - Output: {output_path}")

    if not os.path.exists(csv_path):
        print(f"  [Error] Input CSV not found: {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    pairs_data = []
    
    # 統計用
    count_found = 0
    count_missing = 0

    for index, row in df.iterrows():
        raw_id = row['id']
        label = row['labels']
        
        match = id_pattern.search(raw_id)
        if not match:
            # print(f"Warning: Could not parse ID format: {raw_id}")
            continue
            
        split_prefix = match.group('split') # e.g., "tr", "te", "va"
        conv_num = match.group('conv_num')  # e.g., "354"
        t_idx = match.group('t_idx')        # e.g., "2"
        c_idx = match.group('c_idx')        # e.g., "1"
        
        # 組合 conversation ID (e.g., tr_354, te_100)
        conv_id = f"{split_prefix}_{conv_num}"
        
        t_utt_id = f"u{t_idx}"
        c_utt_id = f"u{c_idx}"
        
        # 檢查該對話是否存在於 JSONL DB 中
        if conv_id not in conv_db:
            # 有時候 dev 的前綴可能不一致，這裡做個簡單的檢查
            count_missing += 1
            continue
            
        utterances = conv_db[conv_id]['utterances']
        
        if t_utt_id not in utterances or c_utt_id not in utterances:
            count_missing += 1
            continue
            
        t_utt = utterances[t_utt_id]
        c_utt = utterances[c_utt_id]
        
        # 準備 Meta 資料
        t_speaker = t_utt['speaker']
        c_speaker = c_utt['speaker']
        
        meta = {
            "t_emotion": t_utt['emotion'],
            "c_emotion": c_utt['emotion'],
            "t_speaker": t_speaker,
            "c_speaker": c_speaker,
            "same_speaker": (t_speaker == c_speaker),
            "self_cause": (t_utt_id == c_utt_id),
            "split": target_split_name  # 使用我們傳入的標準名稱 (train/dev/test)
        }
        
        pair_entry = {
            "conv_id": conv_id,
            "t_utt_id": t_utt_id,
            "c_utt_id": c_utt_id,
            "label": int(label),
            "meta": meta
        }
        
        # 處理重複 Pair (Span 合併邏輯)
        # 檢查 list 中是否已經有這個 pair
        existing_entry = next((item for item in pairs_data if item["conv_id"] == conv_id and item["t_utt_id"] == t_utt_id and item["c_utt_id"] == c_utt_id), None)
        
        if existing_entry:
            # 如果任一 span 是正樣本 (1)，則該 pair 就是正樣本
            if int(label) == 1:
                existing_entry['label'] = 1
        else:
            pairs_data.append(pair_entry)
            count_found += 1

    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in pairs_data:
            f.write(json.dumps(entry) + '\n')

    print(f"  - Done. Written {len(pairs_data)} pairs.")
    if count_missing > 0:
        print(f"  - [Warning] {count_missing} entries skipped due to missing ID/Utterance.")

# ================= 3. 執行批次任務 =================
print("Starting batch processing...")

for task in TASKS:
    process_single_file(
        task['csv'], 
        task['out'], 
        task['split_name']
    )

print("\nAll tasks completed.")