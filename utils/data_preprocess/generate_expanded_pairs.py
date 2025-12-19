import json
import pandas as pd
import re
import os
from collections import defaultdict

# ================= 設定區 =================
CONVERSATIONS_PATH = '/home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl'
BASE_CSV_DIR = '/home/joung/r13725060/Research/RECCON/data/subtask2/fold1'
BASE_OUT_DIR = '/home/joung/r13725060/Research/IMCEE/data/preprocess_expanded_final'

# 確保輸出目錄存在
os.makedirs(BASE_OUT_DIR, exist_ok=True)

TASKS = [
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_train_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_train.jsonl'),
        "split_name": "train"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_valid_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_valid.jsonl'),
        "split_name": "valid"
    },
    {
        "csv": os.path.join(BASE_CSV_DIR, 'dailydialog_classification_test_without_context.csv'),
        "out": os.path.join(BASE_OUT_DIR, 'new_pairs_test.jsonl'),
        "split_name": "test"
    }
]

id_pattern = re.compile(r'dailydialog_(?P<split>\w+)_(?P<conv_num>\d+)_utt_(?P<t_idx>\d+)_(?:.*)_cause_utt_(?P<c_idx>\d+)(?:_span_\d+)?')

# ================= 1. 讀取 Conversations 資料 =================
print(f"Loading conversations from {CONVERSATIONS_PATH}...")
conv_db = {}
if not os.path.exists(CONVERSATIONS_PATH):
    raise FileNotFoundError(f"Conversations file not found: {CONVERSATIONS_PATH}")

with open(CONVERSATIONS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        conv_db[data['conv_id']] = data 

print(f"Loaded {len(conv_db)} conversations.")

# ================= 2. 定義處理函式 (全上下文 + 去重) =================

def process_single_file_final(csv_path, output_path, target_split_name):
    print(f"\n{'='*60}")
    print(f"Processing split: [{target_split_name}] (Past + Future + Deduplicated)")
    print(f"{'='*60}")
    
    if not os.path.exists(csv_path):
        print(f"  [Error] Input CSV not found: {csv_path}. Skipping.")
        return None

    df = pd.read_csv(csv_path)
    
    # --- Phase 1: 建立 Ground Truth 索引 ---
    # 目的：先收集 CSV 裡確定的正樣本，並確認哪些是 Target
    ground_truth = defaultdict(lambda: defaultdict(set)) # {conv_id: {t_id: {c_ids...}}}
    valid_targets = defaultdict(set) # {conv_id: {t_ids...}}

    for index, row in df.iterrows():
        raw_id = row['id']
        label = row['labels']
        match = id_pattern.search(raw_id)
        if not match: continue
            
        split_prefix = match.group('split')
        conv_num = match.group('conv_num')
        t_idx = match.group('t_idx')
        c_idx = match.group('c_idx')
        
        conv_id = f"{split_prefix}_{conv_num}"
        t_utt_id = f"u{t_idx}"
        c_utt_id = f"u{c_idx}"
        
        # 只要這個句子出現在 CSV 的 Target 位置，它就是我們關注的「非中性情緒句」
        valid_targets[conv_id].add(t_utt_id)
        
        # 記錄正樣本
        if label == 1:
            ground_truth[conv_id][t_utt_id].add(c_utt_id)

    # --- Phase 2: 生成 Full Pairs 並 去重 ---
    final_pairs_data = []
    seen_pairs = set() # [去重關鍵] 記錄已經生成的 (conv_id, t_id, c_id)
    
    # 遍歷所有包含目標句的對話
    for conv_id, t_id_set in valid_targets.items():
        if conv_id not in conv_db: continue
        
        conv_data = conv_db[conv_id]
        utterances_list = conv_data['utterances']
        
        # 針對該對話中每一個 Target (情緒句)
        for t_utt_id in t_id_set:
            t_utt_obj = next((u for u in utterances_list if u['utt_id'] == t_utt_id), None)
            if not t_utt_obj: continue
            
            # [核心變更] 遍歷對話中「所有」語句作為候選 (Past + Future)
            for c_utt_obj in utterances_list:
                c_utt_id = c_utt_obj['utt_id']
                
                # [去重檢查] 
                # 防止 CSV 讀取或其他邏輯造成重複 (雖然目前邏輯不太會，但加這層最保險)
                unique_key = (conv_id, t_utt_id, c_utt_id)
                if unique_key in seen_pairs:
                    continue
                seen_pairs.add(unique_key)
                
                # 判斷標籤 (只要在 Ground Truth 裡就是 1，否則 0)
                is_cause = 1 if c_utt_id in ground_truth[conv_id][t_utt_id] else 0
                
                # 準備 Meta 資訊
                t_idx_int = int(t_utt_id[1:])
                c_idx_int = int(c_utt_id[1:])
                
                meta = {
                    "t_emotion": t_utt_obj['emotion'],
                    "c_emotion": c_utt_obj['emotion'],
                    "t_speaker": t_utt_obj['speaker'],
                    "c_speaker": c_utt_obj['speaker'],
                    "same_speaker": (t_utt_obj['speaker'] == c_utt_obj['speaker']),
                    "self_cause": (t_utt_id == c_utt_id),
                    "split": target_split_name,
                    "distance": t_idx_int - c_idx_int 
                    # distance > 0: Cause 在 Past
                    # distance < 0: Cause 在 Future
                }
                
                final_pairs_data.append({
                    "conv_id": conv_id,
                    "t_utt_id": t_utt_id,
                    "c_utt_id": c_utt_id,
                    "label": is_cause,
                    "meta": meta
                })

    # --- Phase 3: 寫入檔案 ---
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_pairs_data:
            f.write(json.dumps(entry) + '\n')

    # --- Phase 4: 統計 ---
    total = len(final_pairs_data)
    pos_count = sum(1 for p in final_pairs_data if p['label'] == 1)
    neg_count = sum(1 for p in final_pairs_data if p['label'] == 0)
    ratio = neg_count / pos_count if pos_count > 0 else 0
    
    # 統計時間分佈 (確保有包含未來)
    future_neg = sum(1 for p in final_pairs_data if p['label'] == 0 and p['meta']['distance'] < 0)
    past_neg = sum(1 for p in final_pairs_data if p['label'] == 0 and p['meta']['distance'] > 0)

    print(f"Output saved to: {output_path}")
    print(f"Total Unique Pairs: {total}")
    print(f"Positives         : {pos_count}")
    print(f"Negatives         : {neg_count}")
    print(f"Neg/Pos Ratio     : 1 : {ratio:.2f}")
    print(f"Negative Breakdown: Past={past_neg}, Future={future_neg} (Check if > 0)")

    return {
        "split": target_split_name,
        "total": total,
        "pos": pos_count,
        "neg": neg_count,
        "ratio": ratio
    }

# ================= 3. 執行 =================
print("Starting Final Generation...")

summary_stats = []

for task in TASKS:
    stats = process_single_file_final(
        task['csv'], 
        task['out'], 
        task['split_name']
    )
    if stats:
        summary_stats.append(stats)

# ================= 4. 總結 =================
print("\n")
print("="*65)
print(f"{'FINAL SUMMARY (Expanded + Deduplicated)':^65}")
print("="*65)
print(f"{'Split':<10} | {'Total':<10} | {'Pos':<8} | {'Neg':<10} | {'Ratio':<12}")
print("-" * 65)
for s in summary_stats:
    print(f"{s['split']:<10} | {s['total']:<10} | {s['pos']:<8} | {s['neg']:<10} | {s['ratio']:.2f}")
print("="*65)