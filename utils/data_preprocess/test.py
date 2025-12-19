import pandas as pd
import re
import os
from collections import defaultdict

# ================= 設定區 =================
# 請修改這裡指向你的 CSV 資料夾路徑
BASE_CSV_DIR = '/home/joung/r13725060/Research/RECCON/data/subtask2/fold1'

# 要檢查的檔案清單
TARGET_FILES = [
    'dailydialog_classification_valid_without_context.csv',
]

# 用來解析 ID 的正則表達式
id_pattern = re.compile(r'dailydialog_(?P<split>\w+)_(?P<conv_num>\d+)_utt_(?P<t_idx>\d+)_(?:.*)_cause_utt_(?P<c_idx>\d+)(?:_span_\d+)?')

def analyze_merges(csv_path):
    print(f"\n{'='*60}")
    print(f"正在分析檔案: {os.path.basename(csv_path)}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 字典結構: Key=(conv_id, t_utt_id, c_utt_id), Value=List of rows
    pair_groups = defaultdict(list)
    parse_errors = 0

    # 1. 分組
    for index, row in df.iterrows():
        raw_id = row['id']
        span_text = row.get('span', 'N/A') # 有些版本可能欄位名不同
        label = row['labels']

        match = id_pattern.search(raw_id)
        if not match:
            parse_errors += 1
            continue

        split_prefix = match.group('split')
        conv_num = match.group('conv_num')
        t_idx = match.group('t_idx')
        c_idx = match.group('c_idx')

        # 這是唯一識別一組 Pair 的 Key
        unique_key = (f"{split_prefix}_{conv_num}", f"u{t_idx}", f"u{c_idx}")
        
        # 儲存該行的資訊
        pair_groups[unique_key].append({
            "id": raw_id,
            "span": span_text,
            "label": label
        })

    # 2. 統計
    total_raw_rows = len(df)
    unique_pairs_count = len(pair_groups)
    merged_groups = {k: v for k, v in pair_groups.items() if len(v) > 1}
    merged_count = len(merged_groups)
    rows_eliminated = total_raw_rows - unique_pairs_count - parse_errors

    # 3. 輸出報告
    print(f"📊 統計數據:")
    print(f"  - 原始 CSV 總行數: {total_raw_rows}")
    print(f"  - 唯一 Pair 數量 (JSONL 最終數量): {unique_pairs_count}")
    print(f"  - 包含多個 Span 的重複組數: {merged_count} 組")
    print(f"  - 因合併減少的行數: {rows_eliminated}")
    if parse_errors > 0:
        print(f"  - Regex 解析失敗: {parse_errors}")

    # 4. 印出詳細範例 (前 3 組)
    if merged_count > 0:
        print(f"\n🔍 發現 {merged_count} 組重複資料，以下列出前 3 組範例：")
        
        for i, (key, entries) in enumerate(merged_groups.items()):
            if i >= 3: break
            
            conv_id, t_id, c_id = key
            print(f"\n  [範例 {i+1}] Conv: {conv_id} | Target: {t_id} | Cause: {c_id}")
            print(f"  共 {len(entries)} 筆原始資料被合併:")
            
            for ent in entries:
                label_str = "Positive (1)" if ent['label'] == 1 else "Negative (0)"
                print(f"    - Label: {ent['label']} | Span: \"{ent['span']}\"")
                # print(f"      ID: {ent['id']}") # 如果想看原始 ID 可打開這行

if __name__ == "__main__":
    for filename in TARGET_FILES:
        file_path = os.path.join(BASE_CSV_DIR, filename)
        analyze_merges(file_path)