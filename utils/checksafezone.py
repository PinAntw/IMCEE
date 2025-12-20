import json
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 設定區 =================
DATA_PATH = '/home/joung/r13725060/Research/IMCEE/data/preprocess/conversations.jsonl'
MODEL_NAME = 'roberta-base'  # 或 'roberta-large'，視您使用的模型而定
MAX_LEN = 512
# =========================================

def analyze_token_length():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    lengths = []
    truncated_count = 0
    total_count = 0
    
    print(f"Reading data from {DATA_PATH}...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            utterances = data['utterances']
            
            # 模擬 Encoder 2 的行為：將所有句子拼在一起
            # 注意：RoBERTa 實際上會是 [CLS] utt1 [SEP] utt2 [SEP] ...
            # 我們這裡用簡單的拼接來估算，誤差極小 (主要看 subword 數量)
            full_text = " ".join([u['text'] for u in utterances])
            
            # 計算 Token 數量 (包含 special tokens)
            token_ids = tokenizer.encode(full_text, add_special_tokens=True)
            length = len(token_ids)
            
            lengths.append(length)
            total_count += 1
            
            if length > MAX_LEN:
                truncated_count += 1

    # --- 統計報告 ---
    lengths = np.array(lengths)
    
    print("\n" + "="*40)
    print(f"📊 Token Length Statistics (Model: {MODEL_NAME})")
    print("="*40)
    print(f"Total Conversations : {total_count}")
    print(f"Truncated Samples   : {truncated_count} ({truncated_count/total_count*100:.2f}%)")
    print(f"Safe Samples (<{MAX_LEN}) : {total_count - truncated_count} ({(total_count - truncated_count)/total_count*100:.2f}%)")
    print("-" * 40)
    print(f"Min Length          : {np.min(lengths)}")
    print(f"Mean Length         : {np.mean(lengths):.2f}")
    print(f"Median Length       : {np.median(lengths):.2f}")
    print(f"Max Length          : {np.max(lengths)}")
    print("-" * 40)
    print(f"75th Percentile     : {np.percentile(lengths, 75):.2f}")
    print(f"90th Percentile     : {np.percentile(lengths, 90):.2f}")
    print(f"95th Percentile     : {np.percentile(lengths, 95):.2f}")
    print(f"99th Percentile     : {np.percentile(lengths, 99):.2f}")
    print("="*40)

    # (選用) 畫圖
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, bins=50, kde=True)
        plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'Limit ({MAX_LEN})')
        plt.title(f'Token Length Distribution ({MODEL_NAME})')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('length_distribution.png')
        print("Histrogram saved to 'length_distribution.png'")
    except:
        print("Skipping plot generation (matplotlib/seaborn missing or display issue).")

if __name__ == "__main__":
    analyze_token_length()