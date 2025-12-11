#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/utils/run_text_inference_baseline.py

import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score
from tqdm import tqdm

class Config:
    BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
    DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
    
    # LLM 輸出路徑
    LLM_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs/plaintext_gpt4omini/explain_test_gpt4omini.jsonl")
    
    # Ground Truth 路徑
    CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
    TEST_PAIRS = os.path.join(DATA_DIR, "pairs_test.jsonl")

cfg = Config()

def load_utterance_indices():
    """建立 conv_id -> {utt_id -> index} 映射，用於計算距離"""
    print("Loading conversation structures...")
    conv_map = {}
    with open(cfg.CONV_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            conv_id = data['conv_id']
            utts = sorted(data['utterances'], key=lambda x: x['turn'])
            utt_index = {u['utt_id']: i for i, u in enumerate(utts)}
            conv_map[conv_id] = utt_index
    return conv_map

def load_ground_truth(conv_map):
    """載入 Ground Truth 並計算距離"""
    print("Loading Ground Truth from pairs_test.jsonl...")
    gt_data = {}
    with open(cfg.TEST_PAIRS, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 格式: conv_id__c_id__t_id
            pair_id = f"{data['conv_id']}__{data['c_utt_id']}__{data['t_utt_id']}"
            
            u_map = conv_map.get(data['conv_id'])
            dist = -1
            if u_map:
                c_idx = u_map.get(data['c_utt_id'])
                t_idx = u_map.get(data['t_utt_id'])
                if c_idx is not None and t_idx is not None:
                    dist = abs(t_idx - c_idx)
            
            gt_data[pair_id] = {
                "label": int(data['label']),
                "dist": dist
            }
    print(f"Loaded {len(gt_data)} ground truth pairs.")
    return gt_data

def load_llm_predictions(path):
    """
    [修正版] 根據 label 欄位做精確比對
    不使用 score
    """
    print(f"Loading LLM predictions from {path}...")
    preds = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                pid = data['id']
                raw_label = data.get('label', '').strip()
                
                # [關鍵修正]: 精確比對
                # 如果是 "cause" 則為 1
                # 如果是 "not_cause" 則為 0
                if raw_label == "cause":
                    pred = 1
                else:
                    pred = 0
                    
                preds[pid] = pred
            except Exception as e:
                print(f"Error parsing line: {line[:50]}... {e}")
                continue
                
    print(f"Loaded {len(preds)} predictions.")
    return preds

def analyze_distance_performance(preds, labels, dists):
    dists = np.array(dists)
    preds = np.array(preds)
    labels = np.array(labels)
    
    unique_dists = sorted(np.unique(dists))
    
    print("\n" + "="*65)
    print("   LLM DIRECT PREDICTION - DISTANCE BREAKDOWN (Label Based)")
    print("="*65)
    print(f"{'Dist':<8} | {'Count':<6} | {'PosRate':<8} | {'Acc':<8} | {'F1':<8} | {'Recall':<8}")
    print("-" * 65)
    
    targets = [0,1, 2, 3, 4]
    for d in targets:
        mask = (dists == d)
        sub_preds = preds[mask]
        sub_labels = labels[mask]
        
        count = len(sub_labels)
        if count == 0:
            print(f"{d:<8} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            continue
        
        acc = accuracy_score(sub_labels, sub_preds)
        f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        rec = recall_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        pos_rate = np.mean(sub_labels)
        
        print(f"{d:<8} | {count:<6} | {pos_rate:.2f}     | {acc:.4f}   | {f1:.4f}   | {rec:.4f}")

    mask_long = (dists >= 5)
    sub_preds_long = preds[mask_long]
    sub_labels_long = labels[mask_long]
    count_long = len(sub_labels_long)
    
    if count_long > 0:
        acc = accuracy_score(sub_labels_long, sub_preds_long)
        f1 = f1_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        rec = recall_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        pos_rate = np.mean(sub_labels_long)
        print(f"{'>=5':<8} | {count_long:<6} | {pos_rate:.2f}     | {acc:.4f}   | {f1:.4f}   | {rec:.4f}")
    else:
        print(f"{'>=5':<8} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

    print("-" * 65)

def main():
    # 1. Load Maps & GT
    conv_map = load_utterance_indices()
    gt_data = load_ground_truth(conv_map)
    
    # 2. Load Preds (Corrected)
    llm_preds = load_llm_predictions(cfg.LLM_OUTPUT_PATH)
    
    # 3. Align Data
    y_true = []
    y_pred = []
    dists = []
    
    missing = 0
    for pid, info in gt_data.items():
        y_true.append(info['label'])
        dists.append(info['dist'])
        
        if pid in llm_preds:
            y_pred.append(llm_preds[pid])
        else:
            y_pred.append(0) # Default negative if missing
            missing += 1
            
    if missing > 0:
        print(f"Warning: {missing} pairs missing in LLM output.")

    # 4. Metrics
    macro = f1_score(y_true, y_pred, average='macro')
    pos_f1 = f1_score(y_true, y_pred, pos_label=1)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("   LLM DIRECT TEXT EVALUATION")
    print("="*40)
    print(f"Macro F1: {macro:.4f}")
    print(f"Pos F1:   {pos_f1:.4f}")
    print(f"Neg F1:   {neg_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    # 5. Breakdown
    analyze_distance_performance(y_pred, y_true, dists)

if __name__ == "__main__":
    main()