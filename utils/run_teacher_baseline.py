#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/utils/run_teacher_baseline.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
from tqdm import tqdm
 # 為了能 import modules，將上一層目錄加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.dataset import CEEEnd2EndDataset

# ============================================================
# Configuration
# ============================================================
class Config:
    BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
    DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
    EXP_DIR_GPT = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
    
    # 這裡是你要測試的目標資料夾 (GPT-4o)
    TARGET_DIR = EXP_DIR_GPT
    
    CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
    TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
    VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
    TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

    # Teacher Vector Paths
    TRAIN_PT  = os.path.join(TARGET_DIR, "explain_train_embeddings.pt")
    TRAIN_TSV = os.path.join(TARGET_DIR, "explain_train_gpt4omini_index.tsv") 
    
    VALID_PT  = os.path.join(TARGET_DIR, "explain_valid_embeddings.pt")
    VALID_TSV = os.path.join(TARGET_DIR, "explain_valid_gpt4omini_index.tsv")
    
    TEST_PT   = os.path.join(TARGET_DIR, "explain_test_embeddings.pt")
    TEST_TSV  = os.path.join(TARGET_DIR, "explain_test_gpt4omini_index.tsv")

    # 參數
    HIDDEN_DIM = 1024
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ============================================================
# Simple MLP Model
# ============================================================
class TeacherProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# Analysis Function (Enhanced for Dist 1-4)
# ============================================================
def analyze_distance_performance(preds, labels, dists):
    """
    精細分析不同距離下的準確度 (針對 Dist 1, 2, 3, 4 及 >=5)
    """
    dists = np.array(dists)
    preds = np.array(preds)
    labels = np.array(labels)
    
    print("\n" + "="*60)
    print("   DISTANCE BREAKDOWN ANALYSIS")
    print("="*60)
    print(f"{'Dist':<8} | {'Count':<6} | {'PosRate':<8} | {'Acc':<8} | {'F1':<8} | {'Recall':<8}")
    print("-" * 60)
    
    # 1. 針對具體距離 1, 2, 3, 4 進行迴圈
    targets = [1, 2, 3, 4]
    
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
        rec = recall_score(sub_labels, sub_preds, zero_division=0, pos_label=1) # 加看 Recall
        pos_rate = np.mean(sub_labels)
        
        print(f"{d:<8} | {count:<6} | {pos_rate:.2f}     | {acc:.4f}   | {f1:.4f}   | {rec:.4f}")

    # 2. 針對長距離 (>= 5) 進行匯總
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

    print("-" * 60)

# ============================================================
# Main Loop
# ============================================================
def main():
    print(f"Running Teacher Baseline Experiment on {cfg.DEVICE}")
    
    # 1. Prepare Data using CEEEnd2EndDataset
    # [修正參數名稱]: explain_embed_path / explain_index_path (原本是 explain_pt_path / explain_tsv_path)
    # 我們不關心 edge explain，所以 expl_embed_path 設 None
    # 關注的是 expl_space (Teacher Vector)
    
    print("Loading Train Dataset...")
    train_ds = CEEEnd2EndDataset(
        cfg.CONV_PATH, cfg.TRAIN_PATH,
        explain_embed_path=None, explain_index_path=None, 
        use_explain=False,
        expl_space_pt=cfg.TRAIN_PT, expl_space_tsv=cfg.TRAIN_TSV,
        use_expl_space=True
    )
    
    print("Loading Valid Dataset...")
    valid_ds = CEEEnd2EndDataset(
        cfg.CONV_PATH, cfg.VALID_PATH,
        explain_embed_path=None, explain_index_path=None, 
        use_explain=False,
        expl_space_pt=cfg.VALID_PT, expl_space_tsv=cfg.VALID_TSV,
        use_expl_space=True
    )
    
    print("Loading Test Dataset...")
    test_ds = CEEEnd2EndDataset(
        cfg.CONV_PATH, cfg.TEST_PATH,
        explain_embed_path=None, explain_index_path=None, 
        use_explain=False,
        expl_space_pt=cfg.TEST_PT, expl_space_tsv=cfg.TEST_TSV,
        use_expl_space=True
    )
    
    # 手寫 Collate: 只需要 Vector, Label, Distance
# [Fix] 修正距離計算邏輯
    def simple_collate(batch_list):
        vecs = []
        labels = []
        dists = []
        for item in batch_list:
            vecs.append(item.expl_space_vec)
            labels.append(item.edge_label)
            
            # 1. 取得超級節點的 Index
            c_node = item.target_node_indices[0, 0].item()
            t_node = item.target_node_indices[0, 1].item()
            
            # 2. 從 Edge Index 中反查它們連接到的真實語句 Index
            # Dataset 建圖時：CauseNode -> CauseUtt (Edge Type 0)
            # 我們在 edge_index[0] (Source) 中找 c_node
            
            edges = item.edge_index
            
            # 找 c_node 連去哪 (取第一個找到的鄰居)
            c_mask = (edges[0] == c_node)
            if c_mask.any():
                c_utt_idx = edges[1][c_mask][0].item()
            else:
                c_utt_idx = 0 # Fallback
                
            # 找 t_node 連去哪
            t_mask = (edges[0] == t_node)
            if t_mask.any():
                t_utt_idx = edges[1][t_mask][0].item()
            else:
                t_utt_idx = 0
            
            # 3. 計算真實語句距離
            real_dist = abs(t_utt_idx - c_utt_idx)
            dists.append(real_dist)
            
        return {
            "vec": torch.stack(vecs),
            "label": torch.stack(labels).float().squeeze(),
            "dist": torch.tensor(dists)
        }

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=simple_collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=simple_collate)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=simple_collate)
    
    # 檢查維度
    sample_dim = train_ds[0].expl_space_vec.shape[0]
    print(f"Detected Vector Dim: {sample_dim}")
    
    # 2. Model & Opt
    model = TeacherProbe(sample_dim, cfg.HIDDEN_DIM).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Train
    best_f1 = 0.0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            vecs = batch['vec'].to(cfg.DEVICE)
            labels = batch['label'].to(cfg.DEVICE)
            
            optimizer.zero_grad()
            logits = model(vecs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                vecs = batch['vec'].to(cfg.DEVICE)
                labels = batch['label']
                logits = model(vecs)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        f1 = f1_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Valid F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_teacher_probe.pt")
            
    print(f"\n>>> Best Valid F1: {best_f1:.4f}")
    
    # 4. Final Test
    print("\nRunning Final Test Analysis...")
    model.load_state_dict(torch.load("best_teacher_probe.pt"))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_dists = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            vecs = batch['vec'].to(cfg.DEVICE)
            labels = batch['label']
            dists = batch['dist']
            
            logits = model(vecs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())
            test_dists.extend(dists.numpy())
            test_probs.extend(probs)
            
    # Metrics
    macro = f1_score(test_labels, test_preds, average='macro')
    pos_f1 = f1_score(test_labels, test_preds, pos_label=1)
    neg_f1 = f1_score(test_labels, test_preds, pos_label=0)
    auc = roc_auc_score(test_labels, test_probs)
    
    print(f"Test Macro F1: {macro:.4f}")
    print(f"Test Pos F1:   {pos_f1:.4f}")
    print(f"Test Neg F1:   {neg_f1:.4f}")
    print(f"Test AUC:      {auc:.4f}")
    
    # Distance Breakdown
    analyze_distance_performance(test_preds, test_labels, test_dists)

if __name__ == "__main__":
    main()