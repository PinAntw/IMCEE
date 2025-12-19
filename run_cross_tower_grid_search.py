#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/run_cross_tower_grid_search_resume.py

import os
import json
import random
import numpy as np
import traceback
import pandas as pd  # 必須安裝 pandas
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# 假設 modules 在同一層目錄或 python path 中
from modules.dataset import CEEEnd2EndDataset, End2EndCollate
from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# ============================================================
# Configuration
# ============================================================
class PathConfig:
    BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
    DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
    EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
    EXP_DIR_GPT  = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
    
    # 這裡改成一個專門的 checkpoint 目錄，以免跟手動跑的混淆
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower_auto")
    
    # 這是最後儲存進度 CSV 的路徑
    REPORT_PATH = os.path.join(BASE_DIR, "outputs", "grid_search_progress.csv")
    
    SAVE_SUFFIX = "default"
    
    CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
    TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
    VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
    TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

    EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
    EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
    EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
    EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
    EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
    EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

    ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
    ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
    ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
    ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
    ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
    ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

    @property
    def SAVE_MODEL_PATH(self):
        os.makedirs(self.CKPT_DIR, exist_ok=True)
        return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
    @property
    def OUT_JSON_PATH(self):
        out_dir = os.path.join(self.BASE_DIR, "outputs", "auto_results")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"test_result_{self.SAVE_SUFFIX}.json")

class ModelConfig:
    TEXT_MODEL = "roberta-base"
    HIDDEN_DIM = 768
    
    # Grid Search Parameters (Defaults)
    FREEZE_TEXT = False
    NUM_GNN_LAYERS = 6
    
    GNN_DROPOUT    = 0.3
    BASE_DROPOUT   = 0.2
    USE_EXPLAIN = False      
    USE_EXPLAIN_SPACE = True 

class TrainConfig:
    SEED = 42
    EPOCHS = 8
    BATCH_SIZE = 4
    ACCUM_STEPS = 1
    PATIENCE = 5
    WARMUP_RATIO = 0.1
    POS_WEIGHT_MULT = 1
    
    LAMBDA_EXPL = 1.0   

class OptimConfig:
    LR_BASE = 1e-4
    WD_BASE = 1e-4
    LR_PLM  = 3e-6
    WD_PLM  = 0.01
    LR_GNN  = 1e-3
    WD_GNN  = 0.0

class Config:
    def __init__(self):
        self.path = PathConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.optim = OptimConfig()

cfg = Config()

# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_pos_weight_from_dataset(train_dataset):
    pos_count, neg_count = 0, 0
    for d in train_dataset.data_list:
        if d.edge_label.item() == 1:
            pos_count += 1
        else:
            neg_count += 1
    ratio = neg_count / pos_count if pos_count > 0 else 1.0
    return ratio * cfg.train.POS_WEIGHT_MULT

def calculate_pair_distance(batch):
    if hasattr(batch, "pair_uttpos") and batch.pair_uttpos is not None:
        pos = batch.pair_uttpos
        d = (pos[:, 1] - pos[:, 0]).abs()
        return d.detach().cpu().numpy()
    
    batch_dists = []
    cause_nodes = batch.target_node_indices[:, 0]
    edge_src = batch.edge_index[0]
    edge_tgt = batch.edge_index[1]
    
    for i in range(len(cause_nodes)):
        c_node = cause_nodes[i].item()
        mask_c = (edge_src == c_node)
        c_utt_idx = edge_tgt[mask_c].min().item() if mask_c.any() else 0
        
        t_node = batch.target_node_indices[:, 1][i].item()
        mask_t = (edge_src == t_node)
        t_utt_idx = edge_tgt[mask_t].min().item() if mask_t.any() else 0

        dist = abs(t_utt_idx - c_utt_idx)
        batch_dists.append(dist)
        
    return np.array(batch_dists)

def analyze_distance_performance(preds, labels, dists):
    dists = np.array(dists)
    preds = np.array(preds)
    labels = np.array(labels)
    
    stats_log = {}
    stats_log['rows'] = []
    
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    for d in targets:
        mask = (dists == d)
        sub_preds = preds[mask]
        sub_labels = labels[mask]
        count = len(sub_labels)
        
        if count == 0: continue
        
        acc = accuracy_score(sub_labels, sub_preds)
        pos_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        neg_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=0)
        rec = recall_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        pos_rate = np.mean(sub_labels)
        
        row_str = f"{d:<5} | {count:<6} | {pos_rate:.2f}     | {acc:.4f}   | {pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}"
        stats_log['rows'].append(row_str)

    mask_long = (dists >= 5)
    if mask_long.any():
        sub_preds_long = preds[mask_long]
        sub_labels_long = labels[mask_long]
        count_long = len(sub_labels_long)
        
        acc = accuracy_score(sub_labels_long, sub_preds_long)
        pos_f1 = f1_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        neg_f1 = f1_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=0)
        rec = recall_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        pos_rate = np.mean(sub_labels_long)
        
        row_str = f"{'>=5':<5} | {count_long:<6} | {pos_rate:.2f}     | {acc:.4f}   | {pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}"
        stats_log['rows'].append(row_str)
    
    return stats_log

def monitor_gated_stats(model, loader, device, desc="Monitoring", max_batches=50):
    model.eval()
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    gate_sum, gate_n = 0.0, 0
    sim_sum,  sim_n  = 0.0, 0
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(device)
            out = model(batch, return_aux=True)
            if isinstance(out, tuple) and len(out) == 2: _, aux = out
            else: aux = {}
            
            gate = aux.get("gate", None)
            if gate is not None:
                gate_sum += gate.mean().item()
                gate_n += 1
            z_stud = aux.get("z_student", None)
            z_teach = aux.get("z_teacher", None)
            if z_stud is not None and z_teach is not None:
                sim_sum += cosine_sim(z_stud, z_teach).mean().item()
                sim_n += 1
            seen += 1
            if seen >= max_batches: break
    
    log_str = ""
    if gate_n > 0: log_str += f"Gate: {gate_sum/gate_n:.4f} "
    if sim_n > 0: log_str += f"ST-Align: {sim_sum/sim_n:.4f}"
    return log_str

def save_progress_to_csv(result_dict, file_path):
    """
    Append a single result dictionary to the CSV file.
    """
    # 定義我們關心的欄位順序
    columns = [
        "run_id", "freeze_text", "gnn_layers", 
        "test_macro_f1", "test_pos_f1", "test_neg_f1", 
        "test_acc", "test_auc", "best_val_macro", "error", "gate_stats"
    ]
    
    # 建立 DataFrame (單行)
    df = pd.DataFrame([result_dict])
    
    # 如果檔案不存在，寫入 Header；如果存在，不寫入 Header
    if not os.path.exists(file_path):
        # 確保只有指定欄位 (如果有額外欄位如 dist_breakdown, 這些不適合放 csv)
        # 我們只存 flat metrics
        df_to_save = df.reindex(columns=columns, fill_value="") 
        df_to_save.to_csv(file_path, index=False, mode='w')
    else:
        # 讀取現有 Header 以確保順序一致 (雖然 mode='a' 通常只需對應 columns)
        # 為了安全，我們 reindex
        df_to_save = df.reindex(columns=columns, fill_value="")
        df_to_save.to_csv(file_path, index=False, mode='a', header=False)

    print(f" >> Progress saved to {file_path}")

# ============================================================
# Core Logic
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating", return_dist_stats=False):
    model.eval()
    all_logits, all_labels = [], []
    all_dists = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        labels = batch.edge_label.float()
        if logits.numel() == 0: continue
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        if return_dist_stats:
            dists = calculate_pair_distance(batch)
            all_dists.extend(dists)

    if not all_logits: return None

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    all_probs  = torch.sigmoid(all_logits_t).cpu().numpy()
    all_labels = all_labels_t.cpu().numpy()

    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5

    best_res = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
    thr_candidates = np.concatenate([np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)])
    final_preds = None
    
    for thr in thr_candidates:
        preds = (all_probs >= thr).astype(int)
        pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
        macro = (pos_f1 + neg_f1) / 2.0
        acc = accuracy_score(all_labels, preds)

        if macro > best_res["macro_f1"]:
            best_res.update({
                "macro_f1": float(macro),
                "pos_f1": float(pos_f1),
                "neg_f1": float(neg_f1),
                "acc": float(acc),
                "thr": float(thr),
            })
            if return_dist_stats:
                final_preds = preds

    print(f"[{desc}] MacroF1: {best_res['macro_f1']:.4f} (Pos={best_res['pos_f1']:.2f}, Neg={best_res['neg_f1']:.2f})")
    
    dist_stats = None
    if return_dist_stats and final_preds is not None:
        dist_stats = analyze_distance_performance(final_preds, all_labels, all_dists)
        best_res["dist_stats"] = dist_stats
        
    return best_res

def run_experiment_pipeline(freeze_text, num_gnn_layers, device):
    run_id = f"gated_L{num_gnn_layers}_Fz{'T' if freeze_text else 'F'}"
    print(f"\n{'#'*60}")
    print(f"STARTING EXPERIMENT: {run_id}")
    print(f"  > Freeze Text: {freeze_text}")
    print(f"  > GNN Layers : {num_gnn_layers}")
    print(f"{'#'*60}\n")
    
    cfg.model.FREEZE_TEXT = freeze_text
    cfg.model.NUM_GNN_LAYERS = num_gnn_layers
    cfg.path.SAVE_SUFFIX = run_id

    # 1. Prepare Datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
    collate_fn = End2EndCollate(tokenizer)

    train_dataset = CEEEnd2EndDataset(
        cfg.path.CONV_PATH, cfg.path.TRAIN_PATH,
        cfg.path.EX_TRAIN_PT, cfg.path.EX_TRAIN_TSV,
        use_explain=cfg.model.USE_EXPLAIN,
        expl_space_pt=cfg.path.ES_TRAIN_PT, expl_space_tsv=cfg.path.ES_TRAIN_TSV,
        use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
    )
    valid_dataset = CEEEnd2EndDataset(
        cfg.path.CONV_PATH, cfg.path.VALID_PATH,
        cfg.path.EX_VALID_PT, cfg.path.EX_VALID_TSV,
        use_explain=cfg.model.USE_EXPLAIN,
        expl_space_pt=cfg.path.ES_VALID_PT, expl_space_tsv=cfg.path.ES_VALID_TSV,
        use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. Model & Optim
    model = CrossTowerCausalModel(
        text_model_name=cfg.model.TEXT_MODEL,
        hidden_dim=cfg.model.HIDDEN_DIM,
        expl_dim=train_dataset.get_explain_dim(),
        num_speakers=train_dataset.num_speakers,
        num_emotions=train_dataset.num_emotions,
        dropout=cfg.model.BASE_DROPOUT,
        gnn_dropout=cfg.model.GNN_DROPOUT,
        num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
        freeze_text=cfg.model.FREEZE_TEXT,
        use_explain=cfg.model.USE_EXPLAIN,
        expl_space_dim=train_dataset.get_expl_space_dim(),
        use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
    ).to(device)

    plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
    gnn_ids = list(map(id, model.structural_tower.parameters()))
    base_params = filter(lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': cfg.optim.LR_BASE, 'weight_decay': cfg.optim.WD_BASE},
        {'params': model.structural_tower.parameters(), 'lr': cfg.optim.LR_GNN, 'weight_decay': cfg.optim.WD_GNN},
        {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.optim.LR_PLM, 'weight_decay': cfg.optim.WD_PLM},
    ])

    pos_weight = compute_pos_weight_from_dataset(train_dataset)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    distill_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)

    total_steps = len(train_loader) * cfg.train.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.train.WARMUP_RATIO), total_steps)

    # 3. Train
    best_macro = 0.0
    no_improve = 0

    for epoch in range(1, cfg.train.EPOCHS + 1):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"[{run_id}] E{epoch}", leave=False)):
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple) and len(out) == 3:
                logits, z_stud, z_teach = out
            else:
                logits, z_stud, z_teach = out, None, None

            labels = batch.edge_label.float()
            if logits.numel() == 0: continue

            task_loss = loss_fn(logits, labels)
            expl_loss = torch.tensor(0.0, device=device)
            if cfg.model.USE_EXPLAIN_SPACE and z_stud is not None and z_teach is not None:
                target = torch.ones(z_stud.size(0)).to(device)
                expl_loss = distill_loss_fn(z_stud, z_teach, target)

            loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss
            loss = loss / cfg.train.ACCUM_STEPS
            loss.backward()

            if (step + 1) % cfg.train.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        # Valid
        val_res = evaluate(model, valid_loader, device, desc="Validating")
        if val_res and val_res["macro_f1"] > best_macro:
            best_macro = val_res["macro_f1"]
            no_improve = 0
            torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
        else:
            no_improve += 1
        
        if no_improve >= cfg.train.PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val MacroF1: {best_macro:.4f}")
            break

    # 4. Test
    print(f"Loading Best Model for Testing: {cfg.path.SAVE_MODEL_PATH}")
    model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
    
    test_dataset = CEEEnd2EndDataset(
        cfg.path.CONV_PATH, cfg.path.TEST_PATH,
        cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV,
        use_explain=cfg.model.USE_EXPLAIN,
        expl_space_pt=cfg.path.ES_TEST_PT, expl_space_tsv=cfg.path.ES_TEST_TSV,
        use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    gate_stats = monitor_gated_stats(model, test_loader, device, desc="Stats", max_batches=100)
    test_res = evaluate(model, test_loader, device, desc="TESTING FINAL", return_dist_stats=True)
    
    summary = {
        "run_id": run_id,
        "freeze_text": freeze_text,
        "gnn_layers": num_gnn_layers,
        "best_val_macro": best_macro,
        "test_macro_f1": test_res["macro_f1"],
        "test_pos_f1": test_res["pos_f1"],
        "test_neg_f1": test_res["neg_f1"],
        "test_acc": test_res["acc"],
        "test_auc": test_res["auc"],
        "gate_stats": gate_stats,
        "dist_breakdown": test_res.get("dist_stats", {}).get("rows", [])
    }
    
    # Save Full JSON details
    with open(cfg.path.OUT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

# ============================================================
# Main Automation
# ============================================================
def main():
    set_seed(cfg.train.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Automation Start. Device: {device}")
    
    freeze_options = [True]
    gnn_layer_options = [5,8,7,9]
    
    # ----------------------------------------------------
    # Check for Existing Progress (Resume Capability)
    # ----------------------------------------------------
    finished_run_ids = set()
    if os.path.exists(cfg.path.REPORT_PATH):
        try:
            print(f"Found existing progress file: {cfg.path.REPORT_PATH}")
            df_exist = pd.read_csv(cfg.path.REPORT_PATH)
            if "run_id" in df_exist.columns:
                finished_run_ids = set(df_exist["run_id"].astype(str).tolist())
                print(f"Already finished: {finished_run_ids}")
        except Exception as e:
            print(f"⚠️ Could not read existing progress file: {e}")

    # ----------------------------------------------------
    # Grid Search Loop
    # ----------------------------------------------------
    total_combos = len(freeze_options) * len(gnn_layer_options)
    curr_idx = 0
    
    for freeze in freeze_options:
        for layers in gnn_layer_options:
            curr_idx += 1
            run_id = f"gated_L{layers}_Fz{'T' if freeze else 'F'}"
            
            # --- Resume Logic ---
            if run_id in finished_run_ids:
                print(f"\n[SKIP] {run_id} is already in the progress report.")
                continue
            
            print(f"\n\n>>> Progress: {curr_idx}/{total_combos} combos | Target: {run_id} <<<")
            
            try:
                res = run_experiment_pipeline(freeze, layers, device)
                print(f"✅ Success: {res['run_id']} -> Test Macro: {res['test_macro_f1']:.4f}")
                
                # --- Immediate Save ---
                save_progress_to_csv(res, cfg.path.REPORT_PATH)
                
            except Exception as e:
                print(f"❌ Error in combo {run_id}: {e}")
                traceback.print_exc()
                
                error_res = {
                    "run_id": run_id,
                    "freeze_text": freeze,
                    "gnn_layers": layers,
                    "error": str(e)
                }
                # --- Immediate Save Error ---
                save_progress_to_csv(error_res, cfg.path.REPORT_PATH)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("ALL DONE.")
    print(f"Check final results in: {cfg.path.REPORT_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()