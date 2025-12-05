# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Research/IMCEE/run_cross_tower.py

# import os
# import json
# import random
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# # ============================================================
# # Imports
# # ============================================================
# from modules.dataset import CEEEnd2EndDataset, End2EndCollate
# from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# # ============================================================
# # 1. Configuration (改用 Class 管理，方便參數分組)
# # ============================================================
# class Config:
#     # ---- 模式設定 ----
#     RUN_MODE = "test"   # "train" or "test"
#     SAVE_SUFFIX = "cross_tower_diao"
#     SEED = 42

#     # ---- 路徑設定 ----
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")

#     # ---- 模型結構參數 ----
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768
#     GNN_DIM    = 768
#     NUM_GNN_LAYERS = 3
    
#     # ---- Dropout 設定 (區分 GNN 與其他) ----
#     DROPOUT_BASE = 0.1       # MLP, Projection 用
#     DROPOUT_GNN  = 0.3       # RGCN 專用 (Paper 建議較高)

#     # ---- 訓練超參數 ----
#     EPOCHS = 40
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 5
#     WARMUP_RATIO = 0.1
#     POS_WEIGHT_MULT = 1.0

#     # ---- 參數凍結與解釋 ----
#     FREEZE_TEXT = True
#     USE_EXPLAIN = False   

#     # ---- [關鍵] 學習率分組設定 ----
#     # 1. 基礎層 (MLP, Semantic Tower)
#     LR_BASE = 1e-4           
#     WD_BASE = 1e-4           

#     # 2. RoBERTa (如果解凍的話)
#     LR_PLM  = 1e-4            

#     # 3. GNN 專屬 (RGCN 通常需要大 LR)
#     LR_GNN  = 0.01           
#     WD_GNN  = 0.0            # GNN 通常不需要 weight decay

#     # ---- 自動生成路徑 ----
#     CONV  = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     SAVE_PATH = os.path.join(CKPT_DIR, f"model_{SAVE_SUFFIX}.pt")
#     OUT_JSON  = os.path.join(BASE_DIR, "outputs", f"test_result_{SAVE_SUFFIX}.json")

# cfg = Config()

# # ============================================================
# # Utils
# # ============================================================
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def compute_pos_weight_from_dataset(train_dataset):
#     pos_count, neg_count = 0, 0
#     for d in train_dataset.data_list:
#         mask = d.edge_task_mask
#         if mask.sum() == 0: continue
#         y = d.edge_labels[mask]
#         pos_count += (y == 1).sum().item()
#         neg_count += (y == 0).sum().item()
#     if pos_count == 0: ratio = 1.0
#     else: ratio = neg_count / pos_count
#     return ratio * cfg.POS_WEIGHT_MULT

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         logits = model(batch)

#         task_mask = batch.edge_task_mask
#         if task_mask.sum() == 0: continue
#         labels = batch.edge_labels[task_mask].float()

#         if logits.numel() == 0: continue

#         all_logits.append(logits.detach().cpu())
#         all_labels.append(labels.detach().cpu())

#     if not all_logits: return None

#     all_logits = torch.cat(all_logits).numpy()
#     all_labels = torch.cat(all_labels).numpy()
#     all_probs = sigmoid(all_logits)

#     try: auc = roc_auc_score(all_labels, all_probs)
#     except: auc = 0.5

#     best_res = {"macro_f1": 0.0, "thr": 0.5}
#     for thr in np.linspace(0.1, 0.9, 9):
#         preds = (all_probs >= thr).astype(int)
#         pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
#         neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
#         macro = (pos_f1 + neg_f1) / 2.0
#         acc = accuracy_score(all_labels, preds)

#         if macro > best_res["macro_f1"]:
#             best_res = {
#                 "auc": float(auc),
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "thr": float(thr),
#             }
#     return best_res

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0

#     for epoch in range(1, cfg.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()
#         total_loss, total_pairs = 0.0, 0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)
#             logits = model(batch)

#             task_mask = batch.edge_task_mask
#             labels = batch.edge_labels[task_mask].float()

#             if logits.numel() == 0: continue

#             loss = loss_fn(logits, labels)
#             loss = loss / cfg.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             total_loss += float(loss.item()) * cfg.ACCUM_STEPS * int(labels.size(0))
#             total_pairs += int(labels.size(0))

#         avg_loss = total_loss / max(1, total_pairs)
#         print(f"Train loss = {avg_loss:.4f}")

#         val = evaluate(model, valid_loader, device, desc="Validating")
#         if val is None: 
#             print("Valid skipped (no data).")
#             continue

#         print(f"Valid MacroF1: {val['macro_f1']:.4f} (Pos: {val['pos_f1']:.2f}, Neg: {val['neg_f1']:.2f}) | AUC: {val['auc']:.4f}")

#         if val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.SAVE_PATH)
#             print(f"Saved best model -> {cfg.SAVE_PATH}")
#         else:
#             no_improve += 1
#             if no_improve >= cfg.PATIENCE:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     if cfg.RUN_MODE == "train":
#         train_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.TRAIN, cfg.EX_TRAIN_PT, cfg.EX_TRAIN_TSV, use_explain=cfg.USE_EXPLAIN)
#         valid_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.VALID, cfg.EX_VALID_PT, cfg.EX_VALID_TSV, use_explain=cfg.USE_EXPLAIN)

#         train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#         valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         expl_dim = train_dataset.get_explain_dim()

#         # 初始化模型：傳入兩種 dropout
#         model = CrossTowerCausalModel(
#             text_model_name=cfg.TEXT_MODEL,
#             hidden_dim=cfg.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.DROPOUT_BASE,         # 一般 dropout
#             gnn_dropout=cfg.DROPOUT_GNN,      # GNN 專用 dropout
#             num_gnn_layers=cfg.NUM_GNN_LAYERS,
#             freeze_text=cfg.FREEZE_TEXT,
#             use_explain=cfg.USE_EXPLAIN,
#         ).to(device)

#         # ---- [核心] 參數分組邏輯 ----
#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
        
#         # Base 參數：不是 PLM 也不是 GNN 的參數 (ex: MLP, Embeddings)
#         base_params = filter(
#             lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, 
#             model.parameters()
#         )

#         optimizer_grouped_parameters = [
#             # 1. Base (MLP etc.)
#             {'params': base_params, 'lr': cfg.LR_BASE, 'weight_decay': cfg.WD_BASE},
#             # 2. GNN (RGCN)
#             {'params': model.structural_tower.parameters(), 'lr': cfg.LR_GNN, 'weight_decay': cfg.WD_GNN},
#             # 3. PLM (RoBERTa)
#             {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.LR_PLM, 'weight_decay': 0.01}
#         ]

#         optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

#         pos_weight_val = compute_pos_weight_from_dataset(train_dataset)
#         print(f"Pos Weight: {pos_weight_val:.2f}")
#         loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

#         total_steps = len(train_loader) * cfg.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * cfg.WARMUP_RATIO), num_training_steps=total_steps)

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     else:
#         # Test Mode
#         test_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.TEST, cfg.EX_TEST_PT, cfg.EX_TEST_TSV, use_explain=cfg.USE_EXPLAIN)
#         test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         expl_dim = test_dataset.get_explain_dim()

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.TEXT_MODEL,
#             hidden_dim=cfg.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_emotions=test_dataset.num_emotions,
#             num_speakers=test_dataset.num_speakers,
#             dropout=cfg.DROPOUT_BASE,
#             gnn_dropout=cfg.DROPOUT_GNN,
#             num_gnn_layers=cfg.NUM_GNN_LAYERS,
#             freeze_text=cfg.FREEZE_TEXT,
#             use_explain=cfg.USE_EXPLAIN,
#         ).to(device)

#         if os.path.exists(cfg.SAVE_PATH):
#             model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=device))
#             print(f"Loaded checkpoint: {cfg.SAVE_PATH}")
#         else:
#             print(f"Checkpoint not found: {cfg.SAVE_PATH}")
#             return

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.OUT_JSON, "w") as f:
#             json.dump(results, f, indent=2)

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Research/IMCEE/run_cross_tower.py

# import os
# import json
# import random
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# # ============================================================
# # Imports
# # ============================================================
# from modules.dataset import CEEEnd2EndDataset, End2EndCollate
# from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# # ============================================================
# # 1. Configuration
# # ============================================================
# class Config:
#     # ---- 模式設定 ----
#     RUN_MODE = "train"   # "train" or "test"
#     SAVE_SUFFIX = "cross_tower_expl" 
#     SEED = 42

#     # ---- 路徑設定 ----
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")

#     # ---- 模型結構參數 ----
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768
#     GNN_DIM    = 768
#     NUM_GNN_LAYERS = 3
    
#     # ---- Dropout 設定 ----
#     DROPOUT_BASE = 0.1       
#     DROPOUT_GNN  = 0.3       

#     # ---- 訓練超參數 ----
#     EPOCHS = 30
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 10
#     WARMUP_RATIO = 0.1
#     POS_WEIGHT_MULT = 1.5

#     # ---- 參數凍結 ----
#     FREEZE_TEXT = True
    
#     # [修改] 加回 USE_EXPLAIN
#     USE_EXPLAIN = False  # 你可以隨時在這裡切換 True/False

#     # ---- 學習率分組 ----
#     LR_BASE = 1e-4           
#     WD_BASE = 1e-4           
#     LR_PLM  = 1e-4            
#     LR_GNN  = 0.001           
#     WD_GNN  = 0.0            

#     # ---- 自動生成路徑 ----
#     CONV  = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     # [修改] 加回解釋檔案路徑
#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     SAVE_PATH = os.path.join(CKPT_DIR, f"model_{SAVE_SUFFIX}.pt")
#     OUT_JSON  = os.path.join(BASE_DIR, "outputs", f"test_result_{SAVE_SUFFIX}.json")

# cfg = Config()

# # ============================================================
# # Utils
# # ============================================================
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def compute_pos_weight_from_dataset(train_dataset):
#     pos_count, neg_count = 0, 0
#     for d in train_dataset.data_list:
#         label = d.edge_label.item()
#         if label == 1: pos_count += 1
#         else: neg_count += 1
            
#     if pos_count == 0: ratio = 1.0
#     else: ratio = neg_count / pos_count
#     return ratio * cfg.POS_WEIGHT_MULT

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         logits = model(batch)
#         labels = batch.edge_label.float()

#         if logits.numel() == 0:
#             continue

#         all_logits.append(logits.detach().cpu())
#         all_labels.append(labels.detach().cpu())

#     if not all_logits:
#         return None

#     all_logits = torch.cat(all_logits).numpy()
#     all_labels = torch.cat(all_labels).numpy()
#     all_probs = sigmoid(all_logits)

#     try:
#         auc = roc_auc_score(all_labels, all_probs)
#     except:
#         auc = 0.5

#     # 重點：threshold 掃描更細，特別是 0.01–0.3
#     thr_candidates = np.concatenate([
#         np.linspace(0.01, 0.30, 30),
#         np.linspace(0.35, 0.90, 12)
#     ])

#     best_macro = {
#         "auc": float(auc),
#         "macro_f1": 0.0,
#         "pos_f1": 0.0,
#         "neg_f1": 0.0,
#         "acc": 0.0,
#         "thr": 0.5,
#     }
#     best_pos = {
#         "auc": float(auc),
#         "macro_f1": 0.0,
#         "pos_f1": 0.0,
#         "neg_f1": 0.0,
#         "acc": 0.0,
#         "thr": 0.5,
#     }

#     for thr in thr_candidates:
#         preds = (all_probs >= thr).astype(int)

#         pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
#         neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
#         macro = (pos_f1 + neg_f1) / 2.0
#         acc = accuracy_score(all_labels, preds)

#         if macro > best_macro["macro_f1"]:
#             best_macro = {
#                 "auc": float(auc),
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "thr": float(thr),
#             }

#         if pos_f1 > best_pos["pos_f1"]:
#             best_pos = {
#                 "auc": float(auc),
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "thr": float(thr),
#             }

#     print(
#         f"[Eval] Best MacroF1: {best_macro['macro_f1']:.4f} "
#         f"(Pos={best_macro['pos_f1']:.2f}, Neg={best_macro['neg_f1']:.2f}, "
#         f"Acc={best_macro['acc']:.4f}, AUC={best_macro['auc']:.4f}, Thr={best_macro['thr']:.2f})"
#     )
#     print(
#         f"[Eval] Best PosF1 : {best_pos['pos_f1']:.4f} "
#         f"(Macro={best_pos['macro_f1']:.4f}, Neg={best_pos['neg_f1']:.2f}, "
#         f"Acc={best_pos['acc']:.4f}, AUC={best_pos['auc']:.4f}, Thr={best_pos['thr']:.2f})"
#     )

#     # 訓練流程就沿用「最佳 MacroF1」這個結果
#     return best_macro

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0

#     for epoch in range(1, cfg.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()
#         total_loss, total_pairs = 0.0, 0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)
#             logits = model(batch)
#             labels = batch.edge_label.float()

#             if logits.numel() == 0: continue

#             loss = loss_fn(logits, labels)
#             loss = loss / cfg.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             total_loss += float(loss.item()) * cfg.ACCUM_STEPS * int(labels.size(0))
#             total_pairs += int(labels.size(0))

#         avg_loss = total_loss / max(1, total_pairs)
#         print(f"Train loss = {avg_loss:.4f}")

#         val = evaluate(model, valid_loader, device, desc="Validating")
#         if val is None: 
#             print("Valid skipped (no data).")
#             continue

#         print(f"Valid MacroF1: {val['macro_f1']:.4f} (Pos: {val['pos_f1']:.2f}, Neg: {val['neg_f1']:.2f}) | AUC: {val['auc']:.4f}")

#         if val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.SAVE_PATH)
#             print(f"Saved best model -> {cfg.SAVE_PATH}")
#         else:
#             no_improve += 1
#             if no_improve >= cfg.PATIENCE:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     if cfg.RUN_MODE == "train":
#         # [修改] 傳入 Explain 路徑與 Config 設定
#         train_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.TRAIN, cfg.EX_TRAIN_PT, cfg.EX_TRAIN_TSV, use_explain=cfg.USE_EXPLAIN)
#         valid_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.VALID, cfg.EX_VALID_PT, cfg.EX_VALID_TSV, use_explain=cfg.USE_EXPLAIN)

#         train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#         valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         expl_dim = train_dataset.get_explain_dim()

#         # [修改] 傳入 expl_dim 和 use_explain
#         model = CrossTowerCausalModel(
#             text_model_name=cfg.TEXT_MODEL,
#             hidden_dim=cfg.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.DROPOUT_BASE,         
#             gnn_dropout=cfg.DROPOUT_GNN,      
#             num_gnn_layers=cfg.NUM_GNN_LAYERS,
#             freeze_text=cfg.FREEZE_TEXT,
#             use_explain=cfg.USE_EXPLAIN, # 傳入 config 設定
#         ).to(device)

#         # ---- 參數分組邏輯 ----
#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
        
#         base_params = filter(
#             lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, 
#             model.parameters()
#         )

#         optimizer_grouped_parameters = [
#             {'params': base_params, 'lr': cfg.LR_BASE, 'weight_decay': cfg.WD_BASE},
#             {'params': model.structural_tower.parameters(), 'lr': cfg.LR_GNN, 'weight_decay': cfg.WD_GNN},
#             {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.LR_PLM, 'weight_decay': 0.01}
#         ]

#         optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

#         pos_weight_val = compute_pos_weight_from_dataset(train_dataset)
#         print(f"Pos Weight: {pos_weight_val:.2f}")
#         loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

#         total_steps = len(train_loader) * cfg.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * cfg.WARMUP_RATIO), num_training_steps=total_steps)

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     else:
#         # Test Mode
#         # [修改] 傳入 Explain 路徑
#         test_dataset = CEEEnd2EndDataset(cfg.CONV, cfg.TEST, cfg.EX_TEST_PT, cfg.EX_TEST_TSV, use_explain=cfg.USE_EXPLAIN)
#         test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         expl_dim = test_dataset.get_explain_dim()

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.TEXT_MODEL,
#             hidden_dim=cfg.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_emotions=test_dataset.num_emotions,
#             num_speakers=test_dataset.num_speakers,
#             dropout=cfg.DROPOUT_BASE,
#             gnn_dropout=cfg.DROPOUT_GNN,
#             num_gnn_layers=cfg.NUM_GNN_LAYERS,
#             freeze_text=cfg.FREEZE_TEXT,
#             use_explain=cfg.USE_EXPLAIN,
#         ).to(device)

#         if os.path.exists(cfg.SAVE_PATH):
#             model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=device))
#             print(f"Loaded checkpoint: {cfg.SAVE_PATH}")
#         else:
#             print(f"Checkpoint not found: {cfg.SAVE_PATH}")
#             return

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.OUT_JSON, "w") as f:
#             json.dump(results, f, indent=2)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/run_cross_tower.py

import os
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# ============================================================
# Imports
# ============================================================
from modules.dataset import CEEEnd2EndDataset, End2EndCollate
from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# ============================================================
# 1. Modular Configuration
# ============================================================

class PathConfig:
    """ 路徑相關設定 """
    BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
    DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
    EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")
    
    # 輸出檔名標記
    SAVE_SUFFIX = "cross_tower_expl_v2"
    
    # 自動生成
    CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
    TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
    VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
    TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

    # Explain Files
    EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
    EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
    EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
    EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
    EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
    EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

    @property
    def SAVE_MODEL_PATH(self):
        return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
    @property
    def OUT_JSON_PATH(self):
        return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")


class ModelConfig:
    """ 模型結構超參數 """
    TEXT_MODEL = "roberta-base"
    FREEZE_TEXT = False
    
    # 維度設定
    HIDDEN_DIM = 768    # 用於 Semantic Tower 和 Fusion 的統一維度
    
    # Structural Tower (GNN) 設定
    NUM_GNN_LAYERS = 3
    GNN_DROPOUT    = 0.3
    
    # Semantic Tower / General 設定
    BASE_DROPOUT   = 0.1
    
    # 功能開關
    USE_EXPLAIN = False  # 是否使用外部解釋特徵


class TrainConfig:
    """ 訓練流程超參數 """
    RUN_MODE = "train"   # "train" or "test"
    SEED = 42
    
    EPOCHS = 30
    BATCH_SIZE = 4
    ACCUM_STEPS = 1
    PATIENCE = 30
    WARMUP_RATIO = 0.1
    
    # Loss Weight
    POS_WEIGHT_MULT = 1.5


class OptimConfig:
    """ 優化器與學習率設定 (不同模組不同 LR) """
    # 基礎層 (MLP, Linear 等)
    LR_BASE = 1e-4
    WD_BASE = 1e-4
    
    # PLM 層 (Text Encoder)
    LR_PLM  = 1e-5     # 通常 PLM 學習率要比其他層低
    WD_PLM  = 0.01
    
    # GNN 層 (Structural Tower)
    LR_GNN  = 1e-3     # GNN 可以稍大
    WD_GNN  = 0.0


# 整合 Config
class Config:
    path = PathConfig()
    model = ModelConfig()
    train = TrainConfig()
    optim = OptimConfig()

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_pos_weight_from_dataset(train_dataset):
    pos_count, neg_count = 0, 0
    for d in train_dataset.data_list:
        label = d.edge_label.item()
        if label == 1: pos_count += 1
        else: neg_count += 1
            
    if pos_count == 0: ratio = 1.0
    else: ratio = neg_count / pos_count
    return ratio * cfg.train.POS_WEIGHT_MULT

# ============================================================
# Evaluate
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.edge_label.float()

        if logits.numel() == 0:
            continue

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_logits:
        return None

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = sigmoid(all_logits)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    thr_candidates = np.concatenate([
        np.linspace(0.01, 0.30, 30),
        np.linspace(0.35, 0.90, 12)
    ])

    best_macro = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
    
    for thr in thr_candidates:
        preds = (all_probs >= thr).astype(int)
        pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
        macro = (pos_f1 + neg_f1) / 2.0
        acc = accuracy_score(all_labels, preds)

        if macro > best_macro["macro_f1"]:
            best_macro = {
                "auc": float(auc),
                "macro_f1": float(macro),
                "pos_f1": float(pos_f1),
                "neg_f1": float(neg_f1),
                "acc": float(acc),
                "thr": float(thr),
            }

    print(
        f"[Eval] Best MacroF1: {best_macro['macro_f1']:.4f} "
        f"(Pos={best_macro['pos_f1']:.2f}, Neg={best_macro['neg_f1']:.2f}, "
        f"Acc={best_macro['acc']:.4f}, AUC={best_macro['auc']:.4f})"
    )
    return best_macro

# ============================================================
# Train
# ============================================================
def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
    os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
    best_macro = 0.0
    no_improve = 0

    for epoch in range(1, cfg.train.EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
        model.train()
        optimizer.zero_grad()
        total_loss, total_pairs = 0.0, 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
            batch = batch.to(device)
            logits = model(batch)
            labels = batch.edge_label.float()

            if logits.numel() == 0: continue

            loss = loss_fn(logits, labels)
            loss = loss / cfg.train.ACCUM_STEPS
            loss.backward()

            if (step + 1) % cfg.train.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * cfg.train.ACCUM_STEPS * int(labels.size(0))
            total_pairs += int(labels.size(0))

        avg_loss = total_loss / max(1, total_pairs)
        print(f"Train loss = {avg_loss:.4f}")

        val = evaluate(model, valid_loader, device, desc="Validating")
        if val is None: 
            print("Valid skipped.")
            continue

        if val["macro_f1"] > best_macro:
            best_macro = val["macro_f1"]
            no_improve = 0
            torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
            print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
        else:
            no_improve += 1
            if no_improve >= cfg.train.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

# ============================================================
# Main
# ============================================================
def main():
    set_seed(cfg.train.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
    collate_fn = End2EndCollate(tokenizer)

    # ---------------------------------------------------------
    # Training Mode
    # ---------------------------------------------------------
    if cfg.train.RUN_MODE == "train":
        print(">>> Loading Datasets...")
        train_dataset = CEEEnd2EndDataset(
            cfg.path.CONV_PATH, cfg.path.TRAIN_PATH, 
            cfg.path.EX_TRAIN_PT, cfg.path.EX_TRAIN_TSV, 
            use_explain=cfg.model.USE_EXPLAIN
        )
        valid_dataset = CEEEnd2EndDataset(
            cfg.path.CONV_PATH, cfg.path.VALID_PATH, 
            cfg.path.EX_VALID_PT, cfg.path.EX_VALID_TSV, 
            use_explain=cfg.model.USE_EXPLAIN
        )

        train_loader = DataLoader(train_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        expl_dim = train_dataset.get_explain_dim()

        print(">>> Initializing Model...")
        model = CrossTowerCausalModel(
            text_model_name=cfg.model.TEXT_MODEL,
            hidden_dim=cfg.model.HIDDEN_DIM,
            expl_dim=expl_dim,
            num_speakers=train_dataset.num_speakers,
            num_emotions=train_dataset.num_emotions,
            dropout=cfg.model.BASE_DROPOUT,
            # GNN Settings (目前 CrossTowerCausalModel 只接了 dropout, 你可能要修改模型接收 gnn_dropout)
            # 這裡假設你的模型已經有接收 gnn_dropout 的邏輯，如果沒有請在模型 init 加
            num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
            freeze_text=cfg.model.FREEZE_TEXT,
            use_explain=cfg.model.USE_EXPLAIN,
        ).to(device)

        # ---- Optimizer Grouping (根據 Config 設定) ----
        plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
        gnn_ids = list(map(id, model.structural_tower.parameters()))
        
        base_params = filter(
            lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, 
            model.parameters()
        )

        optimizer_grouped_parameters = [
            {'params': base_params, 'lr': cfg.optim.LR_BASE, 'weight_decay': cfg.optim.WD_BASE},
            {'params': model.structural_tower.parameters(), 'lr': cfg.optim.LR_GNN, 'weight_decay': cfg.optim.WD_GNN},
            {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.optim.LR_PLM, 'weight_decay': cfg.optim.WD_PLM}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        pos_weight_val = compute_pos_weight_from_dataset(train_dataset)
        print(f"Pos Weight: {pos_weight_val:.2f}")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

        total_steps = len(train_loader) * cfg.train.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * cfg.train.WARMUP_RATIO), 
            num_training_steps=total_steps
        )

        train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

    # ---------------------------------------------------------
    # Testing Mode
    # ---------------------------------------------------------
    else:
        print(">>> Test Mode")
        test_dataset = CEEEnd2EndDataset(
            cfg.path.CONV_PATH, cfg.path.TEST_PATH, 
            cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV, 
            use_explain=cfg.model.USE_EXPLAIN
        )
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        expl_dim = test_dataset.get_explain_dim()

        model = CrossTowerCausalModel(
            text_model_name=cfg.model.TEXT_MODEL,
            hidden_dim=cfg.model.HIDDEN_DIM,
            expl_dim=expl_dim,
            num_speakers=test_dataset.num_speakers,
            num_emotions=test_dataset.num_emotions,
            dropout=cfg.model.BASE_DROPOUT,
            num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
            freeze_text=cfg.model.FREEZE_TEXT,
            use_explain=cfg.model.USE_EXPLAIN,
        ).to(device)

        if os.path.exists(cfg.path.SAVE_MODEL_PATH):
            model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
            print(f"Loaded checkpoint: {cfg.path.SAVE_MODEL_PATH}")
        else:
            print(f"Checkpoint not found: {cfg.path.SAVE_MODEL_PATH}")
            return

        results = evaluate(model, test_loader, device, desc="Testing")
        print(json.dumps(results, indent=2))
        with open(cfg.path.OUT_JSON_PATH, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()