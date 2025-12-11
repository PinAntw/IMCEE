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
# # 1. Modular Configuration
# # ============================================================

# class PathConfig:
#     """ 路徑相關設定 """
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     EXP_DIR_GPT  = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")
    
#     # 輸出檔名標記
#     SAVE_SUFFIX = "cross_tower"
    
#     # 自動生成
#     CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     # 原本的 explain（edge-level）檔案
#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     # 新的 explanation space（GPT 版本）
#     ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
#     ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
#     ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
#     ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
#     ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
#     ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

#     @property
#     def SAVE_MODEL_PATH(self):
#         return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
#     @property
#     def OUT_JSON_PATH(self):
#         return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")


# class ModelConfig:
#     """ 模型結構超參數 """
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768    # 用於 Semantic Tower 和 Fusion 的統一維度

#     # TEXT_MODEL = "roberta-large"
#     # HIDDEN_DIM = 1024    # 用於 Semantic Tower 和 Fusion 的統一維度

#     FREEZE_TEXT = True
    
#     # Structural Tower (GNN) 設定
#     NUM_GNN_LAYERS = 3
#     GNN_DROPOUT    = 0.3
    
#     # Semantic Tower / General 設定
#     BASE_DROPOUT   = 0.2
    
#     # 功能開關
#     USE_EXPLAIN = False          # 舊的 edge-level explain
#     USE_EXPLAIN_SPACE = True     # 新的 explanation space 機制


# class TrainConfig:
#     """ 訓練流程超參數 """
#     RUN_MODE = "test"   # "train" or "test"
#     SEED = 42
    
#     EPOCHS = 30
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 5
#     WARMUP_RATIO = 0.1
    
#     # Loss Weight
#     POS_WEIGHT_MULT = 1
#     LAMBDA_EXPL = 0.8   # explanation alignment loss 的權重


# class OptimConfig:
#     """ 優化器與學習率設定 (不同模組不同 LR) """
#     # 基礎層 (MLP, Linear 等)
#     LR_BASE = 1e-4
#     WD_BASE = 1e-4
    
#     # PLM 層 (Text Encoder)
#     LR_PLM  = 3e-5
#     WD_PLM  = 0.01
    
#     # GNN 層 (Structural Tower)
#     LR_GNN  = 1e-3
#     WD_GNN  = 0.0

#     # # GPT推薦：roberta-large時候可以開啟
#     # # Backbone MLP / residual / predictor
#     # LR_BASE = 5e-5       # 從 1e-4 降一點
#     # WD_BASE = 1e-4

#     # # PLM (RoBERTa) — 真正解凍，LR 要小
#     # LR_PLM  = 5e-6       # 建議先用 5e-6，穩一點
#     # WD_PLM  = 0.01

#     # # GNN (RGCN) — 不要 1e-3 那麼暴力
#     # LR_GNN  = 3e-4       # 從 1e-3 降到 3e-4
#     # WD_GNN  = 0.0


# # 整合 Config
# class Config:
#     path = PathConfig()
#     model = ModelConfig()
#     train = TrainConfig()
#     optim = OptimConfig()

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
#         if label == 1:
#             pos_count += 1
#         else:
#             neg_count += 1
            
#     if pos_count == 0:
#         ratio = 1.0
#     else:
#         ratio = neg_count / pos_count
#     return ratio * cfg.train.POS_WEIGHT_MULT

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         # 評估時只用 logits，不需要 expl 對齊
#         out = model(batch)
#         if isinstance(out, tuple):
#             logits = out[0]
#         else:
#             logits = out

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
#     except Exception:
#         auc = 0.5

#     thr_candidates = np.concatenate([
#         np.linspace(0.01, 0.30, 30),
#         np.linspace(0.35, 0.90, 12),
#     ])

#     best_macro = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
    
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

#     print(
#         f"[Eval] Best MacroF1: {best_macro['macro_f1']:.4f} "
#         f"(Pos={best_macro['pos_f1']:.2f}, Neg={best_macro['neg_f1']:.2f}, "
#         f"Acc={best_macro['acc']:.4f}, AUC={best_macro['auc']:.4f})"
#     )
#     return best_macro

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0

#     for epoch in range(1, cfg.train.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()

#         total_loss, total_pairs = 0.0, 0
#         total_task_loss, total_expl_loss = 0.0, 0.0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)

#             # === 1) 前向傳播 ===
#             out = model(batch)

#             # 如果模型有回傳 (logits, z_student, z_teacher)
#             if (
#                 cfg.model.USE_EXPLAIN_SPACE
#                 and isinstance(out, tuple)
#                 and len(out) == 3
#             ):
#                 logits, z_student, z_teacher = out
#             else:
#                 logits = out
#                 z_student = None
#                 z_teacher = None

#             labels = batch.edge_label.float()

#             if logits.numel() == 0:
#                 continue

#             # === 2) 原本的 task loss ===
#             task_loss = loss_fn(logits, labels)

#             # === 3) explanation alignment loss (MSE on z) ===
#             if (
#                 cfg.model.USE_EXPLAIN_SPACE
#                 and z_student is not None
#                 and z_teacher is not None
#             ):
#                 expl_loss = torch.mean((z_student - z_teacher) ** 2)
#                 loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss
#             else:
#                 expl_loss = torch.tensor(0.0, device=device)
#                 loss = task_loss

#             # === 4) 反向傳播（含 gradient accumulation） ===
#             loss = loss / cfg.train.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.train.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             bs = int(labels.size(0))
#             total_loss += float(loss.item()) * cfg.train.ACCUM_STEPS * bs
#             total_task_loss += float(task_loss.item()) * bs
#             total_expl_loss += float(expl_loss.item()) * bs
#             total_pairs += bs

#         avg_loss = total_loss / max(1, total_pairs)
#         avg_task_loss = total_task_loss / max(1, total_pairs)
#         avg_expl_loss = total_expl_loss / max(1, total_pairs)

#         print(
#             f"Train loss = {avg_loss:.4f} "
#             f"(task={avg_task_loss:.4f}, expl={avg_expl_loss:.4f})"
#         )

#         val = evaluate(model, valid_loader, device, desc="Validating")
#         if val is None: 
#             print("Valid skipped.")
#             continue

#         if val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
#             print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
#         else:
#             no_improve += 1
#             if no_improve >= cfg.train.PATIENCE:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.train.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     # ---------------------------------------------------------
#     # Training Mode
#     # ---------------------------------------------------------
#     if cfg.train.RUN_MODE == "train":
#         print(">>> Loading Datasets...")
#         train_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH,
#             cfg.path.TRAIN_PATH,
#             cfg.path.EX_TRAIN_PT,
#             cfg.path.EX_TRAIN_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TRAIN_PT,
#             expl_space_tsv=cfg.path.ES_TRAIN_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )

#         valid_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH,
#             cfg.path.VALID_PATH,
#             cfg.path.EX_VALID_PT,
#             cfg.path.EX_VALID_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_VALID_PT,
#             expl_space_tsv=cfg.path.ES_VALID_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=cfg.train.BATCH_SIZE,
#             shuffle=True,
#             collate_fn=collate_fn,
#         )
#         valid_loader = DataLoader(
#             valid_dataset,
#             batch_size=cfg.train.BATCH_SIZE,
#             shuffle=False,
#             collate_fn=collate_fn,
#         )

#         expl_dim = train_dataset.get_explain_dim()
#         expl_space_dim = train_dataset.get_expl_space_dim()

#         print(">>> Initializing Model...")
#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=expl_space_dim,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         # ---- Optimizer Grouping ----
#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
        
#         base_params = filter(
#             lambda p: id(p) not in plm_ids and id(p) not in gnn_ids,
#             model.parameters(),
#         )

#         optimizer_grouped_parameters = [
#             {
#                 'params': base_params,
#                 'lr': cfg.optim.LR_BASE,
#                 'weight_decay': cfg.optim.WD_BASE,
#             },
#             {
#                 'params': model.structural_tower.parameters(),
#                 'lr': cfg.optim.LR_GNN,
#                 'weight_decay': cfg.optim.WD_GNN,
#             },
#             {
#                 'params': model.text_encoder.text_encoder.parameters(),
#                 'lr': cfg.optim.LR_PLM,
#                 'weight_decay': cfg.optim.WD_PLM,
#             },
#         ]

#         optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
#         pos_weight_val = compute_pos_weight_from_dataset(train_dataset)
#         print(f"Pos Weight: {pos_weight_val:.2f}")
#         loss_fn = torch.nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor([pos_weight_val], device=device)
#         )
#         # loss_fn = torch.nn.BCEWithLogitsLoss()


#         total_steps = len(train_loader) * cfg.train.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=int(total_steps * cfg.train.WARMUP_RATIO),
#             num_training_steps=total_steps,
#         )

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     # ---------------------------------------------------------
#     # Testing Mode
#     # ---------------------------------------------------------
#     else:
#         print(">>> Test Mode")
#         test_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH,
#             cfg.path.TEST_PATH,
#             cfg.path.EX_TEST_PT,
#             cfg.path.EX_TEST_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TEST_PT,
#             expl_space_tsv=cfg.path.ES_TEST_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=cfg.train.BATCH_SIZE,
#             shuffle=False,
#             collate_fn=collate_fn,
#         )
        
#         expl_dim = test_dataset.get_explain_dim()
#         expl_space_dim = test_dataset.get_expl_space_dim()

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=expl_dim,
#             num_speakers=test_dataset.num_speakers,
#             num_emotions=test_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=expl_space_dim,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         if os.path.exists(cfg.path.SAVE_MODEL_PATH):
#             model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
#             print(f"Loaded checkpoint: {cfg.path.SAVE_MODEL_PATH}")
#         else:
#             print(f"Checkpoint not found: {cfg.path.SAVE_MODEL_PATH}")
#             return

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.path.OUT_JSON_PATH, "w") as f:
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

# from modules.dataset import CEEEnd2EndDataset, End2EndCollate
# from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# # ============================================================
# # Configuration
# # ============================================================
# class PathConfig:
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     EXP_DIR_GPT  = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")
    
#     # 檔名後綴
#     SAVE_SUFFIX = "test"
    
#     CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
#     ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
#     ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
#     ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
#     ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
#     ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

#     @property
#     def SAVE_MODEL_PATH(self):
#         return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
#     @property
#     def OUT_JSON_PATH(self):
#         return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")

# class ModelConfig:
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768
#     FREEZE_TEXT = True
#     NUM_GNN_LAYERS = 3
#     GNN_DROPOUT    = 0.3
#     BASE_DROPOUT   = 0.2
#     USE_EXPLAIN = False
#     USE_EXPLAIN_SPACE = True

# class TrainConfig:
#     RUN_MODE = "train"   # "train" or "test"
#     SEED = 42
#     EPOCHS = 30
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 10
#     WARMUP_RATIO = 0.1
#     POS_WEIGHT_MULT = 1
#     LAMBDA_EXPL = 0.5   # Distillation weight

# class OptimConfig:
#     LR_BASE = 1e-4
#     WD_BASE = 1e-4
#     LR_PLM  = 3e-5
#     WD_PLM  = 0.01
#     LR_GNN  = 1e-3
#     WD_GNN  = 0.0

# class Config:
#     path = PathConfig()
#     model = ModelConfig()
#     train = TrainConfig()
#     optim = OptimConfig()

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
#         if d.edge_label.item() == 1:
#             pos_count += 1
#         else:
#             neg_count += 1
#     ratio = neg_count / pos_count if pos_count > 0 else 1.0
#     return ratio * cfg.train.POS_WEIGHT_MULT

# # ============================================================
# # Advanced Monitoring Function
# # ============================================================
# def monitor_model_internals(model, loader, device, desc="Monitoring"):
#     """
#     Computes deep metrics: Alpha/Beta, Residual Contribution Ratio, Orthogonality
#     """
#     model.eval()
    
#     avg_alpha = 0.0
#     avg_beta = 0.0
    
#     struct_ratio_sum = 0.0
#     sem_ratio_sum = 0.0
#     struct_orth_sum = 0.0
#     sem_orth_sum = 0.0
    
#     count = 0
    
#     cosine_sim = torch.nn.CosineSimilarity(dim=-1)

#     with torch.no_grad():
#         for batch in tqdm(loader, desc=desc):
#             batch = batch.to(device)
#             # Use return_aux=True to get internal tensors
#             logits, aux = model(batch, return_aux=True)
            
#             # 1. Scalar Weights
#             alpha = aux['alpha'].item()
#             beta = aux['beta'].item()
#             avg_alpha += alpha
#             avg_beta += beta
            
#             # 2. Extract Vectors
#             z_base = aux['z_base']          # [B, H]
#             delta_struct = aux['delta_struct'] # [B, H]
#             delta_sem = aux['delta_sem']       # [B, H]
            
#             # 3. Compute Norms
#             norm_base = torch.norm(z_base, dim=-1)
#             norm_struct = torch.norm(delta_struct, dim=-1)
#             norm_sem = torch.norm(delta_sem, dim=-1)
            
#             # 4. Contribution Ratios: ||alpha * delta|| / ||base||
#             r_struct = (alpha * norm_struct) / (norm_base + 1e-6)
#             r_sem = (beta * norm_sem) / (norm_base + 1e-6)
            
#             struct_ratio_sum += r_struct.mean().item()
#             sem_ratio_sum += r_sem.mean().item()
            
#             # 5. Orthogonality: Cosine(delta, base)
#             orth_struct = cosine_sim(delta_struct, z_base)
#             orth_sem = cosine_sim(delta_sem, z_base)
            
#             struct_orth_sum += orth_struct.mean().item()
#             sem_orth_sum += orth_sem.mean().item()
            
#             count += 1
#             if count >= 50: break # Only monitor first 50 batches to save time

#     print(f"\n[Monitor] Stats over {count} batches:")
#     print(f"  > Alpha (Structure): {avg_alpha/count:.4f}")
#     print(f"  > Beta  (Semantic):  {avg_beta/count:.4f}")
#     print(f"  > Struct Contrib:    {struct_ratio_sum/count:.4f} (Ideal: 0.1~0.5)")
#     print(f"  > Sem Contrib:       {sem_ratio_sum/count:.4f}")
#     print(f"  > Struct Orthog:     {struct_orth_sum/count:.4f} (Ideal: near 0)")
#     print(f"  > Sem Orthog:        {sem_orth_sum/count:.4f} (Ideal: near 0)")
    
#     return {
#         "alpha": avg_alpha/count,
#         "beta": avg_beta/count
#     }

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         out = model(batch)
#         if isinstance(out, tuple):
#             logits = out[0]
#         else:
#             logits = out

#         labels = batch.edge_label.float()
#         if logits.numel() == 0: continue

#         all_logits.append(logits.detach().cpu())
#         all_labels.append(labels.detach().cpu())

#     if not all_logits: return None

#     all_logits = torch.cat(all_logits).numpy()
#     all_labels = torch.cat(all_labels).numpy()
#     all_probs = sigmoid(all_logits)

#     try:
#         auc = roc_auc_score(all_labels, all_probs)
#     except:
#         auc = 0.5

#     best_macro = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
#     thr_candidates = np.concatenate([np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)])

#     for thr in thr_candidates:
#         preds = (all_probs >= thr).astype(int)
#         pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
#         neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
#         macro = (pos_f1 + neg_f1) / 2.0
#         acc = accuracy_score(all_labels, preds)

#         if macro > best_macro["macro_f1"]:
#             best_macro = {
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "auc": float(auc),
#                 "thr": float(thr),
#             }

#     # [修正] 這裡把詳細數據 print 出來！
#     print(
#         f"[Eval] Best MacroF1: {best_macro['macro_f1']:.4f} "
#         f"(Pos={best_macro['pos_f1']:.2f}, Neg={best_macro['neg_f1']:.2f}, "
#         f"Acc={best_macro['acc']:.4f}, AUC={best_macro['auc']:.4f})"
#     )
#     return best_macro

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0
    
#     # Cosine Loss for Distillation (Optional experiment)
#     # cosine_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)

#     for epoch in range(1, cfg.train.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()

#         total_loss, total_task, total_expl = 0.0, 0.0, 0.0
#         total_pairs = 0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)
            
#             out = model(batch)
#             if isinstance(out, tuple) and len(out) == 3:
#                 logits, z_student, z_teacher = out
#             else:
#                 logits, z_student, z_teacher = out, None, None

#             labels = batch.edge_label.float()
#             if logits.numel() == 0: continue

#             # Loss
#             task_loss = loss_fn(logits, labels)
            
#             if cfg.model.USE_EXPLAIN_SPACE and z_student is not None and z_teacher is not None:
#                 # Default: MSE
#                 expl_loss = torch.mean((z_student - z_teacher) ** 2)
#             else:
#                 expl_loss = torch.tensor(0.0, device=device)

#             loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss

#             loss = loss / cfg.train.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.train.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             bs = int(labels.size(0))
#             total_loss += loss.item() * cfg.train.ACCUM_STEPS * bs
#             total_task += task_loss.item() * bs
#             total_expl += expl_loss.item() * bs
#             total_pairs += bs

#         print(f"Train loss={total_loss/total_pairs:.4f} (task={total_task/total_pairs:.4f}, expl={total_expl/total_pairs:.2e})")

#         # --- Evaluate ---
#         val = evaluate(model, valid_loader, device)
        
#         # --- Monitor (Show Alpha/Beta etc) ---
#         monitor_model_internals(model, valid_loader, device)

#         # --- Update Best Model Logic ---
#         is_best = False
#         if val and val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             is_best = True
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
#             print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
#         else:
#             no_improve += 1
        
#         # [新增] 顯示目前為止的歷史最佳成績
#         print(f"Current Epoch: {val['macro_f1']:.4f} | >>> Global Best MacroF1: {best_macro:.4f} <<<")

#         if not is_best and no_improve >= cfg.train.PATIENCE:
#             print(f"Early stopping at epoch {epoch} (No improvement for {cfg.train.PATIENCE} epochs)")
#             break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.train.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     if cfg.train.RUN_MODE == "train":
#         train_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TRAIN_PATH,
#             cfg.path.EX_TRAIN_PT, cfg.path.EX_TRAIN_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TRAIN_PT, expl_space_tsv=cfg.path.ES_TRAIN_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         valid_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.VALID_PATH,
#             cfg.path.EX_VALID_PT, cfg.path.EX_VALID_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_VALID_PT, expl_space_tsv=cfg.path.ES_VALID_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         train_loader = DataLoader(train_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#         valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=train_dataset.get_explain_dim(),
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=train_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         # Optimizer Grouping
#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
#         base_params = filter(lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, model.parameters())

#         optimizer = torch.optim.AdamW([
#             {'params': base_params, 'lr': cfg.optim.LR_BASE, 'weight_decay': cfg.optim.WD_BASE},
#             {'params': model.structural_tower.parameters(), 'lr': cfg.optim.LR_GNN, 'weight_decay': cfg.optim.WD_GNN},
#             {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.optim.LR_PLM, 'weight_decay': cfg.optim.WD_PLM},
#         ])

#         pos_weight = compute_pos_weight_from_dataset(train_dataset)
#         loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        
#         total_steps = len(train_loader) * cfg.train.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.train.WARMUP_RATIO), total_steps)

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     else: # TEST
#         test_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TEST_PATH,
#             cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TEST_PT, expl_space_tsv=cfg.path.ES_TEST_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         test_loader = DataLoader(test_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=test_dataset.get_explain_dim(),
#             num_speakers=test_dataset.num_speakers,
#             num_emotions=test_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=test_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         if os.path.exists(cfg.path.SAVE_MODEL_PATH):
#             model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
#             print(f"Loaded: {cfg.path.SAVE_MODEL_PATH}")
        
#         # Monitor on Test Set
#         monitor_model_internals(model, test_loader, device, desc="Monitoring Test")

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.path.OUT_JSON_PATH, "w") as f:
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

# from modules.dataset import CEEEnd2EndDataset, End2EndCollate
# from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# # ============================================================
# # Configuration (維持不變)
# # ============================================================
# class PathConfig:
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     EXP_DIR_GPT  = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")

#     SAVE_SUFFIX = "cross_tower_final_subtraction_monitor"

#     CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
#     ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
#     ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
#     ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
#     ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
#     ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

#     @property
#     def SAVE_MODEL_PATH(self):
#         return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")

#     @property
#     def OUT_JSON_PATH(self):
#         return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")


# class ModelConfig:
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768
#     FREEZE_TEXT = True
#     NUM_GNN_LAYERS = 3
#     GNN_DROPOUT    = 0.3
#     BASE_DROPOUT   = 0.2
#     USE_EXPLAIN = False
#     USE_EXPLAIN_SPACE = True


# class TrainConfig:
#     RUN_MODE = "test"   # "train" or "test"
#     SEED = 42
#     EPOCHS = 30
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 5
#     WARMUP_RATIO = 0.1
#     POS_WEIGHT_MULT = 1
#     LAMBDA_EXPL = 50   # Distillation weight


# class OptimConfig:
#     LR_BASE = 1e-4
#     WD_BASE = 1e-4
#     LR_PLM  = 3e-5
#     WD_PLM  = 0.01
#     LR_GNN  = 1e-3
#     WD_GNN  = 0.0


# class Config:
#     path = PathConfig()
#     model = ModelConfig()
#     train = TrainConfig()
#     optim = OptimConfig()


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
#         if d.edge_label.item() == 1:
#             pos_count += 1
#         else:
#             neg_count += 1
#     ratio = neg_count / pos_count if pos_count > 0 else 1.0
#     return ratio * cfg.train.POS_WEIGHT_MULT

# # ============================================================
# # Diagnostic Suite (NEW: Replaces monitor_model_internals)
# # ============================================================
# def run_diagnostic_suite(model, loader, device, desc="Diagnostics"):
#     """
#     執行三大診斷：
#     A) Separability: 各個組件單獨過 Predictor 的 AUC
#     B) Norm Distribution: 檢查 Semantic 是否塌縮 (Mean, P95)
#     C) Conflict: 檢查 Structure 與 Semantic 是否反向抵銷 (Cosine)
#     """
#     model.eval()

#     # Collectors for A (Separability)
#     # 我們收集 logits 而不是 prob，方便後續處理
#     logits_base = []
#     logits_struct_only = []
#     logits_sem_only = []
#     logits_base_sem = [] # 看 base + sem 的效果
#     all_labels = []

#     # Collectors for B (Norms)
#     norms_sem = []
#     norms_struct = []
#     norms_base = []

#     # Collectors for C (Conflict)
#     cos_struct_sem = []
    
#     # Monitor Scalars
#     avg_alpha = 0.0
#     avg_beta = 0.0
#     count = 0

#     with torch.no_grad():
#         for batch in tqdm(loader, desc=desc):
#             batch = batch.to(device)
#             # 取得 aux
#             _, aux = model(batch, return_aux=True)
#             labels = batch.edge_label.float().cpu().numpy()
            
#             # --- Extract Internals ---
#             z_base = aux["z_base"]             # [B, H]
#             delta_struct = aux["delta_struct"] # [B, H]
#             delta_sem = aux["delta_sem"]       # [B, H]
            
#             alpha = aux["alpha"].item()
#             beta = aux["beta"].item()
#             avg_alpha += alpha
#             avg_beta += beta

#             # --- Diagnostic A: Forward through Predictor ---
#             # Predictor 期望輸入 [B, H]
#             # 1. Base Only
#             l_base = model.predictor(z_base).squeeze(-1).cpu().numpy()
            
#             # 2. Structure Only (純殘差)
#             l_struct = model.predictor(delta_struct).squeeze(-1).cpu().numpy()
            
#             # 3. Semantic Only (純殘差) - 測試你的假設 A
#             l_sem = model.predictor(delta_sem).squeeze(-1).cpu().numpy()

#             # 4. Base + Semantic (增量測試) - 看看加了有沒有比純 Base 好
#             # 注意：這裡不乘 beta，直接看原始能力
#             l_base_sem = model.predictor(z_base + delta_sem).squeeze(-1).cpu().numpy()

#             logits_base.extend(l_base)
#             logits_struct_only.extend(l_struct)
#             logits_sem_only.extend(l_sem)
#             logits_base_sem.extend(l_base_sem)
#             all_labels.extend(labels)

#             # --- Diagnostic B: Norms ---
#             n_sem = torch.norm(delta_sem, dim=-1)
#             n_struct = torch.norm(delta_struct, dim=-1)
#             n_base = torch.norm(z_base, dim=-1)
            
#             norms_sem.extend(n_sem.cpu().numpy())
#             norms_struct.extend(n_struct.cpu().numpy())
#             norms_base.extend(n_base.cpu().numpy())

#             # --- Diagnostic C: Cosine Similarity ---
#             # 檢查兩個殘差是否在打架
#             cos = F.cosine_similarity(delta_struct, delta_sem, dim=-1)
#             cos_struct_sem.extend(cos.cpu().numpy())

#             count += 1

#     # --- Compute Metrics ---
    
#     # Helper for AUC
#     def calc_auc(logits, labels):
#         try:
#             probs = sigmoid(np.array(logits))
#             return roc_auc_score(labels, probs)
#         except:
#             return 0.5

#     # A) AUC Results
#     auc_base = calc_auc(logits_base, all_labels)
#     auc_struct = calc_auc(logits_struct_only, all_labels)
#     auc_sem = calc_auc(logits_sem_only, all_labels)
#     auc_base_sem = calc_auc(logits_base_sem, all_labels)

#     # B) Norm Stats
#     norms_sem = np.array(norms_sem)
#     norms_struct = np.array(norms_struct)
    
#     sem_mean = np.mean(norms_sem)
#     sem_p95 = np.percentile(norms_sem, 95)
    
#     struct_mean = np.mean(norms_struct)
    
#     # C) Conflict Stats
#     cos_mean = np.mean(cos_struct_sem)

#     # Print Report
#     print("\n" + "="*40)
#     print("   DIAGNOSTIC REPORT")
#     print("="*40)
#     print(f"Scalars: Alpha={avg_alpha/count:.4f}, Beta={avg_beta/count:.4f}")
    
#     print("\n[A] Separability (AUC):")
#     print(f"  > Base Only:         {auc_base:.4f}")
#     print(f"  > Struct Only:       {auc_struct:.4f} (Residual Power)")
#     print(f"  > Sem Only:          {auc_sem:.4f} (Is it separable?)")
#     print(f"  > Base + Sem:        {auc_base_sem:.4f} (Does it help Base?)")
    
#     print("\n[B] Norm Distribution:")
#     print(f"  > ||Delta_Struct||:  Mean={struct_mean:.4f}")
#     print(f"  > ||Delta_Sem||:     Mean={sem_mean:.4f}, P95={sem_p95:.4f}")
#     if sem_mean < 0.1:
#         print("    -> WARNING: Semantic Norm is collapsed (Vanishing Gradient?)")
    
#     print("\n[C] Interaction:")
#     print(f"  > Cos(Struct, Sem):  {cos_mean:.4f}")
#     if cos_mean < -0.3:
#         print("    -> WARNING: Structure and Semantic are fighting (Canceling out)")
    
#     print("="*40 + "\n")

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         out = model(batch)
#         if isinstance(out, tuple):
#             logits = out[0]
#         else:
#             logits = out

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

#     best_macro = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
#     thr_candidates = np.concatenate([np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)])

#     for thr in thr_candidates:
#         preds = (all_probs >= thr).astype(int)
#         pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
#         neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
#         macro = (pos_f1 + neg_f1) / 2.0
#         acc = accuracy_score(all_labels, preds)

#         if macro > best_macro["macro_f1"]:
#             best_macro = {
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "auc": float(auc),
#                 "thr": float(thr),
#             }

#     print(
#         f"[Eval] Best MacroF1: {best_macro['macro_f1']:.4f} "
#         f"(Pos={best_macro['pos_f1']:.2f}, Neg={best_macro['neg_f1']:.2f}, "
#         f"Acc={best_macro['acc']:.4f}, AUC={best_macro['auc']:.4f})"
#     )
#     return best_macro

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0

#     for epoch in range(1, cfg.train.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()

#         total_loss, total_task, total_expl = 0.0, 0.0, 0.0
#         total_pairs = 0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)

#             out = model(batch)
#             if isinstance(out, tuple) and len(out) == 3:
#                 logits, z_student, z_teacher = out
#             else:
#                 logits, z_student, z_teacher = out, None, None

#             labels = batch.edge_label.float()
#             if logits.numel() == 0:
#                 continue

#             task_loss = loss_fn(logits, labels)

#             if cfg.model.USE_EXPLAIN_SPACE and z_student is not None and z_teacher is not None:
#                 # Distillation loss: Cosine distance
#                 cos = F.cosine_similarity(z_student, z_teacher, dim=-1)  # [B]
#                 expl_loss = torch.mean(1.0 - cos)
#             else:
#                 expl_loss = torch.tensor(0.0, device=device)

#             loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss

#             loss = loss / cfg.train.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.train.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             bs = int(labels.size(0))
#             total_loss += loss.item() * cfg.train.ACCUM_STEPS * bs
#             total_task += task_loss.item() * bs
#             total_expl += expl_loss.item() * bs
#             total_pairs += bs

#         print(
#             f"Train loss={total_loss/total_pairs:.4f} "
#             f"(task={total_task/total_pairs:.4f}, expl={total_expl/total_pairs:.2e})"
#         )

#         val = evaluate(model, valid_loader, device)

#         # [修改] 使用新的診斷套件
#         run_diagnostic_suite(model, valid_loader, device, desc="Diagnostics")

#         is_best = False
#         if val and val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             is_best = True
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
#             print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
#         else:
#             no_improve += 1

#         print(f"Current Epoch: {val['macro_f1']:.4f} | >>> Global Best MacroF1: {best_macro:.4f} <<<")

#         if (not is_best) and no_improve >= cfg.train.PATIENCE:
#             print(f"Early stopping at epoch {epoch} (No improvement for {cfg.train.PATIENCE} epochs)")
#             break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.train.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     if cfg.train.RUN_MODE == "train":
#         train_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TRAIN_PATH,
#             cfg.path.EX_TRAIN_PT, cfg.path.EX_TRAIN_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TRAIN_PT, expl_space_tsv=cfg.path.ES_TRAIN_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         valid_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.VALID_PATH,
#             cfg.path.EX_VALID_PT, cfg.path.EX_VALID_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_VALID_PT, expl_space_tsv=cfg.path.ES_VALID_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )

#         train_loader = DataLoader(
#             train_dataset, batch_size=cfg.train.BATCH_SIZE,
#             shuffle=True, collate_fn=collate_fn
#         )
#         valid_loader = DataLoader(
#             valid_dataset, batch_size=cfg.train.BATCH_SIZE,
#             shuffle=False, collate_fn=collate_fn
#         )

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=train_dataset.get_explain_dim(),
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=train_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         # 抓出 Alpha/Beta 參數
#         scalar_params = [model.alpha_param, model.beta_param]
#         scalar_ids = list(map(id, scalar_params))

#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
        
#         # Base params 排除掉 scalar_ids
#         base_params = filter(
#             lambda p: id(p) not in plm_ids and id(p) not in gnn_ids and id(p) not in scalar_ids, 
#             model.parameters()
#         )

#         optimizer = torch.optim.AdamW([
#             {'params': base_params, 'lr': cfg.optim.LR_BASE, 'weight_decay': cfg.optim.WD_BASE},
#             {'params': model.structural_tower.parameters(), 'lr': cfg.optim.LR_GNN, 'weight_decay': cfg.optim.WD_GNN},
#             {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.optim.LR_PLM, 'weight_decay': cfg.optim.WD_PLM},
#             {'params': scalar_params, 'lr': 1e-2, 'weight_decay': 0.0},
#         ])

#         pos_weight = compute_pos_weight_from_dataset(train_dataset)
#         loss_fn = torch.nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor([pos_weight], device=device)
#         )

#         total_steps = len(train_loader) * cfg.train.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             int(total_steps * cfg.train.WARMUP_RATIO),
#             total_steps
#         )

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     else:
#         test_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TEST_PATH,
#             cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TEST_PT, expl_space_tsv=cfg.path.ES_TEST_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         test_loader = DataLoader(
#             test_dataset, batch_size=cfg.train.BATCH_SIZE,
#             shuffle=False, collate_fn=collate_fn
#         )

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=test_dataset.get_explain_dim(),
#             num_speakers=test_dataset.num_speakers,
#             num_emotions=test_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=test_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         if os.path.exists(cfg.path.SAVE_MODEL_PATH):
#             model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
#             print(f"Loaded: {cfg.path.SAVE_MODEL_PATH}")

#         # [修改] 使用新的診斷套件
#         run_diagnostic_suite(model, test_loader, device, desc="Diagnostics Test")

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.path.OUT_JSON_PATH, "w") as f:
#             json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Research/IMCEE/run_cross_tower_gated.py

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

# from modules.dataset import CEEEnd2EndDataset, End2EndCollate
# from modules.crossTower.cross_tower_model import CrossTowerCausalModel

# # ============================================================
# # Configuration
# # ============================================================
# class PathConfig:
#     BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
#     DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")
#     EXP_DIR  = os.path.join(BASE_DIR, "outputs/Vexplain")
#     EXP_DIR_GPT  = os.path.join(BASE_DIR, "outputs/Vexplain_gpt")
#     CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")
    
#     # 檔名標記 (Gated Version)
#     SAVE_SUFFIX = "gated_sem_v2"
    
#     CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")
#     TRAIN_PATH = os.path.join(DATA_DIR, "pairs_train.jsonl")
#     VALID_PATH = os.path.join(DATA_DIR, "pairs_valid.jsonl")
#     TEST_PATH  = os.path.join(DATA_DIR, "pairs_test.jsonl")

#     EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
#     EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
#     EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
#     EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
#     EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
#     EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

#     ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
#     ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
#     ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
#     ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
#     ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
#     ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

#     @property
#     def SAVE_MODEL_PATH(self):
#         return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
#     @property
#     def OUT_JSON_PATH(self):
#         return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")

# class ModelConfig:
#     TEXT_MODEL = "roberta-base"
#     HIDDEN_DIM = 768
#     FREEZE_TEXT = True
#     NUM_GNN_LAYERS = 3
#     GNN_DROPOUT    = 0.3
#     BASE_DROPOUT   = 0.2
#     USE_EXPLAIN = False      # edge-level
#     USE_EXPLAIN_SPACE = True # teacher space (必須開)

# class TrainConfig:
#     RUN_MODE = "test"   # "train" or "test"
#     SEED = 42
#     EPOCHS = 30
#     BATCH_SIZE = 4
#     ACCUM_STEPS = 1
#     PATIENCE = 5
#     WARMUP_RATIO = 0.1
#     POS_WEIGHT_MULT = 1
    
#     # 蒸餾權重 (因 Cosine Loss 數值較大，設 1.0~2.0 即可)
#     LAMBDA_EXPL = 1.0   

# class OptimConfig:
#     LR_BASE = 1e-4
#     WD_BASE = 1e-4
#     LR_PLM  = 3e-5
#     WD_PLM  = 0.01
#     LR_GNN  = 1e-3
#     WD_GNN  = 0.0

# class Config:
#     path = PathConfig()
#     model = ModelConfig()
#     train = TrainConfig()
#     optim = OptimConfig()

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
#         if d.edge_label.item() == 1:
#             pos_count += 1
#         else:
#             neg_count += 1
#     ratio = neg_count / pos_count if pos_count > 0 else 1.0
#     return ratio * cfg.train.POS_WEIGHT_MULT

# # ============================================================
# # Monitor Function (Gated Version)
# # ============================================================
# def monitor_gated_stats(model, loader, device, desc="Monitoring"):
#     """
#     監測 Gate 的開啟程度 (Openness) 與 Student 對齊程度
#     """
#     model.eval()
    
#     avg_gate_val = 0.0
#     avg_align_sim = 0.0
#     count = 0
    
#     cosine_sim = torch.nn.CosineSimilarity(dim=-1)

#     with torch.no_grad():
#         for batch in tqdm(loader, desc=desc):
#             batch = batch.to(device)
#             # 使用 return_aux 獲取內部變數
#             _, aux = model(batch, return_aux=True)
            
#             # 1. Gate Openness
#             gate = aux.get("gate", None)
#             if gate is not None:
#                 # gate: [B, H] -> mean scalar
#                 avg_gate_val += gate.mean().item()
            
#             # 2. Student-Teacher Alignment
#             z_stud = aux.get("z_student", None)
#             z_teach = aux.get("z_teacher", None)
            
#             if z_stud is not None and z_teach is not None:
#                 sim = cosine_sim(z_stud, z_teach).mean().item()
#                 avg_align_sim += sim
                
#             count += 1
#             if count >= 50: break # 只看前 50 個 batch

#     print(f"\n[Gated Monitor] Stats over {count} batches:")
#     print(f"  > Gate Openness: {avg_gate_val/count:.4f} (0=Closed, 1=Open)")
#     print(f"  > S-T Alignment: {avg_align_sim/count:.4f} (Cosine Sim, -1~1)")
#     print("-" * 40)

# # ============================================================
# # Evaluate
# # ============================================================
# @torch.no_grad()
# def evaluate(model, loader, device, desc="Evaluating"):
#     model.eval()
#     all_logits, all_labels = [], []

#     for batch in tqdm(loader, desc=desc):
#         batch = batch.to(device)
#         out = model(batch) # return logits only
        
#         if isinstance(out, tuple): 
#             logits = out[0]
#         else: 
#             logits = out

#         labels = batch.edge_label.float()
#         if logits.numel() == 0: continue

#         all_logits.append(logits.detach().cpu())
#         all_labels.append(labels.detach().cpu())

#     if not all_logits: return None

#     all_logits = torch.cat(all_logits).numpy()
#     all_labels = torch.cat(all_labels).numpy()
#     all_probs = sigmoid(all_logits)

#     try:
#         auc = roc_auc_score(all_labels, all_probs)
#     except:
#         auc = 0.5

#     best_res = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
#     thr_candidates = np.concatenate([np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)])

#     for thr in thr_candidates:
#         preds = (all_probs >= thr).astype(int)
#         pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
#         neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
#         macro = (pos_f1 + neg_f1) / 2.0
#         acc = accuracy_score(all_labels, preds)

#         if macro > best_res["macro_f1"]:
#             best_res = {
#                 "macro_f1": float(macro),
#                 "pos_f1": float(pos_f1),
#                 "neg_f1": float(neg_f1),
#                 "acc": float(acc),
#                 "auc": float(auc),
#                 "thr": float(thr),
#             }

#     print(
#         f"[Eval] Best MacroF1: {best_res['macro_f1']:.4f} "
#         f"(Pos={best_res['pos_f1']:.2f}, Neg={best_res['neg_f1']:.2f}, "
#         f"Acc={best_res['acc']:.4f}, AUC={best_res['auc']:.4f})"
#     )
#     return best_res

# # ============================================================
# # Train
# # ============================================================
# def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
#     os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
#     best_macro = 0.0
#     no_improve = 0
    
#     # 用 Cosine Loss 比較適合高維對齊
#     distill_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)

#     for epoch in range(1, cfg.train.EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
#         model.train()
#         optimizer.zero_grad()

#         total_loss, total_task, total_expl = 0.0, 0.0, 0.0
#         total_pairs = 0

#         for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
#             batch = batch.to(device)
            
#             # Forward (train mode returns tuple)
#             out = model(batch)
#             if isinstance(out, tuple) and len(out) == 3:
#                 logits, z_stud, z_teach = out
#             else:
#                 logits, z_stud, z_teach = out, None, None

#             labels = batch.edge_label.float()
#             if logits.numel() == 0: continue

#             # 1. Task Loss
#             task_loss = loss_fn(logits, labels)
            
#             # 2. Alignment Loss (Cosine)
#             if cfg.model.USE_EXPLAIN_SPACE and z_stud is not None and z_teach is not None:
#                 # target=1 means make them similar
#                 target = torch.ones(z_stud.size(0)).to(device)
#                 expl_loss = distill_loss_fn(z_stud, z_teach, target)
#             else:
#                 expl_loss = torch.tensor(0.0, device=device)

#             loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss

#             loss = loss / cfg.train.ACCUM_STEPS
#             loss.backward()

#             if (step + 1) % cfg.train.ACCUM_STEPS == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             bs = int(labels.size(0))
#             total_loss += loss.item() * cfg.train.ACCUM_STEPS * bs
#             total_task += task_loss.item() * bs
#             total_expl += expl_loss.item() * bs
#             total_pairs += bs

#         print(f"Train loss={total_loss/total_pairs:.4f} (task={total_task/total_pairs:.4f}, expl={total_expl/total_pairs:.4f})")

#         # --- Evaluate ---
#         val = evaluate(model, valid_loader, device)
        
#         # --- Monitor Gate & Alignment ---
#         monitor_gated_stats(model, valid_loader, device)

#         is_best = False
#         if val and val["macro_f1"] > best_macro:
#             best_macro = val["macro_f1"]
#             is_best = True
#             no_improve = 0
#             torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
#             print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
#         else:
#             no_improve += 1
        
#         print(f"Current Epoch: {val['macro_f1']:.4f} | >>> Global Best MacroF1: {best_macro:.4f} <<<")

#         if not is_best and no_improve >= cfg.train.PATIENCE:
#             print("Early stopping.")
#             break

# # ============================================================
# # Main
# # ============================================================
# def main():
#     set_seed(cfg.train.SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
#     collate_fn = End2EndCollate(tokenizer)

#     if cfg.train.RUN_MODE == "train":
#         train_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TRAIN_PATH,
#             cfg.path.EX_TRAIN_PT, cfg.path.EX_TRAIN_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TRAIN_PT, expl_space_tsv=cfg.path.ES_TRAIN_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         valid_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.VALID_PATH,
#             cfg.path.EX_VALID_PT, cfg.path.EX_VALID_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_VALID_PT, expl_space_tsv=cfg.path.ES_VALID_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         train_loader = DataLoader(train_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#         valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=train_dataset.get_explain_dim(),
#             num_speakers=train_dataset.num_speakers,
#             num_emotions=train_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=train_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         # Optimizer Grouping (這裡不需要特殊 LR，因為是 MLP Gating)
#         plm_ids = list(map(id, model.text_encoder.text_encoder.parameters()))
#         gnn_ids = list(map(id, model.structural_tower.parameters()))
#         base_params = filter(lambda p: id(p) not in plm_ids and id(p) not in gnn_ids, model.parameters())

#         optimizer = torch.optim.AdamW([
#             {'params': base_params, 'lr': cfg.optim.LR_BASE, 'weight_decay': cfg.optim.WD_BASE},
#             {'params': model.structural_tower.parameters(), 'lr': cfg.optim.LR_GNN, 'weight_decay': cfg.optim.WD_GNN},
#             {'params': model.text_encoder.text_encoder.parameters(), 'lr': cfg.optim.LR_PLM, 'weight_decay': cfg.optim.WD_PLM},
#         ])

#         pos_weight = compute_pos_weight_from_dataset(train_dataset)
#         loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        
#         total_steps = len(train_loader) * cfg.train.EPOCHS
#         scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.train.WARMUP_RATIO), total_steps)

#         train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

#     else: # TEST
#         test_dataset = CEEEnd2EndDataset(
#             cfg.path.CONV_PATH, cfg.path.TEST_PATH,
#             cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_pt=cfg.path.ES_TEST_PT, expl_space_tsv=cfg.path.ES_TEST_TSV,
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         )
#         test_loader = DataLoader(test_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#         model = CrossTowerCausalModel(
#             text_model_name=cfg.model.TEXT_MODEL,
#             hidden_dim=cfg.model.HIDDEN_DIM,
#             expl_dim=test_dataset.get_explain_dim(),
#             num_speakers=test_dataset.num_speakers,
#             num_emotions=test_dataset.num_emotions,
#             dropout=cfg.model.BASE_DROPOUT,
#             gnn_dropout=cfg.model.GNN_DROPOUT,
#             num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
#             freeze_text=cfg.model.FREEZE_TEXT,
#             use_explain=cfg.model.USE_EXPLAIN,
#             expl_space_dim=test_dataset.get_expl_space_dim(),
#             use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
#         ).to(device)

#         if os.path.exists(cfg.path.SAVE_MODEL_PATH):
#             model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
#             print(f"Loaded: {cfg.path.SAVE_MODEL_PATH}")
        
#         monitor_gated_stats(model, test_loader, device, desc="Monitoring Test")

#         results = evaluate(model, test_loader, device, desc="Testing")
#         print(json.dumps(results, indent=2))
#         with open(cfg.path.OUT_JSON_PATH, "w") as f:
#             json.dump(results, f, indent=2)

# if __name__ == "__main__":
#     main()

#跟上面一樣 test加了點東西而已 focal loss壞了就換這個
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/run_cross_tower_gated.py

import os
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")
    
    # Filename tag (Gated Version)
    SAVE_SUFFIX = "cross_tower_gated_layer8"
    
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
        return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")
    
    @property
    def OUT_JSON_PATH(self):
        return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")

class ModelConfig:
    TEXT_MODEL = "roberta-base"
    HIDDEN_DIM = 768
    # TEXT_MODEL = "roberta-large"
    # HIDDEN_DIM = 1024
    FREEZE_TEXT = True
    NUM_GNN_LAYERS = 8
    GNN_DROPOUT    = 0.4
    BASE_DROPOUT   = 0.2
    USE_EXPLAIN = False      # edge-level
    USE_EXPLAIN_SPACE = True # teacher space (Must be enabled)

class TrainConfig:
    RUN_MODE = "test"   # "train" or "test"
    SEED = 42
    EPOCHS = 30
    BATCH_SIZE = 4
    ACCUM_STEPS = 1
    PATIENCE = 5
    WARMUP_RATIO = 0.1
    POS_WEIGHT_MULT = 1
    
    # Distillation weight
    LAMBDA_EXPL = 1.0   

class OptimConfig:
    LR_BASE = 1e-4
    WD_BASE = 1e-4
    LR_PLM  = 3e-6
    WD_PLM  = 0.01
    LR_GNN  = 1e-3
    WD_GNN  = 0.0

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
        if d.edge_label.item() == 1:
            pos_count += 1
        else:
            neg_count += 1
    ratio = neg_count / pos_count if pos_count > 0 else 1.0
    return ratio * cfg.train.POS_WEIGHT_MULT

def calculate_pair_distance(batch):
    """
    Calculates the true utterance distance for each Cause-Target pair in the batch.
    batch.target_node_indices: [B, 2] -> [cause_node_idx, target_node_idx]
    batch.edge_index: [2, E]
    """
    batch_dists = []
    
    cause_nodes = batch.target_node_indices[:, 0]
    target_nodes = batch.target_node_indices[:, 1]
    
    edge_src = batch.edge_index[0]
    edge_tgt = batch.edge_index[1]
    
    for i in range(len(cause_nodes)):
        c_node = cause_nodes[i].item()
        t_node = target_nodes[i].item()
        
        # 1. Find the real Utterance Index connected to Cause Node
        # Dataset graph logic: CauseNode -> CauseUtt (Type 0)
        # Find edge_tgt where edge_src == c_node
        mask_c = (edge_src == c_node)
        if mask_c.any():
            c_utt_idx = edge_tgt[mask_c].min().item()
        else:
            c_utt_idx = 0
            
        # 2. Find the real Utterance Index connected to Target Node
        mask_t = (edge_src == t_node)
        if mask_t.any():
            t_utt_idx = edge_tgt[mask_t].min().item()
        else:
            t_utt_idx = 0
            
        dist = abs(t_utt_idx - c_utt_idx)
        batch_dists.append(dist)
        
    return np.array(batch_dists)

# ============================================================
# Analysis Function (Updated with Pos/Neg F1)
# ============================================================
def analyze_distance_performance(preds, labels, dists):
    """
    Detailed accuracy analysis for different distances
    Includes Pos F1 and Neg F1 specifically.
    """
    dists = np.array(dists)
    preds = np.array(preds)
    labels = np.array(labels)
    
    # 表格拉寬一點以容納更多欄位
    print("\n" + "="*80)
    print("   MODEL PREDICTION - DISTANCE BREAKDOWN")
    print("="*80)
    # Header 加上 PosF1 和 NegF1
    print(f"{'Dist':<5} | {'Count':<6} | {'PosRate':<8} | {'Acc':<8} | {'PosF1':<8} | {'NegF1':<8} | {'Recall':<8}")
    print("-" * 80)
    
    targets = [0, 1, 2, 3, 4,5,6,7,8]
    
    for d in targets:
        mask = (dists == d)
        sub_preds = preds[mask]
        sub_labels = labels[mask]
        
        count = len(sub_labels)
        
        if count == 0:
            print(f"{d:<5} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            continue
        
        pos_rate = np.mean(sub_labels)
        acc = accuracy_score(sub_labels, sub_preds)
        
        # [新增] 分別計算 Positive (1) 和 Negative (0) 的 F1
        pos_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        neg_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=0)
        
        # Recall (針對 Pos)
        rec = recall_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        
        print(f"{d:<5} | {count:<6} | {pos_rate:.2f}     | {acc:.4f}   | {pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}")

    # 針對長距離 (>= 5)
    mask_long = (dists >= 5)
    sub_preds_long = preds[mask_long]
    sub_labels_long = labels[mask_long]
    count_long = len(sub_labels_long)
    
    if count_long > 0:
        pos_rate = np.mean(sub_labels_long)
        acc = accuracy_score(sub_labels_long, sub_preds_long)
        pos_f1 = f1_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        neg_f1 = f1_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=0)
        rec = recall_score(sub_labels_long, sub_preds_long, zero_division=0, pos_label=1)
        
        print(f"{'>=5':<5} | {count_long:<6} | {pos_rate:.2f}     | {acc:.4f}   | {pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}")
    else:
        print(f"{'>=5':<5} | {0:<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

    print("-" * 80)
# ============================================================
# Monitor Function (Gated Version)
# ============================================================
def monitor_gated_stats(model, loader, device, desc="Monitoring"):
    model.eval()
    avg_gate_val = 0.0
    avg_align_sim = 0.0
    count = 0
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)
            _, aux = model(batch, return_aux=True)
            
            gate = aux.get("gate", None)
            if gate is not None:
                avg_gate_val += gate.mean().item()
            
            z_stud = aux.get("z_student", None)
            z_teach = aux.get("z_teacher", None)
            if z_stud is not None and z_teach is not None:
                sim = cosine_sim(z_stud, z_teach).mean().item()
                avg_align_sim += sim
                
            count += 1
            if count >= 50: break 

    print(f"\n[Gated Monitor] Stats over {count} batches:")
    print(f"  > Gate Openness: {avg_gate_val/count:.4f} (0=Closed, 1=Open)")
    print(f"  > S-T Alignment: {avg_align_sim/count:.4f} (Cosine Sim, -1~1)")
    print("-" * 40)

# ============================================================
# Evaluate
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating", return_dist_stats=False):
    model.eval()
    all_logits, all_labels = [], []
    all_dists = []

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        out = model(batch)
        
        if isinstance(out, tuple): 
            logits = out[0]
        else: 
            logits = out

        labels = batch.edge_label.float()
        if logits.numel() == 0: continue

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        if return_dist_stats:
            dists = calculate_pair_distance(batch)
            all_dists.extend(dists)

    if not all_logits: return None

    # 合併 batch logits/labels（仍保留 torch tensor）
    all_logits_t = torch.cat(all_logits)            # torch.Tensor on CPU (你前面已 .cpu() append)
    all_labels_t = torch.cat(all_labels)
    all_probs  = torch.sigmoid(all_logits_t).cpu().numpy()
    all_labels = all_labels_t.cpu().numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    best_res = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
    thr_candidates = np.concatenate([np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)])

    # Find Best Threshold
    final_preds = None
    
    for thr in thr_candidates:
        preds = (all_probs >= thr).astype(int)
        pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
        macro = (pos_f1 + neg_f1) / 2.0
        acc = accuracy_score(all_labels, preds)

        if macro > best_res["macro_f1"]:
            best_res = {
                "macro_f1": float(macro),
                "pos_f1": float(pos_f1),
                "neg_f1": float(neg_f1),
                "acc": float(acc),
                "auc": float(auc),
                "thr": float(thr),
            }
            # Keep best preds for analysis
            if return_dist_stats:
                final_preds = preds

    print(
        f"[Eval] Best MacroF1: {best_res['macro_f1']:.4f} "
        f"(Pos={best_res['pos_f1']:.2f}, Neg={best_res['neg_f1']:.2f}, "
        f"Acc={best_res['acc']:.4f}, AUC={best_res['auc']:.4f})"
    )
    
    # Distance Breakdown (Only if requested)
    if return_dist_stats and final_preds is not None:
        analyze_distance_performance(final_preds, all_labels, all_dists)
        
    return best_res

# ============================================================
# Train
# ============================================================
def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device):
    os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
    best_macro = 0.0
    no_improve = 0
    distill_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)

    for epoch in range(1, cfg.train.EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.train.EPOCHS} ===")
        model.train()
        optimizer.zero_grad()

        total_loss, total_task, total_expl = 0.0, 0.0, 0.0
        total_pairs = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch}")):
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple) and len(out) == 3:
                logits, z_stud, z_teach = out
            else:
                logits, z_stud, z_teach = out, None, None

            labels = batch.edge_label.float()
            if logits.numel() == 0: continue

            task_loss = loss_fn(logits, labels)
            
            if cfg.model.USE_EXPLAIN_SPACE and z_stud is not None and z_teach is not None:
                target = torch.ones(z_stud.size(0)).to(device)
                expl_loss = distill_loss_fn(z_stud, z_teach, target)
            else:
                expl_loss = torch.tensor(0.0, device=device)
            # expl_loss = torch.tensor(0.0, device=device)

            loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss
            loss = loss / cfg.train.ACCUM_STEPS
            loss.backward()

            if (step + 1) % cfg.train.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            bs = int(labels.size(0))
            total_loss += loss.item() * cfg.train.ACCUM_STEPS * bs
            total_task += task_loss.item() * bs
            total_expl += expl_loss.item() * bs
            total_pairs += bs

        print(f"Train loss={total_loss/total_pairs:.4f} (task={total_task/total_pairs:.4f}, expl={total_expl/total_pairs:.4f})")

        # Evaluate (Normal)
        val = evaluate(model, valid_loader, device)
        monitor_gated_stats(model, valid_loader, device)

        is_best = False
        if val and val["macro_f1"] > best_macro:
            best_macro = val["macro_f1"]
            is_best = True
            no_improve = 0
            torch.save(model.state_dict(), cfg.path.SAVE_MODEL_PATH)
            print(f"Saved best model -> {cfg.path.SAVE_MODEL_PATH}")
        else:
            no_improve += 1
        
        print(f"Current Epoch: {val['macro_f1']:.4f} | >>> Global Best MacroF1: {best_macro:.4f} <<<")

        if not is_best and no_improve >= cfg.train.PATIENCE:
            print("Early stopping.")
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

    if cfg.train.RUN_MODE == "train":
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
        
        total_steps = len(train_loader) * cfg.train.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.train.WARMUP_RATIO), total_steps)

        train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device)

    else: # TEST
        test_dataset = CEEEnd2EndDataset(
            cfg.path.CONV_PATH, cfg.path.TEST_PATH,
            cfg.path.EX_TEST_PT, cfg.path.EX_TEST_TSV,
            use_explain=cfg.model.USE_EXPLAIN,
            expl_space_pt=cfg.path.ES_TEST_PT, expl_space_tsv=cfg.path.ES_TEST_TSV,
            use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
        )
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = CrossTowerCausalModel(
            text_model_name=cfg.model.TEXT_MODEL,
            hidden_dim=cfg.model.HIDDEN_DIM,
            expl_dim=test_dataset.get_explain_dim(),
            num_speakers=test_dataset.num_speakers,
            num_emotions=test_dataset.num_emotions,
            dropout=cfg.model.BASE_DROPOUT,
            gnn_dropout=cfg.model.GNN_DROPOUT,
            num_gnn_layers=cfg.model.NUM_GNN_LAYERS,
            freeze_text=cfg.model.FREEZE_TEXT,
            use_explain=cfg.model.USE_EXPLAIN,
            expl_space_dim=test_dataset.get_expl_space_dim(),
            use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
        ).to(device)

        if os.path.exists(cfg.path.SAVE_MODEL_PATH):
            model.load_state_dict(torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device))
            print(f"Loaded: {cfg.path.SAVE_MODEL_PATH}")
        
        monitor_gated_stats(model, test_loader, device, desc="Monitoring Test")

        # [Modified]: Enable distance stats for Test
        results = evaluate(model, test_loader, device, desc="Testing", return_dist_stats=True)
        print(json.dumps(results, indent=2))
        with open(cfg.path.OUT_JSON_PATH, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
