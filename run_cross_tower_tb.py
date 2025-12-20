#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/run_cross_tower_tb.py

import os
import json
import time  # [Added] For timestamping TensorBoard logs
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # [Added] TensorBoard Support
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from modules.dataset import CEEEnd2EndDataset, End2EndCollate
from modules.crossTower.cross_tower_model import CrossTowerCausalModel
from config import cfg

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


# ============================================================
# Distance utilities (restored)
# ============================================================
def calculate_pair_distance(batch):
    """
    Calculates the true utterance distance for each Cause-Target pair in the batch.

    Priority:
      1) Use batch.pair_uttpos if provided (stable)
      2) Fallback: infer anchored utterances via Type-0 edges
         NOTE: current dataset Type-0 is utt -> super (NOT super -> utt)
         and PyG batching makes edge_index global while target_node_indices are local.
    """
    # 1) 優先用 dataset 提供的 mapping（最穩定）
    if hasattr(batch, "pair_uttpos") and batch.pair_uttpos is not None:
        pos = batch.pair_uttpos  # [B,2] or [B,1,2]
        if pos.dim() == 3:
            pos = pos.view(-1, 2)
        d = (pos[:, 1] - pos[:, 0]).abs()
        return d.detach().cpu().numpy()

    # -----------------------------
    # 2) Fallback: use Type-0 edges
    # -----------------------------
    batch_dists = []

    # local node indices per-graph
    cause_nodes_local = batch.target_node_indices[:, 0]
    target_nodes_local = batch.target_node_indices[:, 1]

    # edge_index are global after batching
    edge_src = batch.edge_index[0]
    edge_tgt = batch.edge_index[1]
    edge_types = batch.edge_types if hasattr(batch, "edge_types") else None

    # need ptr offsets to convert local->global and global->local
    if not hasattr(batch, "ptr"):
        # keep old behavior as safe fallback (best-effort, but may be wrong for batch>1)
        for i in range(len(cause_nodes_local)):
            # old heuristic (NOT ideal)
            c_node = cause_nodes_local[i].item()
            t_node = target_nodes_local[i].item()

            # old logic assumed super -> utt; we keep a conservative fallback to avoid crash
            mask_c = (edge_src == c_node)
            c_utt_idx = edge_tgt[mask_c].min().item() if mask_c.any() else 0

            mask_t = (edge_src == t_node)
            t_utt_idx = edge_tgt[mask_t].min().item() if mask_t.any() else 0

            batch_dists.append(abs(t_utt_idx - c_utt_idx))

        return np.array(batch_dists)

    offsets = batch.ptr[:-1]  # [B]

    for i in range(len(cause_nodes_local)):
        off = offsets[i].item()

        # convert local super-node idx -> global idx
        c_super_g = int(cause_nodes_local[i].item() + off)
        t_super_g = int(target_nodes_local[i].item() + off)

        # Type0 is utt -> super, so we search edges where tgt == super
        if edge_types is not None:
            mask_c = (edge_types == 0) & (edge_tgt == c_super_g)
            mask_t = (edge_types == 0) & (edge_tgt == t_super_g)
        else:
            # best-effort if no edge_types
            mask_c = (edge_tgt == c_super_g)
            mask_t = (edge_tgt == t_super_g)

        # get anchored utterance global idx from edge_src, then to local by subtracting off
        if mask_c.any():
            c_utt_local = int((edge_src[mask_c].min().item()) - off)
            c_utt_local = max(c_utt_local, 0)
        else:
            c_utt_local = 0

        if mask_t.any():
            t_utt_local = int((edge_src[mask_t].min().item()) - off)
            t_utt_local = max(t_utt_local, 0)
        else:
            t_utt_local = 0

        batch_dists.append(abs(t_utt_local - c_utt_local))

    return np.array(batch_dists)



def analyze_distance_performance(preds, labels, dists):
    """
    Detailed accuracy analysis for different distances
    Includes Pos F1 and Neg F1 specifically.
    """
    dists = np.array(dists)
    preds = np.array(preds)
    labels = np.array(labels)
    
    print("\n" + "="*80)
    print("   MODEL PREDICTION - DISTANCE BREAKDOWN")
    print("="*80)
    print(f"{'Dist':<5} | {'Count':<6} | {'PosRate':<8} | "
          f"{'Acc':<8} | {'PosF1':<8} | {'NegF1':<8} | {'Recall':<8}")
    print("-" * 80)
    
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    for d in targets:
        mask = (dists == d)
        sub_preds = preds[mask]
        sub_labels = labels[mask]
        
        count = len(sub_labels)
        if count == 0:
            print(f"{d:<5} | {0:<6} | {'N/A':<8} | {'N/A':<8} | "
                  f"{'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            continue
        
        pos_rate = np.mean(sub_labels)
        acc = accuracy_score(sub_labels, sub_preds)
        pos_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        neg_f1 = f1_score(sub_labels, sub_preds, zero_division=0, pos_label=0)
        rec = recall_score(sub_labels, sub_preds, zero_division=0, pos_label=1)
        
        print(f"{d:<5} | {count:<6} | {pos_rate:.2f}     | {acc:.4f}   | "
              f"{pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}")

    # 長距離 (>= 5)
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
        
        print(f"{'>=5':<5} | {count_long:<6} | {pos_rate:.2f}     | {acc:.4f}   | "
              f"{pos_f1:.4f}   | {neg_f1:.4f}   | {rec:.4f}")
    else:
        print(f"{'>=5':<5} | {0:<6} | {'N/A':<8} | {'N/A':<8} | "
              f"{'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

    print("-" * 80)


# ============================================================
# Monitor Function (Gated Version)
# ============================================================
def monitor_gated_stats(model, loader, device, desc="Monitoring", max_batches=50):
    model.eval()

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    gate_sum, gate_n = 0.0, 0
    sim_sum,  sim_n  = 0.0, 0
    seen = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)

            out = model(batch, return_aux=True)
            if isinstance(out, tuple) and len(out) == 2:
                _, aux = out
            else:
                aux = {}

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
            if seen >= max_batches:
                break

    print(f"\n[Gated Monitor] Stats over {seen} batches:")
    if gate_n > 0:
        print(f"  > Gate Openness: {gate_sum / gate_n:.4f} (0=Closed, 1=Open) [gate_batches={gate_n}]")
    else:
        print(f"  > Gate Openness: N/A (no gate found)")

    if sim_n > 0:
        print(f"  > S-T Alignment: {sim_sum / sim_n:.4f} (Cosine Sim, -1~1) [align_batches={sim_n}]")
    else:
        print(f"  > S-T Alignment: N/A (no teacher/student found)")

    print("-" * 40)


# ============================================================
# Evaluate
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating", return_dist_stats=False):
    """
    Evaluate model; if return_dist_stats=True, will also print distance breakdown.
    """
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
        if logits.numel() == 0:
            continue

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        if return_dist_stats:
            dists = calculate_pair_distance(batch)
            all_dists.extend(dists)

    if not all_logits:
        return None

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    all_probs  = torch.sigmoid(all_logits_t).cpu().numpy()
    all_labels = all_labels_t.cpu().numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    best_res = {"auc": float(auc), "macro_f1": 0.0, "thr": 0.5}
    thr_candidates = np.concatenate(
        [np.linspace(0.01, 0.30, 30), np.linspace(0.35, 0.90, 12)]
    )

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
            if return_dist_stats:
                final_preds = preds

    print(
        f"[Eval] Best MacroF1: {best_res['macro_f1']:.4f} "
        f"(Pos={best_res['pos_f1']:.2f}, Neg={best_res['neg_f1']:.2f}, "
        f"Acc={best_res['acc']:.4f}, AUC={best_res['auc']:.4f})"
    )
    
    if return_dist_stats and final_preds is not None:
        analyze_distance_performance(final_preds, all_labels, all_dists)
        
    return best_res


# ============================================================
# Train (with TensorBoard)
# ============================================================
def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, writer=None):
    os.makedirs(os.path.dirname(cfg.path.SAVE_MODEL_PATH), exist_ok=True)
    best_macro = 0.0
    no_improve = 0
    distill_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)
    
    global_step = 0  # [Added] Global step for TensorBoard

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
            if logits.numel() == 0:
                continue

            task_loss = loss_fn(logits, labels)
            
            if cfg.model.USE_EXPLAIN_SPACE and z_stud is not None and z_teach is not None:
                target = torch.ones(z_stud.size(0)).to(device)
                expl_loss = distill_loss_fn(z_stud, z_teach, target)
            else:
                expl_loss = torch.tensor(0.0, device=device)

            loss = task_loss + cfg.train.LAMBDA_EXPL * expl_loss
            loss = loss / cfg.train.ACCUM_STEPS
            loss.backward()

            if (step + 1) % cfg.train.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # [Added] TensorBoard logging per update
                if writer is not None:
                    writer.add_scalar('Train/Total_Loss', loss.item() * cfg.train.ACCUM_STEPS, global_step)
                    writer.add_scalar('Train/Task_Loss', task_loss.item(), global_step)
                    writer.add_scalar('Train/Expl_Loss', expl_loss.item(), global_step)
                    writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
                    
                    # [新增] 紀錄 Alpha 變化
                    if hasattr(model.structural_tower, 'alpha'):
                        writer.add_scalar('Train/Alpha', model.structural_tower.alpha.item(), global_step)

                global_step += 1

            bs = int(labels.size(0))
            total_loss += loss.item() * cfg.train.ACCUM_STEPS * bs
            total_task += task_loss.item() * bs
            total_expl += expl_loss.item() * bs
            total_pairs += bs

        print(f"Train loss={total_loss/total_pairs:.4f} "
              f"(task={total_task/total_pairs:.4f}, expl={total_expl/total_pairs:.4f})")

        # Validation (no distance breakdown here, same as你舊版行為)
        val = evaluate(model, valid_loader, device, desc="Validating", return_dist_stats=False)
        monitor_gated_stats(model, valid_loader, device)

        # [Added] Log Validation Metrics to TensorBoard
        if writer is not None and val is not None:
            writer.add_scalar('Valid/Macro_F1', val['macro_f1'], epoch)
            writer.add_scalar('Valid/Pos_F1', val['pos_f1'], epoch)
            writer.add_scalar('Valid/Neg_F1', val['neg_f1'], epoch)
            writer.add_scalar('Valid/Accuracy', val['acc'], epoch)
            writer.add_scalar('Valid/AUC', val['auc'], epoch)

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
# Main (with TensorBoard)
# ============================================================
def main():
    set_seed(cfg.train.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.TEXT_MODEL)
    # collate_fn = End2EndCollate(tokenizer)

    if cfg.train.RUN_MODE == "train":
        # [修改重點] 使用明確的關鍵字參數來避免位置衝突
        train_dataset = CEEEnd2EndDataset(
            conversations_path=cfg.path.CONV_PATH,
            pairs_path=cfg.path.TRAIN_PATH,
            llm_shortcut_path=cfg.path.LLM_SHORTCUT_TRAIN,
            explain_embed_path=cfg.path.EX_TRAIN_PT,
            explain_index_path=cfg.path.EX_TRAIN_TSV,
            use_explain=cfg.model.USE_EXPLAIN,
            expl_space_pt=cfg.path.ES_TRAIN_PT,
            expl_space_tsv=cfg.path.ES_TRAIN_TSV,
            use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
        )
        valid_dataset = CEEEnd2EndDataset(
            conversations_path=cfg.path.CONV_PATH,
            pairs_path=cfg.path.VALID_PATH,
            llm_shortcut_path=cfg.path.LLM_SHORTCUT_VALID,
            explain_embed_path=cfg.path.EX_VALID_PT,
            explain_index_path=cfg.path.EX_VALID_TSV,
            use_explain=cfg.model.USE_EXPLAIN,
            expl_space_pt=cfg.path.ES_VALID_PT,
            expl_space_tsv=cfg.path.ES_VALID_TSV,
            use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
        )
        collate_fn = End2EndCollate(
            tokenizer,
            spk_pad_id=train_dataset.spk_pad_id,  # 確保對齊
            emo_pad_id=train_dataset.emo_pad_id
        )



        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.train.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

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
        base_params = filter(
            lambda p: id(p) not in plm_ids and id(p) not in gnn_ids,
            model.parameters()
        )

        optimizer = torch.optim.AdamW([
            {'params': base_params,
             'lr': cfg.optim.LR_BASE,
             'weight_decay': cfg.optim.WD_BASE},
            {'params': model.structural_tower.parameters(),
             'lr': cfg.optim.LR_GNN,
             'weight_decay': cfg.optim.WD_GNN},
            {'params': model.text_encoder.text_encoder.parameters(),
             'lr': cfg.optim.LR_PLM,
             'weight_decay': cfg.optim.WD_PLM},
        ])

        pos_weight = compute_pos_weight_from_dataset(train_dataset)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        
        total_steps = len(train_loader) * cfg.train.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * cfg.train.WARMUP_RATIO),
            total_steps
        )

        # [Added] Initialize TensorBoard Writer
        timestamp = time.strftime('%m%d_%H%M')
        log_dir_name = f"{cfg.path.SAVE_SUFFIX}_{timestamp}"
        log_dir_path = os.path.join(cfg.path.TENSORBOARD_LOG_DIR, log_dir_name)
        print(f"[TensorBoard] Logging to: {log_dir_path}")
        writer = SummaryWriter(log_dir=log_dir_path)

        try:
            train(
                model,
                train_loader,
                valid_loader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                writer=writer
            )
        finally:
            writer.close()

    else:  # TEST
        # [修改重點] 使用明確的關鍵字參數
        test_dataset = CEEEnd2EndDataset(
            conversations_path=cfg.path.CONV_PATH,
            pairs_path=cfg.path.TEST_PATH,
            llm_shortcut_path=cfg.path.LLM_SHORTCUT_TEST,
            explain_embed_path=cfg.path.EX_TEST_PT,
            explain_index_path=cfg.path.EX_TEST_TSV,
            use_explain=cfg.model.USE_EXPLAIN,
            expl_space_pt=cfg.path.ES_TEST_PT,
            expl_space_tsv=cfg.path.ES_TEST_TSV,
            use_expl_space=cfg.model.USE_EXPLAIN_SPACE,
        )
        collate_fn = End2EndCollate(
                    tokenizer,
                    spk_pad_id=test_dataset.spk_pad_id,
                    emo_pad_id=test_dataset.emo_pad_id
                )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.train.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

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
            model.load_state_dict(
                torch.load(cfg.path.SAVE_MODEL_PATH, map_location=device)
            )
            print(f"Loaded: {cfg.path.SAVE_MODEL_PATH}")
        
        monitor_gated_stats(model, test_loader, device, desc="Monitoring Test")

        # Distance stats restored here
        results = evaluate(
            model,
            test_loader,
            device,
            desc="Testing",
            return_dist_stats=True
        )
        print(json.dumps(results, indent=2))
        with open(cfg.path.OUT_JSON_PATH, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()