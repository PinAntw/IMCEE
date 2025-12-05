# -*- coding: utf-8 -*-
# IMCEE/utils/save_predictions.py

import csv
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


@torch.no_grad()
def evaluate_and_save(model, loader, device, csv_path):
    """
    驗證或測試時同時計算指標並輸出每個 pair 的預測結果。
    只輸出任務邊 (edge_task_mask=True)。
    """
    model.eval()
    all_logits, all_labels, all_probs = [], [], []
    records = []

    for batch in tqdm(loader, desc="Evaluating and saving predictions"):
        batch = batch.to(device)
        logits = model(batch)
        mask = batch.edge_task_mask

        if mask.sum() == 0:
            continue

        logits = logits[mask]
        labels = batch.edge_labels[mask]
        probs = torch.sigmoid(logits)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

        # 若資料中有 conv_id / edge_index / texts，則可輸出更完整內容
        # 注意：這裡假設每個 batch 是 Batch(from_data_list)，保留了每個 node 的 texts
        # 為安全起見僅輸出 prob, label, pred
        for p, y in zip(probs.cpu().numpy(), labels.cpu().numpy()):
            records.append({"prob": float(p), "label": int(y)})

    # === 聚合所有結果 ===
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    best_macro_f1, best_thr, best_pos_f1, best_neg_f1, best_acc = 0, 0.5, 0, 0, 0
    for thr in np.linspace(0.1, 0.9, 9):
        preds = (all_probs >= thr).astype(int)
        pos_f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        neg_f1 = f1_score(all_labels, preds, pos_label=0, zero_division=0)
        macro_f1 = (pos_f1 + neg_f1) / 2
        acc = accuracy_score(all_labels, preds)
        if macro_f1 > best_macro_f1:
            best_macro_f1, best_thr, best_pos_f1, best_neg_f1, best_acc = (
                macro_f1, thr, pos_f1, neg_f1, acc
            )

    # === 輸出 CSV ===
    preds_final = (all_probs >= best_thr).astype(int)
    for i in range(len(records)):
        records[i]["pred"] = int(preds_final[i])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prob", "label", "pred"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[Saved predictions to {csv_path}]")

    return {
        "auc": auc,
        "pos_f1": best_pos_f1,
        "neg_f1": best_neg_f1,
        "macro_f1": best_macro_f1,
        "acc": best_acc,
        "thr": best_thr,
    }
