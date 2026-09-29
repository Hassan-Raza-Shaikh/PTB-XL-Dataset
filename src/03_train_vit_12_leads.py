"""
STEP 3: Train and evaluate the 1D ViT on all 12 denoised leads (500 Hz).

  - Training set  : balanced ecg_ids from step 1, read from the step 2 cache,
                    with on-the-fly batch augmentation (scale, shift, cutout)
  - Validation    : fold 9, used for checkpoint selection, early stopping and
                    per-class threshold tuning
  - Test          : fold 10, evaluated once with the best checkpoint

Outputs:
  artifacts/models/vit_12lead_500hz_best.pth
  artifacts/models/vit_12lead_500hz_roc_curves.png
  artifacts/models/vit_12lead_500hz_training_history.png
  docs/results/vit_12lead_500hz_metrics.csv          (per-class)
  docs/results/vit_12lead_500hz_summary.csv          (macro, thresholds 0.5 vs tuned)
  docs/results/vit_12lead_500hz_thresholds.csv

Usage:
  PYTHONPATH=. python3 src/03_train_vit_12_leads.py
"""

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tqdm import tqdm

from src.augmentation import BatchECGAugmenter
from src.data import SUPERCLASSES, label_matrix, load_config, load_metadata, processed_path, split_masks
from src.imbalance import tune_thresholds
from src.vit import ECGViT1D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iterate_batches(X, y, rows, batch_size, shuffle):
    order = np.random.permutation(rows) if shuffle else rows
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield torch.from_numpy(X[idx].astype(np.float32)), torch.from_numpy(y[idx])


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, X, y, rows, batch_size, criterion, optimizer, scheduler, augmenter, device):
    model.train()
    total = 0.0
    n_batches = math.ceil(len(rows) / batch_size)
    for xb, yb in tqdm(iterate_batches(X, y, rows, batch_size, shuffle=True), total=n_batches, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        xb = augmenter(xb)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(yb)
    return total / len(rows)


@torch.no_grad()
def predict(model, X, y, rows, batch_size, criterion, device):
    model.eval()
    total, probs = 0.0, []
    for xb, yb in iterate_batches(X, y, rows, batch_size, shuffle=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += criterion(logits, yb).item() * len(yb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return total / len(rows), np.vstack(probs)


def macro_auc(y_true, y_prob):
    return float(np.mean([roc_auc_score(y_true[:, c], y_prob[:, c]) for c in range(y_true.shape[1])]))


def summarize(y_true, y_prob, thresholds):
    pred = (y_prob >= thresholds).astype(int)
    return {
        "Macro_ROC_AUC": macro_auc(y_true, y_prob),
        "Macro_F1": f1_score(y_true, pred, average="macro", zero_division=0),
        "Macro_Precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, pred, average="macro", zero_division=0),
    }


def main():
    config = load_config()
    tcfg = config["training"]
    set_seed(tcfg["seed"])
    device = pick_device(tcfg["device"])

    df = load_metadata(config)
    y_all = label_matrix(df)
    _, val_mask, test_mask = split_masks(df, config)

    X = np.load(processed_path(config, "signals_500hz_12lead_denoised.npy"))  # (N, 12, 5000) float16, ~2.6 GB
    cache_ids = np.load(processed_path(config, "signals_500hz_12lead_ecg_ids.npy"))
    assert np.array_equal(cache_ids, df.index.values), "Denoised cache row order does not match metadata; rerun step 2."

    row_of = pd.Series(np.arange(len(df)), index=df.index)
    train_rows = row_of.loc[np.load(processed_path(config, "balanced_train_ecg_ids.npy"))].values
    val_rows = np.where(val_mask)[0]
    test_rows = np.where(test_mask)[0]

    print("=" * 72)
    print("STEP 3: 1D ViT ON ALL 12 DENOISED LEADS @ 500 Hz")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Train (balanced) = {len(train_rows)} | Val = {len(val_rows)} | Test = {len(test_rows)}")

    model = ECGViT1D.from_config(config, num_classes=len(SUPERCLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ECGViT1D: 12 leads, {model.num_patches} overlapping tokens, {n_params:,} parameters")

    batch_size = tcfg["batch_size"]
    epochs = tcfg["epochs"]
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = warmup_cosine(optimizer, tcfg["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch)
    augmenter = BatchECGAugmenter.from_config(config)

    models_dir = os.path.join(config["output"]["artifacts_dir"], "models")
    results_dir = config["output"]["results_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "vit_12lead_500hz_best.pth")

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, stale = -1.0, 0
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, X, y_all, train_rows, batch_size, criterion, optimizer, scheduler, augmenter, device)
        v_loss, v_prob = predict(model, X, y_all, val_rows, batch_size, criterion, device)
        v_auc = macro_auc(y_all[val_rows], v_prob)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_auc"].append(v_auc)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {t_loss:.4f} | val loss {v_loss:.4f} | val macro AUC {v_auc:.4f}")

        if v_auc > best_auc:
            best_auc, stale = v_auc, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> saved checkpoint (best val AUC {best_auc:.4f})")
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                print(f"Early stopping: no val AUC improvement for {stale} epochs.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, val_prob = predict(model, X, y_all, val_rows, batch_size, criterion, device)
    _, test_prob = predict(model, X, y_all, test_rows, batch_size, criterion, device)
    y_val, y_test = y_all[val_rows], y_all[test_rows]

    default_thr = np.full(len(SUPERCLASSES), 0.5)
    tuned_thr = tune_thresholds(y_val, val_prob) if config["imbalance"]["tune_thresholds"] else default_thr
    pd.DataFrame({"Superclass": SUPERCLASSES, "Threshold": tuned_thr}).to_csv(
        os.path.join(results_dir, "vit_12lead_500hz_thresholds.csv"), index=False)

    summary = pd.DataFrame([
        {"Thresholds": "0.5", **summarize(y_test, test_prob, default_thr)},
        {"Thresholds": "tuned on val", **summarize(y_test, test_prob, tuned_thr)},
    ])
    summary.to_csv(os.path.join(results_dir, "vit_12lead_500hz_summary.csv"), index=False)

    pred = (test_prob >= tuned_thr).astype(int)
    per_class = pd.DataFrame([{
        "Superclass": c,
        "Threshold": tuned_thr[i],
        "ROC_AUC": roc_auc_score(y_test[:, i], test_prob[:, i]),
        "F1_Score": f1_score(y_test[:, i], pred[:, i], zero_division=0),
        "Precision": precision_score(y_test[:, i], pred[:, i], zero_division=0),
        "Recall": recall_score(y_test[:, i], pred[:, i], zero_division=0),
        "Support": int(y_test[:, i].sum()),
    } for i, c in enumerate(SUPERCLASSES)])
    per_class.to_csv(os.path.join(results_dir, "vit_12lead_500hz_metrics.csv"), index=False)

    print("\n" + "=" * 72)
    print("TEST RESULTS (fold 10, best checkpoint)")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    print("\nPer class (tuned thresholds):")
    print(per_class.round(4).to_string(index=False))

    plt.figure(figsize=(9, 7))
    for i, c in enumerate(SUPERCLASSES):
        fpr, tpr, _ = roc_curve(y_test[:, i], test_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"12-lead ViT, 500 Hz, balanced + denoised\nTest macro ROC-AUC {summary.Macro_ROC_AUC[0]:.4f}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(models_dir, "vit_12lead_500hz_roc_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, history["train_loss"], "b-o", label="train")
    ax1.plot(ep, history["val_loss"], "r-o", label="val")
    ax1.set_title("BCE loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(ep, history["val_auc"], "g-o")
    ax2.set_title("Validation macro ROC-AUC")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "vit_12lead_500hz_training_history.png"), dpi=200)
    plt.close()
    print(f"\nSaved checkpoint, plots and CSVs.")


if __name__ == "__main__":
    main()
