"""
STEP 1: Resolve class imbalance (training folds only).

Applies multi-label random oversampling (see src/imbalance.py) to folds 1-8 and
saves the balanced list of training ecg_ids. Validation (fold 9) and test (fold 10)
are never resampled, so reported metrics reflect the real class distribution.

The oversampling is index-level: a cloned record is simply listed twice. Step 2
denoises each unique record once, and step 3 reads the balanced list, so this is
equivalent to denoising the oversampled set (denoising is deterministic per record),
while on-the-fly augmentation in step 3 makes each copy look different.

Outputs:
  data/processed/balanced_train_ecg_ids.npy
  docs/results/class_distribution_before_after.csv
  artifacts/imbalance/class_distribution_before_after.png

Usage:
  PYTHONPATH=. python3 src/01_resolve_class_imbalance.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import SUPERCLASSES, label_matrix, load_config, load_metadata, processed_path, split_masks
from src.imbalance import irlbl, label_counts, mean_ir, multilabel_random_oversample


def main():
    config = load_config()
    cfg = config["imbalance"]

    df = load_metadata(config)
    train_mask, _, _ = split_masks(df, config)
    train_df = df[train_mask]
    y = label_matrix(train_df)

    rows = multilabel_random_oversample(y, target_ir=cfg["target_ir"], max_growth=cfg["max_growth"], seed=cfg["seed"])
    y_bal = y[rows]
    balanced_ids = train_df.index.values[rows]

    np.save(processed_path(config, "balanced_train_ecg_ids.npy"), balanced_ids)

    summary = pd.DataFrame({
        "Superclass": SUPERCLASSES,
        "Count_Before": label_counts(y),
        "IRLbl_Before": irlbl(y).round(3),
        "Count_After": label_counts(y_bal),
        "IRLbl_After": irlbl(y_bal).round(3),
    })
    print("=" * 72)
    print("STEP 1: CLASS IMBALANCE RESOLUTION (multi-label random oversampling)")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nTraining records : {len(y)} -> {len(y_bal)} (+{len(y_bal) - len(y)} clones)")
    print(f"MeanIR           : {mean_ir(y):.3f} -> {mean_ir(y_bal):.3f}")

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    summary.to_csv(os.path.join(results_dir, "class_distribution_before_after.csv"), index=False)

    plot_dir = os.path.join(config["output"]["artifacts_dir"], "imbalance")
    os.makedirs(plot_dir, exist_ok=True)
    x = np.arange(len(SUPERCLASSES))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary.Count_Before, 0.4, label=f"Before (MeanIR {mean_ir(y):.2f})")
    plt.bar(x + 0.2, summary.Count_After, 0.4, label=f"After ML-ROS (MeanIR {mean_ir(y_bal):.2f})")
    plt.xticks(x, SUPERCLASSES)
    plt.ylabel("Training records with label")
    plt.title("PTB-XL superclass distribution, training folds 1-8")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "class_distribution_before_after.png"), dpi=200)
    plt.close()
    print(f"\nSaved balanced ids, CSV summary and plot.")


if __name__ == "__main__":
    main()
