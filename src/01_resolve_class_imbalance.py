"""
STEP 1: Resolve class imbalance with 1D augmentation (training folds only).

Decides how many synthetic minority-class samples to add, and which real record
each one is generated from (see src/imbalance.py). The synthetic signals are not
stored: step 3 regenerates each one from its source record with a fresh random 1D
augmentation (time stretch, shift, scaling, noise, cutout) every epoch. The model
therefore never trains on an exact duplicate, which is what makes plain random
oversampling overfit.
Validation (fold 9) and test (fold 10) are never resampled, so reported metrics
reflect the real class distribution.

Outputs:
  data/processed/balanced_train_ecg_ids.npy        source ecg_id of every training sample
  data/processed/balanced_train_is_synthetic.npy   True where the sample is synthetic
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
from src.imbalance import irlbl, label_counts, mean_ir, select_synthetic_sources


def main():
    config = load_config()
    cfg = config["imbalance"]

    df = load_metadata(config)
    train_mask, _, _ = split_masks(df, config)
    train_df = df[train_mask]
    y = label_matrix(train_df)

    rows, is_synthetic = select_synthetic_sources(y, target_ir=cfg["target_ir"], max_growth=cfg["max_growth"], seed=cfg["seed"])
    y_bal = y[rows]

    np.save(processed_path(config, "balanced_train_ecg_ids.npy"), train_df.index.values[rows])
    np.save(processed_path(config, "balanced_train_is_synthetic.npy"), is_synthetic)

    summary = pd.DataFrame({
        "Superclass": SUPERCLASSES,
        "Count_Before": label_counts(y),
        "IRLbl_Before": irlbl(y).round(3),
        "Synthetic_Added": label_counts(y_bal[is_synthetic]),
        "Count_After": label_counts(y_bal),
        "IRLbl_After": irlbl(y_bal).round(3),
    })
    print("=" * 72)
    print("STEP 1: CLASS IMBALANCE RESOLUTION (augmentation-based oversampling)")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nTraining samples : {len(y)} real + {int(is_synthetic.sum())} synthetic = {len(y_bal)}")
    print(f"MeanIR           : {mean_ir(y):.3f} -> {mean_ir(y_bal):.3f}")

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    summary.to_csv(os.path.join(results_dir, "class_distribution_before_after.csv"), index=False)

    plot_dir = os.path.join(config["output"]["artifacts_dir"], "imbalance")
    os.makedirs(plot_dir, exist_ok=True)
    x = np.arange(len(SUPERCLASSES))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary.Count_Before, 0.4, label=f"Before (MeanIR {mean_ir(y):.2f})")
    plt.bar(x + 0.2, summary.Count_Before, 0.4, label="After: real")
    plt.bar(x + 0.2, summary.Synthetic_Added, 0.4, bottom=summary.Count_Before,
            label=f"After: synthetic 1D-augmented (MeanIR {mean_ir(y_bal):.2f})")
    plt.xticks(x, SUPERCLASSES)
    plt.ylabel("Training samples with label")
    plt.title("PTB-XL superclass distribution, training folds 1-8")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "class_distribution_before_after.png"), dpi=200)
    plt.close()
    print(f"\nSaved training plan, CSV summary and plot.")


if __name__ == "__main__":
    main()
