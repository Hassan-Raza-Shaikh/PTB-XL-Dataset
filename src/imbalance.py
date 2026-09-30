"""
Multi-label class-imbalance handling for PTB-XL superclasses.

PTB-XL is multi-label (a record can be MI + STTC + HYP at once), so single-label
tools such as SMOTE or a class-balanced sampler do not apply directly. We use the
imbalance measures from Charte et al. (2015):

    IRLbl(c) = max_k count(k) / count(c)      (1.0 for the majority label)
    MeanIR   = mean_c IRLbl(c)

Balancing is augmentation-based oversampling. Source records are chosen the way
ML-ROS (multi-label random oversampling) would choose them: repeatedly pick a
training record carrying the label with the highest IRLbl, avoiding records that
also carry the majority label so balancing a minority label does not inflate the
majority. Unlike plain ML-ROS, a pick is not trained on as a duplicate. It becomes
a *synthetic* sample that the step-3 augmenter regenerates from its source with a
fresh random 1D augmentation every time it is drawn (see src/augmentation.py).
"""

import numpy as np
from sklearn.metrics import f1_score


def label_counts(y):
    return y.sum(axis=0).astype(np.int64)


def irlbl(y):
    counts = label_counts(y).astype(float)
    return counts.max() / np.maximum(counts, 1.0)


def mean_ir(y):
    return float(irlbl(y).mean())


def select_synthetic_sources(y, target_ir=1.1, max_growth=1.0, seed=42):
    """
    Returns (rows, is_synthetic). `rows` indexes into `y`: all original rows first,
    then the source row of each synthetic sample. Synthetic samples are added until
    every IRLbl <= target_ir, or until max_growth * len(y) have been added,
    whichever comes first.
    """
    rng = np.random.default_rng(seed)
    n, num_labels = y.shape
    y_bin = y.astype(bool)
    counts = label_counts(y).astype(float)
    majority = int(counts.argmax())

    pools = []
    for c in range(num_labels):
        pool = np.where(y_bin[:, c] & ~y_bin[:, majority])[0] if c != majority else np.where(y_bin[:, c])[0]
        if len(pool) == 0:
            pool = np.where(y_bin[:, c])[0]
        pools.append(pool)

    budget = int(max_growth * n)
    sources = []
    while len(sources) < budget:
        ratios = counts.max() / np.maximum(counts, 1.0)
        if ratios.max() <= target_ir:
            break
        c = int(ratios.argmax())
        i = int(rng.choice(pools[c]))
        sources.append(i)
        counts += y[i]

    rows = np.concatenate([np.arange(n), np.asarray(sources, dtype=np.int64)])
    is_synthetic = np.arange(len(rows)) >= n
    return rows, is_synthetic


def tune_thresholds(y_true, y_prob, grid=np.linspace(0.05, 0.95, 91)):
    """Per-class decision threshold that maximises F1 on the given (validation) set."""
    thresholds = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, c], y_prob[:, c] >= t, zero_division=0) for t in grid]
        thresholds[c] = grid[int(np.argmax(scores))]
    return thresholds
