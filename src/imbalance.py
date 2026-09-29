"""
Multi-label class-imbalance handling for PTB-XL superclasses.

PTB-XL is multi-label (a record can be MI + STTC + HYP at once), so single-label
tools such as SMOTE or a class-balanced sampler do not apply directly. We use the
imbalance measures from Charte et al. (2015):

    IRLbl(c) = max_k count(k) / count(c)      (1.0 for the majority label)
    MeanIR   = mean_c IRLbl(c)

and a greedy variant of ML-ROS (multi-label random oversampling): repeatedly clone
a training record that carries the label with the highest IRLbl. Records that also
carry the majority label are avoided when possible, so balancing a minority label
does not inflate the majority at the same time.
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


def multilabel_random_oversample(y, target_ir=1.1, max_growth=1.0, seed=42):
    """
    Returns an array of row indices into `y` (original rows first, then clones)
    whose label counts have every IRLbl <= target_ir, or which has grown by
    max_growth * len(y) extra rows, whichever comes first.
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
    clones = []
    while len(clones) < budget:
        ratios = counts.max() / np.maximum(counts, 1.0)
        if ratios.max() <= target_ir:
            break
        c = int(ratios.argmax())
        i = int(rng.choice(pools[c]))
        clones.append(i)
        counts += y[i]

    return np.concatenate([np.arange(n), np.asarray(clones, dtype=np.int64)])


def tune_thresholds(y_true, y_prob, grid=np.linspace(0.05, 0.95, 91)):
    """Per-class decision threshold that maximises F1 on the given (validation) set."""
    thresholds = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, c], y_prob[:, c] >= t, zero_division=0) for t in grid]
        thresholds[c] = grid[int(np.argmax(scores))]
    return thresholds
