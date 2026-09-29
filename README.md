# PTB-XL: Class Balancing → Denoising → 12-Lead Vision Transformer

Multi-label classification of the 5 PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP)
from **all 12 leads at 500 Hz** (5,000 samples per lead). The pipeline runs in three steps,
in this order:

1. **Resolve class imbalance** (multi-label random oversampling on the training folds)
2. **Denoise** all 12 leads (Butterworth HP/LP, 50 Hz notch, sym8 wavelet, z-score)
3. **Train a 1D Vision Transformer** on the balanced, denoised 12-lead signals

## Quick start

```bash
pip install -r requirements.txt

PYTHONPATH=. python3 src/00_download_dataset.py        # PTB-XL v1.0.3 from PhysioNet (skip if you have it)
PYTHONPATH=. python3 src/01_resolve_class_imbalance.py # seconds
PYTHONPATH=. python3 src/02_denoise_12_leads.py        # builds a ~2.6 GB denoised cache
PYTHONPATH=. python3 src/03_train_vit_12_leads.py      # trains + evaluates on fold 10
```

If you already have PTB-XL, point `data.raw_dir`, `data.database_csv`, and `data.scp_csv` in
[`configs/config.yaml`](configs/config.yaml) at it, or symlink it to `data/raw/ptbxl`.
All hyperparameters live in that config file.

Hardware: step 3 picks CUDA, then Apple MPS, then CPU automatically. Step 3 loads the whole
denoised cache into RAM (~2.6 GB float16).

## Splits

The official PTB-XL stratified folds: **train = folds 1–8, validation = fold 9, test = fold 10**.
Only the training folds are resampled. Validation and test keep the real class
distribution, so the reported metrics are honest.

## Step 1: Class imbalance

PTB-XL is **multi-label**: 4,068 records have two superclasses and 1,076 have three or more.
Single-label tools such as SMOTE or a class-balanced sampler don't apply directly.
Training-fold label counts:

| | NORM | MI | STTC | CD | HYP | MeanIR |
|---|---:|---:|---:|---:|---:|---:|
| Before | 7,596 | 4,379 | 4,186 | 3,907 | 2,119 | 2.02 |
| After ML-ROS | 7,596 | 6,907 | 7,264 | 6,908 | 6,906 | 1.07 |

(17,418 → 23,962 training records.)

Method ([`src/imbalance.py`](src/imbalance.py)):

- The imbalance ratio per label is `IRLbl(c) = max_count / count(c)`, and `MeanIR` is its mean (Charte et al., 2015).
- **Greedy ML-ROS**: repeatedly clone a training record that carries the label with the highest
  IRLbl. Records that also carry the majority label (NORM) are avoided, so balancing HYP doesn't
  inflate NORM. It stops when every IRLbl ≤ `target_ir` (1.1) or the set has grown by `max_growth`.
- Clones are never exact repeats in training: step 3 applies random amplitude scaling, time
  shift, and cutout to every batch.
- **Per-class decision thresholds** are tuned on validation to maximise F1. They are then applied
  once to test. The test metrics are reported with both 0.5 and tuned thresholds.

Oversampling happens at the index level (`balanced_train_ecg_ids.npy` lists cloned ids more than once).
Denoising is deterministic per record, so denoising each unique record once and reading it
through the balanced index is equivalent to denoising the oversampled set.

## Step 2: Denoising (all 12 leads, 500 Hz)

[`src/denoising.py`](src/denoising.py). This is the same chain as `ECGPreprocessor` in the
ECG-Dataset repo (outputs match it with correlation 1.000):

1. High-pass Butterworth, 0.5 Hz, order 5 (baseline wander)
2. Low-pass Butterworth, 40 Hz, order 5 (EMG / muscle noise)
3. IIR notch, 50 Hz, Q = 30 (powerline hum)
4. DWT soft thresholding, `sym8`, level 5, universal threshold with a per-lead MAD noise estimate
5. Per-lead z-score

The filters run as second-order sections (`sosfiltfilt`). At 500 Hz, the 0.5 Hz high-pass in
`(b, a)` form has poles at |z| ≈ 0.998, which is numerically fragile.

Step 2 also writes a raw-vs-denoised 12-lead plot and runs a controlled-noise check: it injects
10 dB AWGN, 0.2 Hz wander, and 50 Hz hum into 100 records and measures how close the denoised
output stays to the denoised clean record. On a 3-record spot check this gave about +13.5 dB
SNR gain with Pearson r ≈ 0.98.

## Step 3: 1D Vision Transformer on 12 leads

[`src/vit.py`](src/vit.py):

```
12 leads × 5000 samples
 → Conv1d patch embedding (12 → 192, kernel 100 = 0.2 s, stride 50 = 50% overlap) → 99 tokens
 → [CLS] + learnable positional embeddings
 → 6 × Pre-LN transformer encoder layers (6 heads, MLP ratio 2, dropout 0.1)
 → LayerNorm([CLS]) → Linear → 5 logits (sigmoid, multi-label)
```

About 2.0 M parameters. Each token mixes all 12 leads over a 0.2 s window.

Training setup:

- Loss: `BCEWithLogitsLoss`. There is no `pos_weight`, because balancing is already done in step 1.
- Optimizer: AdamW (lr 5e-4, weight decay 0.05), 2-epoch linear warmup, then cosine decay.
- Up to 40 epochs, with early stopping on validation macro ROC-AUC (patience 8).
- Batch augmentation on the device: amplitude ×[0.85, 1.15], circular shift ±0.5 s, and 2 cutouts
  up to 0.5 s. No added noise, since the input was just denoised.

## Outputs

| File | Step |
|---|---|
| `data/processed/balanced_train_ecg_ids.npy` | 1 |
| `docs/results/class_distribution_before_after.csv`, `artifacts/imbalance/*.png` | 1 |
| `data/processed/signals_500hz_12lead_denoised.npy` (N, 12, 5000) float16 + `..._ecg_ids.npy` | 2 |
| `docs/results/denoising_noise_robustness.csv`, `artifacts/denoising/*.png` | 2 |
| `artifacts/models/vit_12lead_500hz_best.pth` | 3 |
| `docs/results/vit_12lead_500hz_summary.csv` (macro metrics, thresholds 0.5 vs tuned) | 3 |
| `docs/results/vit_12lead_500hz_metrics.csv` (per class), `..._thresholds.csv` | 3 |
| `artifacts/models/vit_12lead_500hz_roc_curves.png`, `..._training_history.png` | 3 |

## Reference results to compare against (ECG-Dataset repo, test fold 10)

| Model | Leads | Rate | Balancing | Denoising | Macro AUC | Macro F1 |
|---|---|---|---|---|---:|---:|
| Overlapping ViT + augmentation | 4 (aVF, III, I, II) | 500 Hz | none | z-score only | 0.8627 | 0.5953 |
| ViT, distinct patches | 4 (aVF, III, I, II) | 100 Hz | none | full | 0.8400 | 0.5765 |
| ResNet1D | 12 | 100 Hz | none | full | 0.9079 | 0.6696 |

The weakest class in those runs was HYP (recall 0.18–0.21 at threshold 0.5). Step 1's
balancing and threshold tuning target exactly that class.

## Layout

```
configs/config.yaml               all paths & hyperparameters
src/
  00_download_dataset.py          step 0: fetch PTB-XL
  01_resolve_class_imbalance.py   step 1: ML-ROS on train folds
  02_denoise_12_leads.py          step 2: denoise + cache + quality checks
  03_train_vit_12_leads.py        step 3: train / tune thresholds / evaluate ViT
  data.py                         metadata, superclass labels, fold splits
  imbalance.py                    IRLbl / MeanIR, ML-ROS, threshold tuning
  denoising.py                    ECGDenoiser
  augmentation.py                 batched on-device augmentation
  vit.py                          ECGViT1D
```
