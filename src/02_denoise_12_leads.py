"""
STEP 2: Denoise all 12 leads of every PTB-XL record at 500 Hz.

Reads each WFDB record, runs ECGDenoiser (HP 0.5 Hz, LP 40 Hz, notch 50 Hz,
sym8 wavelet, z-score) and writes one cache of shape (N, 12, 5000) in float16
(~2.6 GB) that step 3 trains from. Also writes a raw-vs-denoised plot and a
controlled-noise check so the denoising can be inspected before training.

Outputs:
  data/processed/signals_500hz_12lead_denoised.npy   (N, 12, 5000) float16
  data/processed/signals_500hz_12lead_ecg_ids.npy    (N,) row order of the cache
  artifacts/denoising/raw_vs_denoised_ecg_<id>.png
  docs/results/denoising_noise_robustness.csv

Usage:
  PYTHONPATH=. python3 src/02_denoise_12_leads.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from src.data import load_config, load_metadata, processed_path
from src.denoising import ECGDenoiser

_DENOISER = None


def _init_worker(config):
    global _DENOISER
    _DENOISER = ECGDenoiser.from_config(config)


def _read_raw(path):
    sig, _ = wfdb.rdsamp(path)
    return sig.astype(np.float32)  # (5000, 12) mV


def _denoise_record(path):
    return _DENOISER.process(_read_raw(path)).T.astype(np.float16)  # (12, 5000)


def add_controlled_noise(clean, fs, snr_db, rng):
    """AWGN at `snr_db` + 0.2 Hz baseline wander + 50 Hz powerline hum."""
    t = np.arange(clean.shape[0]) / fs
    power = np.mean(clean ** 2, axis=0, keepdims=True)
    awgn = rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), size=clean.shape)
    wander = 0.15 * np.sin(2 * np.pi * 0.2 * t)[:, None]
    hum = 0.08 * np.sin(2 * np.pi * 50.0 * t)[:, None]
    return clean + awgn + wander + hum


def noise_robustness(df, config, n_records=100, snr_db=10.0, seed=0):
    """
    How much does injected noise change the denoiser's output?
    Reference = denoiser(raw); test = denoiser(raw + noise). Both unnormalised.
    """
    rng = np.random.default_rng(seed)
    denoiser = ECGDenoiser.from_config(config)
    fs = config["data"]["sampling_rate"]
    rows = []
    for ecg_id in rng.choice(df.index.values, size=n_records, replace=False):
        raw = _read_raw(os.path.join(config["data"]["raw_dir"], df.at[ecg_id, "filename_hr"])).astype(np.float64)
        ref = denoiser.process(raw, normalize=False)
        noisy = add_controlled_noise(raw, fs, snr_db, rng)
        out = denoiser.process(noisy, normalize=False)
        ref_c = ref - ref.mean(axis=0)
        snr_in = 10 * np.log10(np.sum(ref_c ** 2) / np.sum((noisy - raw) ** 2))
        snr_out = 10 * np.log10(np.sum(ref_c ** 2) / np.sum((out - ref) ** 2))
        r = np.mean([np.corrcoef(ref[:, i], out[:, i])[0, 1] for i in range(ref.shape[1]) if ref[:, i].std() > 0])
        rows.append({"ecg_id": ecg_id, "SNR_in_dB": snr_in, "SNR_out_dB": snr_out, "Pearson_r": r})
    res = pd.DataFrame(rows)
    res["SNR_gain_dB"] = res.SNR_out_dB - res.SNR_in_dB
    return res


def plot_raw_vs_denoised(raw, den, leads, ecg_id, fs, path):
    t = np.arange(raw.shape[0]) / fs
    raw_z = (raw - raw.mean(0)) / np.where(raw.std(0) == 0, 1, raw.std(0))
    fig, axes = plt.subplots(len(leads), 1, figsize=(15, 22), sharex=True)
    fig.suptitle(f"ECG {ecg_id} @ {fs} Hz: raw (z-scored, red) vs denoised (navy)", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes):
        ax.plot(t, raw_z[:, i], color="#e63946", alpha=0.5, lw=0.9)
        ax.plot(t, den[:, i], color="#1d3557", lw=1.1)
        ax.set_ylabel(leads[i], fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    config = load_config()
    df = load_metadata(config)
    raw_dir = config["data"]["raw_dir"]
    leads = config["data"]["leads"]
    seq_len = config["data"]["seq_len"]
    fs = config["data"]["sampling_rate"]

    print("=" * 72)
    print("STEP 2: DENOISING ALL 12 LEADS @ 500 Hz")
    print("=" * 72)

    # Quality checks first, so a bad filter setting shows up before the long run
    plot_dir = os.path.join(config["output"]["artifacts_dir"], "denoising")
    os.makedirs(plot_dir, exist_ok=True)
    denoiser = ECGDenoiser.from_config(config)
    example_id = df.index.values[0]
    raw = _read_raw(os.path.join(raw_dir, df.at[example_id, "filename_hr"]))
    plot_raw_vs_denoised(raw, denoiser.process(raw), leads, example_id, fs,
                         os.path.join(plot_dir, f"raw_vs_denoised_ecg_{example_id}.png"))

    robustness = noise_robustness(df, config)
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    robustness.to_csv(os.path.join(config["output"]["results_dir"], "denoising_noise_robustness.csv"), index=False)
    print("Controlled-noise check (100 records, AWGN 10 dB + wander + 50 Hz hum):")
    print(robustness[["SNR_in_dB", "SNR_out_dB", "SNR_gain_dB", "Pearson_r"]].describe().loc[["mean", "std"]].round(3).to_string())

    # Full cache
    cache_path = processed_path(config, "signals_500hz_12lead_denoised.npy")
    ids_path = processed_path(config, "signals_500hz_12lead_ecg_ids.npy")
    if os.path.exists(cache_path) and os.path.exists(ids_path):
        print(f"\nCache already exists at {cache_path}; delete it to rebuild.")
        return

    paths = [os.path.join(raw_dir, f) for f in df.filename_hr]
    tmp_path = cache_path + ".partial.npy"
    cache = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(len(paths), len(leads), seq_len))

    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init_worker, initargs=(config,)) as ex:
        for i, sig in enumerate(tqdm(ex.map(_denoise_record, paths, chunksize=64), total=len(paths), desc="Denoising")):
            cache[i] = sig
    cache.flush()
    del cache
    os.replace(tmp_path, cache_path)
    np.save(ids_path, df.index.values)
    print(f"\nSaved denoised cache {cache_path} ({len(paths)}, {len(leads)}, {seq_len}) float16")


if __name__ == "__main__":
    main()
