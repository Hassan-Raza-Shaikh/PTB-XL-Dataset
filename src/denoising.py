"""
12-lead ECG denoiser (same chain as ECGPreprocessor in the ECG-Dataset repo):

  1. High-pass Butterworth 0.5 Hz   -> baseline wander
  2. Low-pass Butterworth 40 Hz     -> EMG / muscle noise
  3. IIR notch 50 Hz                -> powerline hum
  4. DWT soft-thresholding (sym8)   -> residual broadband noise
  5. Per-lead z-score

Filters are applied as second-order sections (sosfiltfilt). At 500 Hz a 5th-order
0.5 Hz high-pass in (b, a) form has poles at |z| ~ 0.998, which is numerically
fragile; SOS gives the same zero-phase response without that risk. Wavelet
decomposition runs on all leads at once (axis=0) with a per-lead noise estimate.
"""

import numpy as np
import pywt
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


class ECGDenoiser:
    def __init__(
        self,
        sampling_rate=500,
        highpass_cutoff=0.5,
        lowpass_cutoff=40.0,
        notch_freq=50.0,
        notch_q=30.0,
        filter_order=5,
        wavelet_name="sym8",
        wavelet_level=5,
        normalize="zscore",
    ):
        self.fs = sampling_rate
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.normalize_method = normalize

        self.sos_hp = butter(filter_order, highpass_cutoff, btype="highpass", fs=sampling_rate, output="sos") if highpass_cutoff > 0 else None
        self.sos_lp = butter(filter_order, lowpass_cutoff, btype="lowpass", fs=sampling_rate, output="sos") if 0 < lowpass_cutoff < sampling_rate / 2 else None
        self.sos_notch = tf2sos(*iirnotch(notch_freq, notch_q, fs=sampling_rate)) if 0 < notch_freq < sampling_rate / 2 else None

    @classmethod
    def from_config(cls, config):
        d = config["denoising"]
        return cls(
            sampling_rate=config["data"]["sampling_rate"],
            highpass_cutoff=d["highpass_cutoff"],
            lowpass_cutoff=d["lowpass_cutoff"],
            notch_freq=d["notch_freq"],
            notch_q=d["notch_q"],
            filter_order=d["filter_order"],
            wavelet_name=d["wavelet_name"],
            wavelet_level=d["wavelet_level"],
            normalize=d["normalize"],
        )

    def filter(self, sig):
        """sig: (time_steps, num_leads)"""
        for sos in (self.sos_hp, self.sos_lp, self.sos_notch):
            if sos is not None:
                sig = sosfiltfilt(sos, sig, axis=0)
        return sig

    def wavelet_denoise(self, sig):
        coeffs = pywt.wavedec(sig, self.wavelet_name, level=self.wavelet_level, axis=0)
        finest = coeffs[-1]
        sigma = np.median(np.abs(finest - np.median(finest, axis=0)), axis=0) / 0.6745  # (num_leads,)
        thresh = sigma * np.sqrt(2 * np.log(sig.shape[0]))
        denoised = [coeffs[0]] + [
            np.sign(c) * np.maximum(np.abs(c) - thresh, 0.0) for c in coeffs[1:]
        ]
        out = pywt.waverec(denoised, self.wavelet_name, axis=0)[: sig.shape[0]]
        # A flat lead has sigma == 0; its filtered signal passes through unchanged.
        flat = sigma <= 1e-8
        out[:, flat] = sig[:, flat]
        return out

    def normalize(self, sig):
        if self.normalize_method == "zscore":
            std = sig.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (sig - sig.mean(axis=0, keepdims=True)) / std
        return sig

    def process(self, sig, normalize=True):
        """(time_steps, num_leads) raw mV -> denoised (time_steps, num_leads) float32."""
        sig = np.nan_to_num(np.asarray(sig, dtype=np.float64))
        sig = self.wavelet_denoise(self.filter(sig))
        if normalize:
            sig = self.normalize(sig)
        return sig.astype(np.float32)
