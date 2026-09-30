"""
Batched 1D ECG augmentation, run on the training device.

This is also how step 1's class balancing is realised. Every minority-class
sample added in step 1 is *synthetic*: it is regenerated from its source record
with a fresh random augmentation each time it is drawn, so no epoch ever sees
an exact copy. Synthetic samples always get the time-domain transforms
(p = p_synthetic, default 1.0); original records get them with p = p_original.

Transforms (per sample, independent random parameters):
  1. Amplitude scaling  - one gain for all 12 leads, keeps inter-lead ratios
  2. Time stretch       - resample by a factor in [min_stretch, max_stretch]
                          (heart-rate / cycle-length variation)
  3. Circular shift     - phase jitter up to max_shift samples
  4. Gaussian noise     - low level (noise_std in z-score units, far below what
                          step 2 removed); acts as a regulariser
  5. Temporal cutout    - all leads zeroed together, like a brief electrode dropout
"""

import torch


class BatchECGAugmenter:
    def __init__(
        self,
        min_scale=0.85,
        max_scale=1.15,
        min_stretch=0.9,
        max_stretch=1.1,
        max_shift=250,
        noise_std=0.02,
        cutout_holes=2,
        cutout_max_len=250,
        p_original=0.5,
        p_synthetic=1.0,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.cutout_holes = cutout_holes
        self.cutout_max_len = cutout_max_len
        self.p_original = p_original
        self.p_synthetic = p_synthetic

    @classmethod
    def from_config(cls, config):
        return cls(**config["augmentation"])

    @staticmethod
    def _uniform(B, low, high, device):
        return torch.empty(B, device=device).uniform_(low, high)

    def __call__(self, x, synthetic=None):
        """
        x:         (B, leads, T) float tensor on any device
        synthetic: (B,) bool tensor, True for step-1 synthetic samples (None = all original)
        """
        B, _, T = x.shape
        device = x.device
        if synthetic is None:
            synthetic = torch.zeros(B, dtype=torch.bool, device=device)
        p = torch.where(synthetic, self.p_synthetic, self.p_original).to(device)

        def fires():
            return torch.rand(B, device=device) < p

        # 1. Amplitude scaling
        scale = torch.where(fires(), self._uniform(B, self.min_scale, self.max_scale, device), torch.ones(B, device=device))
        x = x * scale[:, None, None]

        # 2 + 3. Time stretch and circular shift, as one resampling grid with linear interpolation
        stretch = torch.where(fires(), self._uniform(B, self.min_stretch, self.max_stretch, device), torch.ones(B, device=device))
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=device).float()
        shift = torch.where(fires(), shift, torch.zeros_like(shift))
        t = torch.arange(T, device=device, dtype=torch.float32)[None, :]
        src = ((t - T / 2) / stretch[:, None] + T / 2 - shift[:, None]) % T
        i0 = src.floor().long()
        w = (src - i0)[:, None, :]
        i1 = (i0 + 1) % T
        x0 = torch.gather(x, 2, i0[:, None, :].expand_as(x))
        x1 = torch.gather(x, 2, i1[:, None, :].expand_as(x))
        x = x0 * (1 - w) + x1 * w

        # 4. Low-level Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * fires()[:, None, None]

        # 5. Temporal cutout
        keep = torch.ones(B, T, dtype=torch.bool, device=device)
        active = fires()
        for _ in range(self.cutout_holes):
            length = torch.randint(10, self.cutout_max_len + 1, (B,), device=device)
            start = (torch.rand(B, device=device) * (T - length)).long()
            hole = (t >= start[:, None]) & (t < (start + length)[:, None]) & active[:, None]
            keep &= ~hole
        return x * keep[:, None, :]
