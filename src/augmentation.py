"""
Batched 1D augmentation, run on the training device.

Oversampling (step 1) duplicates minority-class records; augmenting every batch
means each duplicate is seen as a slightly different signal rather than an exact
copy, which is what keeps oversampling from turning into memorisation.

No Gaussian noise is added: the inputs have just been denoised in step 2.
"""

import torch


class BatchECGAugmenter:
    def __init__(self, min_scale=0.85, max_scale=1.15, max_shift=250, cutout_holes=2, cutout_max_len=250, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.cutout_holes = cutout_holes
        self.cutout_max_len = cutout_max_len
        self.p = p

    @classmethod
    def from_config(cls, config):
        return cls(**config["augmentation"])

    def _apply_mask(self, B, device):
        return torch.rand(B, device=device) < self.p

    def __call__(self, x):
        """x: (B, leads, T) on any device."""
        B, _, T = x.shape
        device = x.device

        # Amplitude scaling (same factor across leads, preserving inter-lead ratios)
        scale = torch.empty(B, device=device).uniform_(self.min_scale, self.max_scale)
        scale = torch.where(self._apply_mask(B, device), scale, torch.ones_like(scale))
        x = x * scale[:, None, None]

        # Circular time shift
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=device)
        shift = torch.where(self._apply_mask(B, device), shift, torch.zeros_like(shift))
        idx = (torch.arange(T, device=device)[None, :] - shift[:, None]) % T
        x = torch.gather(x, 2, idx[:, None, :].expand_as(x))

        # Temporal cutout (all leads zeroed together, like a brief electrode dropout)
        t = torch.arange(T, device=device)[None, :]
        keep = torch.ones(B, T, dtype=torch.bool, device=device)
        active = self._apply_mask(B, device)
        for _ in range(self.cutout_holes):
            length = torch.randint(10, self.cutout_max_len + 1, (B,), device=device)
            start = (torch.rand(B, device=device) * (T - length)).long()
            hole = (t >= start[:, None]) & (t < (start + length)[:, None]) & active[:, None]
            keep &= ~hole
        return x * keep[:, None, :]
