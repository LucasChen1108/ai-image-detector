"""
Low-level frequency branch: catches GAN/diffusion up-sampling artifacts and
sensor-noise mismatches that live in high-frequency detail — exactly the
signal that survives when CLIP's high-level semantics can't tell (a very
"clean" fake still leaves a spectral fingerprint).

We take the log-magnitude 2D FFT spectrum of the grayscale image and pass it
through a small CNN. This is intentionally simple (not a copy of any single
published detector's exact frequency featurization) — swap for a DCT-block
statistics variant per FREQ_MODE if your team wants to try that ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_magnitude_fft(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, W) grayscale in [0,1]. Returns log-magnitude spectrum,
    shifted so DC sits at the center, normalized to roughly [0,1]."""
    fft = torch.fft.fft2(x)
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    mag = torch.log1p(torch.abs(fft))
    mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return mag


class FrequencyBranch(nn.Module):
    def __init__(self, input_size: int = 224, out_dim: int = 128):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """rgb_01: (B, 3, H, W) in [0,1] — pass the UN-normalized image here,
        not the CLIP-normalized tensor, so the spectrum reflects real pixel
        statistics rather than CLIP's mean/std shift."""
        if rgb_01.shape[-1] != self.input_size:
            rgb_01 = F.interpolate(rgb_01, size=(self.input_size, self.input_size),
                                    mode="bilinear", align_corners=False)
        gray = (0.299 * rgb_01[:, 0] + 0.587 * rgb_01[:, 1] + 0.114 * rgb_01[:, 2]).unsqueeze(1)
        spectrum = log_magnitude_fft(gray)
        feats = self.net(spectrum)
        return self.project(feats)
