"""
Fusion head: concatenate CLIP-semantic embedding + frequency embedding,
pass through a small MLP to a single real-vs-fake logit, with a learned
temperature for calibrated probabilities (needed for thresholding + the
error-analysis pass, not just a raw label).
"""
import torch
import torch.nn as nn

from .clip_backbone import ClipSemanticBranch
from .frequency_branch import FrequencyBranch


class HybridDetector(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", clip_pretrained="openai",
                 unfreeze_last_n_blocks=0, clip_out_dim=256, freq_out_dim=128,
                 fusion_hidden=128, dropout=0.2):
        super().__init__()
        self.semantic = ClipSemanticBranch(
            clip_model_name, clip_pretrained, unfreeze_last_n_blocks, clip_out_dim
        )
        self.frequency = FrequencyBranch(out_dim=freq_out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(clip_out_dim + freq_out_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )
        # Learned temperature for calibration; keep >0 via softplus.
        self.raw_temperature = nn.Parameter(torch.tensor(0.0))

    def temperature(self):
        return nn.functional.softplus(self.raw_temperature) + 1e-3

    def forward(self, clip_input: torch.Tensor, raw_rgb_01: torch.Tensor):
        sem = self.semantic(clip_input)
        freq = self.frequency(raw_rgb_01)
        logit = self.fusion(torch.cat([sem, freq], dim=-1)).squeeze(-1)
        return logit

    def predict_proba(self, clip_input: torch.Tensor, raw_rgb_01: torch.Tensor):
        logit = self.forward(clip_input, raw_rgb_01)
        return torch.sigmoid(logit / self.temperature())
