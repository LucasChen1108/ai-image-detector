#!/usr/bin/env python3
"""Sanity test for the frequency branch + fusion head forward pass.

Creates a dummy semantic vector, runs the real `FrequencyBranch` on a random
RGB batch, concatenates features, runs a small fusion MLP, and computes
sigmoid probabilities using a learned temperature.

Run with: `.venv/bin/python3 scripts/test_fusion_forward.py`
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the repo root to sys.path so imports work when this runs from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.frequency_branch import FrequencyBranch


def build_fusion(clip_out_dim=256, freq_out_dim=128, fusion_hidden=128, dropout=0.2):
    fusion = nn.Sequential(
        nn.Linear(clip_out_dim + freq_out_dim, fusion_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(fusion_hidden, 1),
    )
    raw_temperature = nn.Parameter(torch.tensor(0.0))
    return fusion, raw_temperature


def softplus(x):
    return F.softplus(x) + 1e-3


def main():
    torch.manual_seed(0)
    device = "cpu"

    B = 4
    H = W = 224

    # Stand in for the embedding that CLIP would normally produce.
    clip_out_dim = 256
    sem = torch.randn(B, clip_out_dim, dtype=torch.float32, device=device)

    # Make a random RGB batch in the [0, 1] range for the frequency branch.
    raw_rgb_01 = torch.rand(B, 3, H, W, dtype=torch.float32, device=device)

    # Build and run the actual frequency branch.
    freq_branch = FrequencyBranch(input_size=H, out_dim=128).to(device)
    freq_feats = freq_branch(raw_rgb_01)

    # Build the fusion head.
    fusion, raw_temp = build_fusion(clip_out_dim=clip_out_dim, freq_out_dim=freq_feats.shape[1])
    fusion = fusion.to(device)

    # Join both outputs and run the forward pass.
    concat = torch.cat([sem, freq_feats], dim=-1)
    logits = fusion(concat).squeeze(-1)
    temp = softplus(raw_temp)
    probs = torch.sigmoid(logits / temp)

    print("sem shape:", sem.shape, "dtype:", sem.dtype)
    print("freq_feats shape:", freq_feats.shape, "dtype:", freq_feats.dtype)
    print("concat shape:", concat.shape)
    print("logits shape:", logits.shape, "sample:", logits[:4].detach().cpu().numpy())
    print("temperature:", float(temp.detach().cpu().item()))
    print("probs shape:", probs.shape, "sample:", probs[:4].detach().cpu().numpy())


if __name__ == "__main__":
    main()
