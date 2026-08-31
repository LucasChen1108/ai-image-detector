#!/usr/bin/env python3
"""Sanity test for `HybridDetector` end-to-end using a dummy semantic branch.

This avoids downloading CLIP weights by replacing `model.semantic` with a
small module that returns a random semantic embedding of the right shape.

Run with: `.venv/bin/python3 scripts/test_hybrid_forward.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.models.detector import HybridDetector
from src.models.frequency_branch import FrequencyBranch


class DummySemantic(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        # Use a small placeholder instead of the real image preprocessing.
        self.preprocess = lambda img: img

    def forward(self, x):
        B = x.shape[0]
        return torch.randn(B, self.out_dim, dtype=torch.float32, device=x.device)


def softplus(x):
    return nn.functional.softplus(x) + 1e-3


def main():
    device = 'cpu'
    B = 2
    H = W = 224

    model = HybridDetector(clip_model_name='ViT-B-32', clip_pretrained='openai')
    # Use the dummy semantic branch so this test doesn't download CLIP.
    model.semantic = DummySemantic(out_dim=256)
    model.to(device)

    # HybridDetector already creates the frequency branch, so use that real
    # branch without changing the model.

    # DummySemantic ignores clip_input, but the tensor still needs the right shape.
    clip_input = torch.zeros(B, 3, H, W, dtype=torch.float32, device=device)
    raw_rgb_01 = torch.rand(B, 3, H, W, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(clip_input, raw_rgb_01)
        probs = model.predict_proba(clip_input, raw_rgb_01)

    print('logits shape:', logits.shape)
    print('probs shape:', probs.shape)
    print('sample probs:', probs.cpu().numpy())


if __name__ == '__main__':
    main()
