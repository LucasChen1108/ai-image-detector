"""
Diagnostic for whoever owns the frequency branch (frequency_branch.py +
the fusion head in detector.py): a completed training run proves the CODE
doesn't crash, but says nothing about whether the frequency branch is
actually CONTRIBUTING to predictions. The CLIP branch is an ~88M-parameter
pretrained model that's already good at this; the frequency branch is a
small CNN starting from random weights. It's entirely possible the fusion
layer learns to ignore it, especially on an easy set like CIFAKE, and a
great val AUC would hide that completely.

Two checks:

  1. GRADIENT CHECK -- does the frequency branch receive real learning
     signal at all? Runs one batch, one backward pass, reports the
     gradient norm on frequency-branch params vs. the semantic branch's
     trainable projection head, so you can see if they're in the same
     ballpark or if one is orders of magnitude smaller (a sign it's not
     learning much).

  2. ABLATION -- with a trained checkpoint, compares val AUC normally
     against val AUC with the frequency embedding zeroed out right before
     fusion. If AUC barely moves, the fusion layer isn't using the
     frequency branch's output in any way that matters. This does NOT
     modify detector.py -- it replicates the forward pass here so the
     check stays self-contained and doesn't touch shared model code.

Usage:
    # This only checks gradients, so it uses fresh random weights.
    python3 scripts/check_frequency_branch.py --config configs/baseline_clip.yaml --mode gradient

    # This ablation needs a trained checkpoint.
    python3 scripts/check_frequency_branch.py --config configs/baseline_clip.yaml --mode ablation --checkpoint checkpoints/best.pt

    # Compare one condition from the robustness manifest, such as the
    # held-out-generator rows. Build the test set first if it isn't ready.
    python3 scripts/check_frequency_branch.py --mode ablation --checkpoint checkpoints/best.pt \
        --manifest data/robustness_manifest.csv --split unseen_generator
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.datasets import ManifestDataset  # noqa: E402
from models.detector import HybridDetector  # noqa: E402
from train import get_clip_norm_stats, to_raw_rgb01  # noqa: E402


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def run_gradient_check(model, loader, device, clip_mean, clip_std):
    print("\n=== GRADIENT CHECK ===")
    model.train()
    batch = next(iter(loader))
    clip_img = batch["image"].to(device)
    raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
    labels = batch["label"].to(device)

    logit = model(clip_img, raw_img)
    loss = nn.BCEWithLogitsLoss()(logit, labels)
    model.zero_grad()
    loss.backward()

    freq_norm = grad_norm(model.frequency.parameters())
    sem_head_norm = grad_norm(model.semantic.project.parameters())
    fusion_norm = grad_norm(model.fusion.parameters())

    print(f"frequency branch grad norm:        {freq_norm:.6f}")
    print(f"semantic projection head grad norm: {sem_head_norm:.6f}")
    print(f"fusion head grad norm:               {fusion_norm:.6f}")

    if freq_norm == 0.0:
        print("\n!! FAIL: frequency branch got ZERO gradient. It is not learning at all.")
    elif freq_norm < sem_head_norm * 0.01:
        print(f"\n?? WARNING: frequency branch gradient is >100x smaller than the semantic "
              f"head's. It may be learning too slowly to ever matter.")
    else:
        print("\nOK: frequency branch is receiving a comparable-magnitude gradient signal.")


def run_ablation(model, loader, device, clip_mean, clip_std):
    print("\n=== ABLATION: with vs. without the frequency branch ===")
    model.eval()
    probs_full, probs_no_freq, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            clip_img = batch["image"].to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)

            sem = model.semantic(clip_img)
            freq = model.frequency(raw_img)

            logit_full = model.fusion(torch.cat([sem, freq], dim=-1)).squeeze(-1)
            logit_no_freq = model.fusion(torch.cat([sem, torch.zeros_like(freq)], dim=-1)).squeeze(-1)

            probs_full.extend(torch.sigmoid(logit_full / model.temperature()).cpu().tolist())
            probs_no_freq.extend(torch.sigmoid(logit_no_freq / model.temperature()).cpu().tolist())
            labels_all.extend(batch["label"].tolist())

    auc_full = roc_auc_score(labels_all, probs_full)
    auc_no_freq = roc_auc_score(labels_all, probs_no_freq)
    delta = auc_full - auc_no_freq

    print(f"AUC with frequency branch:    {auc_full:.4f}")
    print(f"AUC with frequency zeroed:    {auc_no_freq:.4f}")
    print(f"delta:                        {delta:+.4f}")

    if abs(delta) < 0.005:
        print("\n?? WARNING: near-zero delta. The fusion head is barely using the frequency "
              "branch's output -- the hybrid design may not be doing real work yet.")
    else:
        print("\nOK: zeroing the frequency branch measurably changes predictions, "
              "so it's contributing to the fused result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    ap.add_argument("--checkpoint", default=None,
                     help="required for --mode ablation; omit for --mode gradient to use fresh weights")
    ap.add_argument("--mode", choices=["gradient", "ablation", "both"], default="both")
    ap.add_argument("--manifest", default=None,
                     help="manifest CSV to evaluate against; defaults to data.manifest_csv "
                          "in --config. Point this at data/robustness_manifest.csv to check "
                          "e.g. the unseen_generator condition instead of the training val split.")
    ap.add_argument("--split", default="val",
                     help='split column value to filter to (default "val"; use '
                          '"unseen_generator" together with --manifest data/robustness_manifest.csv)')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridDetector(
        clip_model_name=cfg["model"]["clip_model_name"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    elif args.mode in ("ablation", "both"):
        raise SystemExit("--mode ablation (or both) needs --checkpoint — a trained model, "
                          "not fresh random weights, or the comparison is meaningless.")

    manifest_csv = args.manifest or cfg["data"]["manifest_csv"]
    print(f"Evaluating against manifest={manifest_csv} split={args.split}")
    val_ds = ManifestDataset(manifest_csv, args.split, model.semantic.preprocess, None)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    clip_mean, clip_std = get_clip_norm_stats(model.semantic.preprocess)

    if args.mode in ("gradient", "both"):
        run_gradient_check(model, val_loader, device, clip_mean, clip_std)
    if args.mode in ("ablation", "both"):
        run_ablation(model, val_loader, device, clip_mean, clip_std)


if __name__ == "__main__":
    main()
