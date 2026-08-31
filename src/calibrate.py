"""
Temperature scaling: fit a single scalar temperature on the held-out val set
so predict_proba() outputs are calibrated probabilities, not just a
monotonic ranking score. Needed for thresholding and the error-analysis pass
(calibrated confidence is more useful than AUC alone).

Usage:
    python3 src/calibrate.py --config configs/baseline_clip.yaml --checkpoint checkpoints/best.pt
"""
import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.datasets import ManifestDataset
from models.detector import HybridDetector
from train import get_clip_norm_stats, to_raw_rgb01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HybridDetector(
        clip_model_name=cfg["model"]["clip_model_name"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    for p in model.parameters():
        p.requires_grad = False
    model.raw_temperature.requires_grad = True

    clip_mean, clip_std = get_clip_norm_stats(model.semantic.preprocess)

    val_ds = ManifestDataset(cfg["data"]["manifest_csv"], "val", model.semantic.preprocess, None)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    opt = torch.optim.LBFGS([model.raw_temperature], lr=0.05, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    logits_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            clip_img = batch["image"].to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
            logit = model(clip_img, raw_img)
            logits_all.append(logit)
            labels_all.append(batch["label"].to(device))
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    def closure():
        opt.zero_grad()
        loss = criterion(logits_all / model.temperature(), labels_all)
        loss.backward()
        return loss

    opt.step(closure)
    print(f"Fitted temperature = {model.temperature().item():.4f}")
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved calibrated checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
