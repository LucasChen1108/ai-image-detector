"""
Training entrypoint.

Usage:
    python3 src/train.py --config configs/baseline_clip.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.augmentations import build_train_transform
from data.datasets import ManifestDataset
from models.detector import HybridDetector


def to_raw_rgb01(pil_batch_tensor):
    """The dataset's `preprocess` from open_clip already resizes+normalizes
    for CLIP. The frequency branch instead wants plain [0,1] RGB, so the
    dataset returns the CLIP-preprocessed tensor and we separately recompute
    a [0,1] view here by un-normalizing using CLIP's known mean/std.
    Simpler alternative used here: dataset stores PIL->ToTensor() output
    alongside the CLIP tensor. See datasets.py note if you extend this."""
    return pil_batch_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridDetector(
        clip_model_name=cfg["model"]["clip_model_name"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
        unfreeze_last_n_blocks=cfg["model"].get("unfreeze_last_n_blocks", 0),
    ).to(device)

    preprocess = model.semantic.preprocess
    train_aug = build_train_transform(cfg["data"]["image_size"])

    def clip_and_raw(img):
        img = train_aug(img) if cfg["data"].get("apply_train_aug", True) else img
        clip_tensor = preprocess(img)
        raw_tensor = ToTensor()(img.resize((cfg["data"]["image_size"],) * 2))
        return clip_tensor, raw_tensor

    # NOTE: ManifestDataset currently returns a single "image" tensor via
    # `preprocess`. For the two-branch model, swap `preprocess=clip_and_raw`
    # and adapt __getitem__ to return both tensors — left as a Day-1 wiring
    # task for whoever owns training (see docs/ROLES.md, Track B).
    train_ds = ManifestDataset(cfg["data"]["manifest_csv"], "train", preprocess, train_aug)
    val_ds = ManifestDataset(cfg["data"]["manifest_csv"], "val", preprocess, None)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"].get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"].get("num_workers", 4))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    ckpt_dir = Path(cfg["train"].get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            clip_img = batch["image"].to(device)
            raw_img = batch["image"].to(device)  # placeholder: see clip_and_raw note above
            labels = batch["label"].to(device)

            logit = model(clip_img, raw_img)
            loss = criterion(logit, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

        # --- validation AUC each epoch ---
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                clip_img = batch["image"].to(device)
                raw_img = batch["image"].to(device)
                probs = model.predict_proba(clip_img, raw_img)
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(batch["label"].tolist())
        auc = roc_auc_score(all_labels, all_probs)
        print(f"epoch {epoch}: val AUC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  saved new best checkpoint (AUC={auc:.4f})")

    print(f"Training done. Best val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
