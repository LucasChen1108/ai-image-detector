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
from tqdm import tqdm

from data.augmentations import build_train_transform
from data.datasets import ManifestDataset
from models.detector import HybridDetector


def get_clip_norm_stats(preprocess):
    """Pull CLIP's actual mean/std out of its own preprocess pipeline rather
    than hardcoding OpenAI's constants — different pretrained checkpoints
    (e.g. laion2b vs openai) can use different normalization stats, so this
    stays correct regardless of what configs/baseline_clip.yaml selects."""
    for t in preprocess.transforms:
        if hasattr(t, "mean") and hasattr(t, "std"):
            mean = torch.tensor(t.mean).view(1, 3, 1, 1)
            std = torch.tensor(t.std).view(1, 3, 1, 1)
            return mean, std
    raise ValueError("Could not find a Normalize transform in CLIP preprocess")


def to_raw_rgb01(clip_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert CLIP's normalization to recover an approximate [0,1] RGB image
    from the CLIP-preprocessed tensor. This is what the frequency branch
    needs (it computes an FFT over raw pixel statistics, not CLIP-normalized
    values) without requiring ManifestDataset to return two separate tensors
    per sample. Bonus: both branches then see exactly the same resize/crop,
    which keeps pixel-domain and frequency-domain statistics aligned (the
    DDA insight from the brief)."""
    mean = mean.to(clip_tensor.device)
    std = std.to(clip_tensor.device)
    return (clip_tensor * std + mean).clamp(0.0, 1.0)


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
    clip_mean, clip_std = get_clip_norm_stats(preprocess)
    train_aug = build_train_transform(cfg["data"]["image_size"])

    # Dual-tensor wiring fixed: raw_img is now reconstructed from clip_img via
    # to_raw_rgb01() above (see that docstring) rather than needing
    # ManifestDataset to return two tensors. evaluate.py, calibrate.py, and
    # scripts/check_frequency_branch.py all reuse this same pattern now.
    train_ds = ManifestDataset(cfg["data"]["manifest_csv"], "train", preprocess, train_aug)
    val_ds = ManifestDataset(cfg["data"]["manifest_csv"], "val", preprocess, None)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"].get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"].get("num_workers", 4))

    # The frequency branch is a small CNN starting from random weights, fused
    # against an ~88M-param pretrained CLIP branch that already produces a
    # strong, low-loss signal almost immediately. check_frequency_branch.py's
    # gradient check showed the frequency branch's gradient norm running
    # ~65x smaller than the semantic head's under a single shared LR — with
    # AdamW's per-parameter adaptive scaling this still under-trains it in
    # practice (confirmed by a near-zero with/without ablation delta on the
    # resulting checkpoint), a known "modality imbalance" pattern in
    # multi-branch fusion. Give the frequency branch its own higher LR via a
    # separate param group so its small gradients still translate into
    # meaningful updates over the same epoch budget, rather than changing
    # the architecture itself.
    freq_lr_mult = cfg["train"].get("freq_lr_multiplier", 1.0)
    freq_params = list(model.frequency.parameters())
    freq_param_ids = {id(p) for p in freq_params}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in freq_param_ids]

    base_lr = cfg["train"]["lr"]
    weight_decay = cfg["train"].get("weight_decay", 1e-4)
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": [p for p in freq_params if p.requires_grad],
             "lr": base_lr * freq_lr_mult},
        ],
        weight_decay=weight_decay,
    )
    if freq_lr_mult != 1.0:
        print(f"frequency branch LR = {base_lr * freq_lr_mult:.6f} "
              f"({freq_lr_mult}x base LR {base_lr:.6f})")
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    ckpt_dir = Path(cfg["train"].get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            clip_img = batch["image"].to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
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
                raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
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
