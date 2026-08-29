"""
Evaluation: computes AUC + Accuracy per robustness condition (clean,
jpeg_q90/70/50/30, blur_sigma2, crop_80pct, unseen_generator), prints/saves a
compact markdown table matching the slide deck's format, and computes:

    Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust

where AUC_robust = mean AUC across all non-clean, non-unseen-generator
conditions (unseen-generator is reported separately since it measures a
different axis — generalization, not robustness to post-processing — don't
average the two together or you'll hide whichever one is weaker).

Usage:
    python3 src/evaluate.py --config configs/baseline_clip.yaml \
        --checkpoint checkpoints/best.pt --manifest data/robustness_manifest.csv
"""
import argparse
import json

import torch
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from data.datasets import ManifestDataset
from models.detector import HybridDetector
from train import get_clip_norm_stats, to_raw_rgb01

ROBUSTNESS_CONDITIONS = [
    "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "blur_sigma2", "crop_80pct",
]


def eval_condition(
    model,
    manifest_csv,
    condition,
    preprocess,
    clip_mean,
    clip_std,
    device,
    batch_size,
):
    ds = ManifestDataset(manifest_csv, split=condition, preprocess=preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            clip_img = batch["image"].to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
            p = model.predict_proba(clip_img, raw_img)
            probs.extend(p.cpu().tolist())
            labels.extend(batch["label"].tolist())
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, [1 if p >= 0.5 else 0 for p in probs])
    return auc, acc, probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--manifest", default="data/robustness_manifest.csv",
                     help="Manifest whose `split` column holds condition names "
                          "(clean/jpeg_q30/.../unseen_generator) — see "
                          "scripts/build_robustness_testset.py")
    ap.add_argument("--out", default="docs/robustness_results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = HybridDetector(
        clip_model_name=cfg["model"]["clip_model_name"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    clip_mean, clip_std = get_clip_norm_stats(model.semantic.preprocess)

    results = {}
    for cond in ROBUSTNESS_CONDITIONS + ["unseen_generator"]:
        try:
            auc, acc, _, _ = eval_condition(
                model, args.manifest, cond, model.semantic.preprocess, clip_mean,
                clip_std, device, cfg["train"]["batch_size"],
            )
            results[cond] = {"auc": auc, "acc": acc}
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"skipping '{cond}': {e}")

    robust_conditions = [c for c in ROBUSTNESS_CONDITIONS if c != "clean" and c in results]
    auc_clean = results.get("clean", {}).get("auc")
    auc_robust = (
        sum(results[c]["auc"] for c in robust_conditions) / len(robust_conditions)
        if robust_conditions else None
    )
    final_score = (
        0.5 * auc_clean + 0.5 * auc_robust
        if auc_clean is not None and auc_robust is not None else None
    )

    print("\n| Condition | Acc. | AUC |")
    print("|---|---|---|")
    for cond, r in results.items():
        print(f"| {cond} | {r['acc']:.2f} | {r['auc']:.2f} |")
    print(f"\nFinal Score (0.5*AUC_clean + 0.5*AUC_robust) = {final_score}")

    with open(args.out, "w") as f:
        json.dump({"results": results, "final_score": final_score}, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
