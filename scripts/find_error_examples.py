"""
Finds representative false positives and false negatives for the Error
Analysis Note deliverable (5.5.5): "Highlight representative false
positives, false negatives, and any trade-offs in the proposed approach."

Scores every image in the `clean` condition of the robustness manifest
(420 images, balanced real/AI, no post-processing applied -- so any
misclassification here is about the model's core signal, not a transform
robustness failure), then pulls the most confidently-wrong example of
each error type:
  - false positive: label == 0 (real), highest predicted prob (model is
    sure it's AI-generated, and it's wrong)
  - false negative: label == 1 (AI-generated), lowest predicted prob
    (model is sure it's real, and it's wrong)

Copies the top-K of each into docs/examples/ (renamed to carry the
prediction in the filename) and writes docs/error_examples.json with the
full record (path, true label, generator, predicted prob) for every
example plus the total error counts, so the writeup isn't just K
cherry-picked images with no denominator.

Usage:
    python3 scripts/find_error_examples.py --k 4
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
import pandas as pd
from PIL import Image

from models.detector import HybridDetector
from train import get_clip_norm_stats, to_raw_rgb01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--manifest", default="data/robustness_manifest.csv")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--k", type=int, default=4, help="How many FP and FN examples to save.")
    ap.add_argument("--examples_dir", default="docs/examples")
    ap.add_argument("--out", default="docs/error_examples.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridDetector(
        clip_model_name=cfg["model"]["clip_model_name"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    clip_mean, clip_std = get_clip_norm_stats(model.semantic.preprocess)
    preprocess = model.semantic.preprocess

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.condition].reset_index(drop=True)
    print(f"Scoring {len(df)} images from condition={args.condition!r}")

    records = []
    with torch.no_grad():
        for i, row in df.iterrows():
            img = Image.open(row["path"]).convert("RGB")
            clip_img = preprocess(img).unsqueeze(0).to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
            prob = model.predict_proba(clip_img, raw_img).item()
            records.append({
                "path": row["path"],
                "label": int(row["label"]),
                "generator": row["generator"],
                "pred": prob,
            })
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(df)}")

    reals = [r for r in records if r["label"] == 0]
    fakes = [r for r in records if r["label"] == 1]

    fp_all = sorted([r for r in reals if r["pred"] >= 0.5], key=lambda r: -r["pred"])
    fn_all = sorted([r for r in fakes if r["pred"] < 0.5], key=lambda r: r["pred"])

    print(f"\nTotal false positives (real predicted as AI, clean/no transform): {len(fp_all)} / {len(reals)}")
    print(f"Total false negatives (AI predicted as real, clean/no transform): {len(fn_all)} / {len(fakes)}")

    # Per-generator breakdown of false negatives -- is the model missing one
    # generator family more than others, or is it spread evenly?
    generators = sorted(set(r["generator"] for r in fakes))
    fn_by_generator = {}
    for g in generators:
        g_total = sum(1 for r in fakes if r["generator"] == g)
        g_fn = sum(1 for r in fn_all if r["generator"] == g)
        fn_by_generator[g] = {"total": g_total, "false_negatives": g_fn,
                               "rate": round(g_fn / g_total, 4) if g_total else None}
    print("\nFalse negative rate by generator:")
    for g, d in fn_by_generator.items():
        print(f"  {g}: {d['false_negatives']}/{d['total']} ({d['rate']:.1%})")

    examples_dir = Path(args.examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    for old in examples_dir.glob("fp_*.jpg"):
        old.unlink()
    for old in examples_dir.glob("fn_*.jpg"):
        old.unlink()

    def save_examples(items, prefix):
        saved = []
        for rank, r in enumerate(items[: args.k], start=1):
            src = Path(r["path"])
            dst = examples_dir / f"{prefix}_{rank}_{r['generator']}_pred{r['pred']:.2f}.jpg"
            img = Image.open(src).convert("RGB")
            img.thumbnail((512, 512))
            img.save(dst, "JPEG", quality=90)
            r = dict(r)
            r["example_file"] = str(dst)
            saved.append(r)
        return saved

    fp_examples = save_examples(fp_all, "fp")
    fn_examples = save_examples(fn_all, "fn")

    summary = {
        "condition": args.condition,
        "n_real": len(reals),
        "n_fake": len(fakes),
        "n_false_positives": len(fp_all),
        "n_false_negatives": len(fn_all),
        "false_negative_rate_by_generator": fn_by_generator,
        "false_positive_examples": fp_examples,
        "false_negative_examples": fn_examples,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(fp_examples)} FP + {len(fn_examples)} FN example images to {examples_dir}")
    print(f"Wrote full summary to {args.out}")

    print("\n--- FP examples (real, predicted AI) ---")
    for r in fp_examples:
        print(f"  {r['example_file']}: generator={r['generator']} pred={r['pred']:.3f}")
    print("--- FN examples (AI, predicted real) ---")
    for r in fn_examples:
        print(f"  {r['example_file']}: generator={r['generator']} pred={r['pred']:.3f}")


if __name__ == "__main__":
    main()
