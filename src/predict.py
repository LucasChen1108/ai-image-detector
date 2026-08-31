"""
Standalone inference script that takes an image directory as input and
writes a confidence score for each image to a JSON file with image_path and
pred fields.

Unlike train.py/evaluate.py/calibrate.py, this does NOT read a manifest CSV
or require labels -- it just walks a directory of images and scores each
one. `pred` is the model's calibrated probability that the image is
AI-generated (label convention: 0 = real, 1 = AI-generated, matching
ManifestDataset / the rest of the pipeline).

Usage:
    python3 src/predict.py --input_dir path/to/images --out predictions.json
    # Or run it through run.sh:
    bash run.sh predict --input_dir path/to/images --out predictions.json
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.detector import HybridDetector
from train import get_clip_norm_stats, to_raw_rgb01

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageDirDataset(Dataset):
    """Every image file under `input_dir` (recursive), no labels/manifest
    needed. Corrupt/unreadable files fail loudly at __getitem__ rather than
    being silently skipped, since a bad prediction file is worse than a
    crash you can see."""

    def __init__(self, input_dir: str, preprocess):
        self.paths = sorted(
            p for p in Path(input_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise SystemExit(
                f"No images found under {input_dir} "
                f"(looked for extensions: {sorted(IMAGE_EXTENSIONS)})"
            )
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return {"image": self.preprocess(img), "path": str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                     help="Directory of images to score (searched recursively).")
    ap.add_argument("--config", default="configs/baseline_clip.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--batch_size", type=int, default=32)
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
    ds = ImageDirDataset(args.input_dir, model.semantic.preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Scoring {len(ds)} images from {args.input_dir}")

    results = []
    with torch.no_grad():
        for batch in loader:
            clip_img = batch["image"].to(device)
            raw_img = to_raw_rgb01(clip_img, clip_mean, clip_std)
            probs = model.predict_proba(clip_img, raw_img)
            for path, prob in zip(batch["path"], probs.cpu().tolist()):
                results.append({"image_path": path, "pred": prob})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
