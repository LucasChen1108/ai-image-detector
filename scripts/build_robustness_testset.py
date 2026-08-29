"""
Materializes physical copies of the clean test set under each robustness
condition (JPEG q90/70/50/30, blur sigma=2, crop 80%), plus writes a combined
manifest whose `split` column holds the condition name — this is what
src/evaluate.py reads.

This is included (per competition rules: "include generation scripts for
reproducibility") so anyone can regenerate the exact robustness test set
byte-for-byte.

Usage:
    python3 scripts/build_robustness_testset.py \
        --source_manifest manifest/manifest.csv \
        --out_dir data/robustness_testset \
        --out_manifest data/robustness_manifest.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.augmentations import EVAL_CONDITIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_manifest", default="manifest/manifest.csv")
    ap.add_argument("--out_dir", default="data/robustness_testset")
    ap.add_argument("--out_manifest", default="data/robustness_manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.source_manifest)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cond_name, transform in EVAL_CONDITIONS.items():
        cond_dir = out_dir / cond_name
        cond_dir.mkdir(exist_ok=True)
        for i, row in test_df.iterrows():
            img = Image.open(row["path"]).convert("RGB")
            transformed = transform(img)
            out_path = cond_dir / f"{i}.jpg"
            transformed.save(out_path, quality=95)
            rows.append({
                "path": str(out_path), "label": row["label"],
                "generator": row["generator"], "split": cond_name,
            })

    # unseen_generator: reuse the *original* clean images from whatever
    # source the data-prep manifest already held out of training entirely
    # (fetch_dataset.py tags these rows split="heldout" — SID_Set is the
    # held-out generator in the current WildFake+SID_Set manifest). We map
    # that manifest split name to the "unseen_generator" condition name that
    # evaluate.py looks for, since the two scripts used different vocab for
    # the same concept.
    held = df[df["split"] == "heldout"]
    if held.empty:
        print("warning: no rows with split=='heldout' in source manifest — "
              "'unseen_generator' condition will be skipped by evaluate.py")
    for _, row in held.iterrows():
        rows.append({
            "path": row["path"], "label": row["label"],
            "generator": row["generator"], "split": "unseen_generator",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_manifest, index=False)
    print(f"Wrote {len(out_df)} rows across {out_df['split'].nunique()} conditions "
          f"to {args.out_manifest}")


if __name__ == "__main__":
    main()
