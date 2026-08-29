"""
Build the CIFAKE smoke-test manifest.

CIFAKE ships pre-labelled by directory (REAL/ vs FAKE/) and pre-split
(train/ vs test/), so there is nothing to label by hand — this script just
indexes it into the repo's manifest schema:

    path,label,generator,split

Design notes (matter if you extend this to WildFake):

  * We RESPECT CIFAKE's own train/test split rather than pooling everything
    and reshuffling. CIFAR-10 (the source of the real images) has known
    near-duplicates; a global shuffle can land a near-duplicate pair on both
    sides of the train/test line and quietly inflate AUC. The dataset
    authors' split already separates them.
    train/ -> "train" + "val"   (val carved off by --val-frac)
    test/  -> "test"

  * File lists are SORTED before sampling. Path.iterdir() returns
    filesystem order, which differs across machines, so sorting + a fixed
    --seed is what makes this manifest byte-identical for every teammate.
    That is the whole point of committing the manifest instead of having
    each person rebuild their own.

  * Paths are written repo-relative so they resolve on any clone.

CAVEAT: CIFAKE images are 32x32 and CLIP upsamples to 224x224 — a 7x
enlargement that destroys the high-frequency detail the FFT branch exists
to read. This checkpoint validates that the pipeline RUNS. Do not tune
against its AUC. See docs/PLAN.md §2.

Usage:
    python3 scripts/build_cifake_manifest.py                  # 2k/class smoke set
    python3 scripts/build_cifake_manifest.py --per-class-train 0   # use all 50k/class
"""
import argparse
import random
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(directory: Path) -> list:
    """Sorted image paths, ignoring .DS_Store and other non-image cruft."""
    if not directory.is_dir():
        raise SystemExit(f"missing directory: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def sample(rng: random.Random, paths: list, n: int) -> list:
    """Take n paths deterministically; n<=0 or n>len means take everything."""
    if n <= 0 or n >= len(paths):
        return list(paths)
    return rng.sample(paths, n)


def rel(p: Path) -> str:
    resolved = p.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/CIFake",
                    help="unpacked CIFAKE root (contains train/ and test/)")
    ap.add_argument("--out", default="data/manifest_cifake.csv")
    ap.add_argument("--generator-name", default="sd14_cifake",
                    help="`generator` value for fakes; reals are always 'real'")
    ap.add_argument("--per-class-train", type=int, default=2000,
                    help="images per class drawn from train/ (0 = all 50k)")
    ap.add_argument("--per-class-test", type=int, default=500,
                    help="images per class drawn from test/ (0 = all 10k)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of the train/ draw held out as val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    rows = []

    # --- train/ -> train + val, carved per class so both stay balanced ---
    for folder, label, generator in [("REAL", 0, "real"), ("FAKE", 1, args.generator_name)]:
        pool = sample(rng, list_images(root / "train" / folder), args.per_class_train)
        rng.shuffle(pool)
        n_val = int(len(pool) * args.val_frac)
        for i, p in enumerate(pool):
            rows.append({
                "path": rel(p), "label": label, "generator": generator,
                "split": "val" if i < n_val else "train",
            })

    # --- test/ -> test (the dataset's own held-out split) ---
    for folder, label, generator in [("REAL", 0, "real"), ("FAKE", 1, args.generator_name)]:
        for p in sample(rng, list_images(root / "test" / folder), args.per_class_test):
            rows.append({
                "path": rel(p), "label": label, "generator": generator,
                "split": "test",
            })

    df = pd.DataFrame(rows, columns=["path", "label", "generator", "split"])

    # --- verification: fail loudly here, not inside a DataLoader worker ---
    missing = [p for p in df["path"] if not (REPO / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest paths do not exist, e.g. {missing[:3]}")
    if df["path"].duplicated().any():
        raise SystemExit("duplicate paths in manifest — an image would appear in two splits")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out} — {len(df)} rows\n")
    print(pd.crosstab(df["split"], [df["label"], df["generator"]]))
    print("\nNOTE: CIFAKE is 32x32 upsampled 7x to CLIP's 224 — the frequency")
    print("branch sees interpolation artifacts, not generator fingerprints.")
    print("This validates plumbing only; do not tune against its AUC.")


if __name__ == "__main__":
    main()
