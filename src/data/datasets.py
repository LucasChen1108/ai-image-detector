"""
Manifest-driven dataset loader.

We deliberately do NOT hardcode a folder-per-class layout, because the
eval protocol needs a `generator` column to hold out entire generators for
cross-generator testing (train on generators {A, B, C}, test on generator D
that never appeared in training — the real generalization test per the
brief).

Expected manifest CSV columns:
    path,label,generator,split
      path      - absolute or repo-relative image path
      label     - 0 = real, 1 = AI-generated
      generator - "real" for real images, else the generator name
                  (e.g. "sd_xl", "midjourney", "wildfake_ganA", ...)
      split     - "train" | "val" | "test"

Build these manifests with a small preprocessing script per dataset
(WildFake / CIFAKE / SID_Set) rather than editing this loader — keeps the
loader dataset-agnostic.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        preprocess: Callable,
        train_augment: Optional[Callable] = None,
        exclude_generators: Optional[list] = None,
        include_only_generators: Optional[list] = None,
    ):
        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if exclude_generators:
            df = df[~df["generator"].isin(exclude_generators)].reset_index(drop=True)
        if include_only_generators:
            df = df[df["generator"].isin(include_only_generators)].reset_index(drop=True)
        self.df = df
        self.preprocess = preprocess  # CLIP's image preprocess (resize/normalize)
        self.train_augment = train_augment  # RedistributionAugment, train split only
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.split == "train" and self.train_augment is not None:
            img = self.train_augment(img)
        tensor = self.preprocess(img)
        return {
            "image": tensor,
            "label": float(row["label"]),
            "generator": str(row["generator"]),
        }


def make_manifest_from_folders(real_dir: str, fake_dir: str, generator_name: str,
                                out_csv: str, split_ratios=(0.8, 0.1, 0.1), seed: int = 42):
    """Helper: turn a real/ and fake/ folder pair into a manifest CSV with a
    random train/val/test split. Use once per dataset, then hand-edit the
    `generator` column for fake-only sources if you have finer-grained labels
    (e.g. per-GAN subfolders) — that granularity is what makes cross-generator
    holdout evaluation possible."""
    import random
    random.seed(seed)
    rows = []
    for p in Path(real_dir).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rows.append((str(p), 0, "real"))
    for p in Path(fake_dir).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rows.append((str(p), 1, generator_name))
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df = pd.DataFrame(rows, columns=["path", "label", "generator"])
    df["split"] = splits
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {n} rows ({n_train} train / {n_val} val / {n - n_train - n_val} test)")
