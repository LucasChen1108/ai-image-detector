"""
Choose the WildFake subset and write the download recipe.

Run ONCE (by whoever owns data). Teammates never run this — they run
scripts/fetch_dataset.py against the committed manifest/data_spec.csv and get
byte-identical images.

Selection principles (docs/PLAN.md §2):

  * Three different generation MECHANISMS, not three checkpoints of one.
    ADM (pixel-space guided diffusion), VQVAE (discrete autoencoder) and
    StyleGAN3 (GAN) leave genuinely different fingerprints; three Stable
    Diffusion variants would cost 3x the download for ~1x the signal.

  * Every generator is paired with a CONTENT-MATCHED real source, and each
    domain contributes equal reals and fakes:

        adm       (ImageNet classes) <-> Real/imagenet
        vqvae     (tt-coco)          <-> Real/coco
        stylegan3 (FFHQ-U faces)     <-> Real/ffhq

    Without this the model learns "faces = fake, street scene = real" — a
    content shortcut that survives every augmentation in the pipeline and
    would post a beautiful, meaningless test AUC. Balancing per domain makes
    domain carry zero information about the label.

  * DDIM was rejected despite being the cheapest archive: it is 76% bedrooms
    and the corpus ships no bedroom reals, so it cannot be paired cleanly.

Selection is deterministic (sorted member lists + fixed seed), so this file
regenerates identically on any machine.

Usage:
    python3 scripts/build_data_spec.py --per-source 700
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wildfake_remote import HttpFile, resolve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# (key, archive, member filter, label, generator, domain)
SOURCES = [
    ("adm",       "Images/Diffusion_based/ADM.zip", "ADM/imgs/",      1, "adm",       "imagenet"),
    ("vqvae",     "Images/Other_based.zip",         "/VQVAE/tt-coco/", 1, "vqvae",     "coco"),
    ("stylegan3", "Images/GAN_based.zip",           "/styleGAN/",     1, "stylegan3", "ffhq"),
    ("r_imagenet", "Images/Real/imagenet.zip",      "",               0, "real",      "imagenet"),
    ("r_coco",    "Images/Real/coco.zip",           "",               0, "real",      "coco"),
    ("r_ffhq",    "Images/Real/ffhq.zip",           "",               0, "real",      "ffhq"),
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def namelist_cached(archive: str, cache_dir: Path) -> list:
    """Read a remote archive's central directory, caching it locally.

    The directory read is the expensive part (GAN_based has ~493k members),
    and it never changes, so cache it rather than re-reading on every tweak.
    """
    cache = cache_dir / (archive.replace("/", "_") + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"  reading central directory of {archive} ...", flush=True)
    cdn, size = resolve(archive)
    with zipfile.ZipFile(HttpFile(cdn, size)) as zf:
        names = [n for n in zf.namelist()
                 if not n.endswith("/") and n.lower().endswith(IMAGE_SUFFIXES)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=700,
                    help="images per source; 3 fake + 3 real sources")
    ap.add_argument("--out", default="manifest/data_spec.csv")
    ap.add_argument("--cache-dir", default=".cache/wildfake_namelists")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    for key, archive, filt, label, generator, domain in SOURCES:
        names = namelist_cached(archive, REPO / args.cache_dir)
        sel = sorted(n for n in names if filt in n)
        if len(sel) < args.per_source:
            raise SystemExit(f"{key}: only {len(sel)} members match '{filt}'")
        picked = rng.sample(sel, args.per_source)
        picked.sort()
        rng.shuffle(picked)

        n_val = int(len(picked) * args.val_frac)
        n_test = int(len(picked) * args.test_frac)
        for i, member in enumerate(picked):
            split = "val" if i < n_val else ("test" if i < n_val + n_test else "train")
            stem = hashlib.sha1(f"{archive}::{member}".encode()).hexdigest()[:16]
            rows.append({
                "source": "wildfake", "archive": archive, "member": member,
                "dest_path": f"data/wildfake/{generator}/{stem}.jpg",
                "label": label, "generator": generator, "domain": domain,
                "split": split, "sha256": "",
            })
        print(f"  {key:11} {len(picked):>5} picked from {len(sel):>7} available")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source", "archive", "member", "dest_path", "label",
            "generator", "domain", "split", "sha256"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} — {len(rows)} rows "
          f"(sha256 column filled by fetch_dataset.py --write-hashes)")


if __name__ == "__main__":
    main()
