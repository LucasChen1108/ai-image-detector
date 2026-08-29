"""
Append the SID_Set held-out slice to manifest/data_spec.csv.

Run ONCE by whoever owns data. Teammates just run scripts/download_data.sh.

WHY NOT `load_dataset("saberzl/SID_Set")`: that downloads the whole corpus
first — 124 GB train + 17 GB validation — and only then lets you filter. We
need a few hundred images.

Instead we read the parquet directly over HTTP range requests, the same
mechanism used for WildFake. Parquet stores data in row groups with a footer
index, so we can (a) read only the `img_id`/`label` columns to find out what
is where, which costs almost nothing, then (b) fetch only the row groups
holding the images we picked, skipping the `mask` column entirely.

Cost floor: each row is a 1024px image (~540 KB), so ~500 images is ~330 MB.
That cannot be reduced without lowering resolution or taking fewer images.

LABELS (verified against the live data, not the card):
    0 -> real                (img_id is a content hash)
    1 -> full_synthetic_*    -> our label 1
    2 -> tampered_*          -> EXCLUDED by default

Tampered images are a different task: most of their pixels are authentic
camera output, so detecting them is localization, not generated-image
detection. Folding them into the unseen-generator condition would conflate
"can't generalize to a new generator" with "can't do tamper localization,
which we never trained for" — and that number exists to isolate the first.
Pass --include-tampered to keep them under their own generator tag.

Everything lands with split="heldout", so it is invisible to train.py (asks
for train/val) and to build_robustness_testset.py (asks for test). That is
what makes the unseen-generator AUC honest — docs/PLAN.md §2.

Usage:
    python3 scripts/build_sid_spec.py --per-class 250
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wildfake_remote import HttpFile  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHARD_URL = ("https://huggingface.co/datasets/saberzl/SID_Set/resolve/main/"
             "data/validation-{:05d}-of-00034.parquet")

LABEL_MAP = {                       # dataset label -> (our label, generator)
    0: (0, "real"),
    1: (1, "sid_synthetic"),
    2: (1, "sid_tampered"),
}


def open_shard(idx: int):
    url = SHARD_URL.format(idx)
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    size = int(r.headers["content-length"])
    return pq.ParquetFile(pa.PythonFile(HttpFile(r.url, size))), size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--include-tampered", action="store_true")
    ap.add_argument("--spec", default="manifest/data_spec.csv")
    ap.add_argument("--max-shards", type=int, default=4)
    args = ap.parse_args()

    keep = {0, 1} | ({2} if args.include_tampered else set())
    quota = {k: args.per_class for k in keep}
    rows, seen = [], collections.Counter()

    for shard in range(args.max_shards):
        if sum(quota.values()) == 0:
            break
        pf, size = open_shard(shard)
        print(f"shard {shard}: {size/1e6:.0f} MB, {pf.metadata.num_rows} rows, "
              f"{pf.metadata.num_row_groups} row groups")
        for rg in range(pf.metadata.num_row_groups):
            if sum(quota.values()) == 0:
                break
            # cheap: only the two small columns, no image bytes
            t = pf.read_row_group(rg, columns=["img_id", "label"])
            ids = t.column("img_id").to_pylist()
            labs = t.column("label").to_pylist()
            for i, (iid, lab) in enumerate(zip(ids, labs)):
                if lab in quota and quota[lab] > 0:
                    our_label, generator = LABEL_MAP[lab]
                    rows.append({
                        "source": "sid_set",
                        "archive": SHARD_URL.format(shard),
                        "member": iid,
                        "dest_path": f"data/sid_set/{generator}/{iid}.jpg",
                        "label": our_label, "generator": generator,
                        "domain": "sid_set", "split": "heldout",
                        "rowgroup": rg, "sha256": "",
                    })
                    quota[lab] -= 1
                    seen[generator] += 1
        print(f"  after shard {shard}: {dict(seen)} (remaining {quota})")

    if sum(quota.values()) > 0:
        raise SystemExit(f"quota unfilled after {args.max_shards} shards: {quota}")

    spec = REPO / args.spec
    existing = list(csv.DictReader(open(spec))) if spec.exists() else []
    existing = [r for r in existing if r.get("source") != "sid_set"]
    cols = ["source", "archive", "member", "dest_path", "label",
            "generator", "domain", "split", "rowgroup", "sha256"]
    for r in existing:
        r.setdefault("rowgroup", "")
    with open(spec, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(existing + rows)
    print(f"\nappended {len(rows)} SID_Set rows to {args.spec} "
          f"({len(existing)} existing rows preserved)")


if __name__ == "__main__":
    main()
