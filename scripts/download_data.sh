#!/usr/bin/env bash
# Dataset acquisition — one command, run from the repo root.
#
#   bash scripts/download_data.sh
#
# Pulls the exact images listed in manifest/data_spec.csv and writes
# manifest/manifest.csv. Everyone on the team ends up with byte-identical
# data because the spec pins each image and records its sha256.
#
# WildFake is 1.29 TB and its download unit is a whole archive, so a normal
# clone or `git lfs pull` is not viable. We instead read each archive's index
# remotely and fetch only the members we need over HTTP range requests —
# under 1 GB total, no ModelScope account required. See
# scripts/wildfake_remote.py for the mechanism.
#
# Competition rules require public/licensed datasets and reproducible
# acquisition scripts (docs/PLAN.md §6). WildFake is Apache-2.0; nothing is
# redistributed here — this fetches from the official source.
set -e
cd "$(dirname "$0")/.."

echo "==> Fetching both datasets (~125 MB total):"
echo "      WildFake  4200 imgs - 3 generators x 700 + 3 matched real sources x 700"
echo "                           -> train / val / test"
echo "      SID_Set    500 imgs - 250 real + 250 synthetic, split=heldout"
echo "                           -> unseen-generator eval only, never trained on"
python3 scripts/fetch_dataset.py

echo
echo "==> Done. Sanity-check the result:"
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("manifest/manifest.csv")
print(pd.crosstab([df["split"]], [df["label"], df["domain"]]))
print(f"\n{len(df)} rows -> manifest/manifest.csv")
PY

echo
echo "Next: point configs/baseline_clip.yaml at manifest/manifest.csv, then"
echo "  bash run.sh train"
