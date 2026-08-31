#!/usr/bin/env bash
# Download the dataset with one command from the repo root.
#
#   bash scripts/download_data.sh
#
# Fetch the exact images listed in manifest/data_spec.csv and write them to
# manifest/manifest.csv. The spec pins every image and records its SHA-256,
# so everyone gets the same bytes.
#
# WildFake is 1.29 TB, and each download normally pulls a whole archive, so
# cloning it or running `git lfs pull` isn't practical. We read each archive's
# index remotely and fetch only the needed members with HTTP range requests.
# The total download stays under 1 GB, and no ModelScope account is needed.
# See scripts/wildfake_remote.py for the details.
#
# WildFake uses Apache-2.0, and this script fetches it from the official
# source without redistributing it here.
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
