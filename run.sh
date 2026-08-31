#!/usr/bin/env bash
# This is the single entrypoint for the project.
#
# Usage:
#   bash run.sh train           # Train the hybrid detector.
#   bash run.sh calibrate       # Fit temperature scaling on the validation set.
#   bash run.sh build_testset   # Build the robustness test set.
#   bash run.sh evaluate        # Run the full robustness evaluation and overall score.
#   bash run.sh predict --input_dir DIR [--out predictions.json]
#                               # Score every image in DIR and write
#                               # [{"image_path": ..., "pred": ...}, ...].
#   bash run.sh find_errors     # Find representative false positives and
#                               # false negatives for the error analysis note
#                               # in docs/examples/ and docs/error_examples.json.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

CMD="${1:-evaluate}"
case "$CMD" in
  train)
    python3 src/train.py --config configs/baseline_clip.yaml
    ;;
  calibrate)
    python3 src/calibrate.py --config configs/baseline_clip.yaml --checkpoint checkpoints/best.pt
    ;;
  build_testset)
    python3 scripts/build_robustness_testset.py \
      --source_manifest manifest/manifest.csv \
      --out_dir data/robustness_testset \
      --out_manifest data/robustness_manifest.csv
    ;;
  evaluate)
    python3 src/evaluate.py --config configs/baseline_clip.yaml \
      --checkpoint checkpoints/best.pt --manifest data/robustness_manifest.csv
    ;;
  predict)
    shift
    python3 src/predict.py --config configs/baseline_clip.yaml \
      --checkpoint checkpoints/best.pt "$@"
    ;;
  find_errors)
    shift
    python3 scripts/find_error_examples.py --config configs/baseline_clip.yaml \
      --checkpoint checkpoints/best.pt "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: bash run.sh {train|calibrate|build_testset|evaluate|predict|find_errors}"
    exit 1
    ;;
esac
