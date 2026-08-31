#!/usr/bin/env bash
# Single entrypoint required by the competition rules ("run script").
#
# Usage:
#   bash run.sh train           # train the hybrid detector
#   bash run.sh calibrate       # fit temperature scaling on the val set
#   bash run.sh build_testset   # materialize the robustness test set
#   bash run.sh evaluate        # run full robustness eval + Final Score
#   bash run.sh predict --input_dir DIR [--out predictions.json]
#                                # score every image in DIR, writing
#                                # [{"image_path": ..., "pred": ...}, ...]
#   bash run.sh find_errors     # pull representative false positive /
#                                # false negative examples for the error
#                                # analysis note (docs/examples/, docs/error_examples.json)
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
