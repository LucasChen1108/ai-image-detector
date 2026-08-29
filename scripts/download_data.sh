#!/usr/bin/env bash
# Dataset acquisition — FILL IN before Day 1 data work.
#
# Competition rules require public/licensed datasets only (no proprietary or
# production data, no test-label training). The brief names WildFake, CIFAKE,
# and SID_Set as example approved sources. Recommended split for this repo:
#
#   TRAIN (primary):  WildFake   — large multi-generator (GAN + diffusion)
#                      corpus, built specifically for cross-generator
#                      robustness research. Gives the model varied fingerprints
#                      to learn from instead of overfitting one generator family.
#   FAST SANITY CHECK: CIFAKE    — small, low-res, single-generator (Stable
#                      Diffusion vs CIFAR-10 real). Use ONLY on Day 1 to verify
#                      the full pipeline runs end-to-end in minutes, before
#                      committing hours to the full WildFake run.
#   HELD-OUT EVAL:     SID_Set   — kept OUT of training entirely, used only
#                      as the "unseen_generator" condition in evaluate.py.
#                      This is what makes the cross-generator number honest —
#                      if you train on it too, that AUC stops meaning anything.
#
# 1. Download each dataset per its official license terms (fill in the actual
#    URLs/instructions your team finds — do not commit raw data to git,
#    see .gitignore).
# 2. Point each dataset at data/<name>/real/ and data/<name>/fake/ (or your
#    own layout).
# 3. Build a manifest per dataset with src/data/datasets.py::make_manifest_from_folders,
#    then concatenate into data/manifest.csv with a `generator` column set
#    correctly per source (real images always generator="real").

set -e
echo "TODO: fill in official dataset download commands here."
echo "See comments at the top of this file for the train/sanity/eval split plan."
