# Workstream Ownership

Today's plan (see `docs/PLAN.md` §7) covers only the main technical task — data, model, training, robustness evaluation. Demo recording, repo/infra cleanup, and the Devpost write-up are deliberately **not** scheduled here; they get planned as a separate pass once the four of you confirm the core pipeline and results are done.

## Ngiam — Data & Augmentation
- Own dataset acquisition and the `manifest.csv` build for WildFake / CIFAKE / SID_Set
- Own `src/data/augmentations.py` and `src/data/datasets.py`
- Responsible for the `generator` column being correct, and for keeping SID_Set fully excluded from training — this is the single dependency Aarav's cross-generator eval relies on

## Letao — Semantic Branch & Training Loop
- Own `src/models/clip_backbone.py` and `src/train.py`
- First concrete task: fix the dual-tensor (CLIP-preprocessed + raw [0,1]) wiring flagged as a TODO in `train.py`
- Co-owns the training run and the trained checkpoint with Aaron

## Aaron — Frequency Branch & Fusion
- Own `src/models/frequency_branch.py` and `src/models/detector.py` (fusion head + calibration temperature)
- Early task: sanity-test the fusion forward pass on a dummy batch before real data is ready, so shape/dtype bugs surface before the training run
- Co-owns the training run and the trained checkpoint with Letao

## Aarav — Robustness & Evaluation
- Own `scripts/build_robustness_testset.py`, `src/evaluate.py`, `src/calibrate.py`
- Own the robustness table and the Final Score computation
- Owns the error-analysis note — needs Letao/Aaron's checkpoint and Ngiam's held-out SID_Set data

## Handoff points (don't let these block silently — flag early)
- Ngiam → Letao/Aaron: `manifest.csv` must exist before a real (non-CIFAKE-smoke-test) training run
- Letao/Aaron → Aarav: checkpoint must exist before real robustness eval
- Ngiam → Aarav: SID_Set manifest rows must be present and excluded from training before the unseen-generator number means anything
- All 4 → (later, separate pass): final numbers and one failure-case example are what the demo video and Devpost draw from — keep `docs/robustness_results.json` accurate so that pass isn't blocked either
