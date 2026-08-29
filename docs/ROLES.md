# Workstream Ownership

Generic A/B/C/D — assign your 4 names to these however fits your team's strengths (someone comfortable with PyTorch training loops → B; someone comfortable with stats/eval → C, etc.).

## Track A — Data & Augmentation
- Own dataset acquisition, licensing, and the `manifest.csv` build for WildFake / CIFAKE / SID_Set
- Own `src/data/augmentations.py` and `src/data/datasets.py`
- Responsible for the `generator` column being correct — this is the single dependency Track C's cross-generator eval relies on

## Track B — Model & Training
- Own `src/models/*.py` and `src/train.py`
- First concrete task: fix the dual-tensor (CLIP-preprocessed + raw [0,1]) wiring flagged as a TODO in `train.py`
- Owns hyperparameters in `configs/baseline_clip.yaml` and the trained checkpoint

## Track C — Robustness & Evaluation
- Own `scripts/build_robustness_testset.py`, `src/evaluate.py`, `src/calibrate.py`
- Own the robustness table and the Final Score computation
- Owns the error-analysis note (Day 3) — needs Track B's checkpoint and Track A's held-out SID_Set data

## Track D — Infra, Repo & Demo
- Own the GitHub repo, README accuracy, `run.sh` correctness end-to-end
- Own the Devpost write-up and demo video (script, record, edit)
- Runs the Day-3 rules-compliance checklist in `docs/PLAN.md` §6 before submission

## Handoff points (don't let these block silently — flag early)
- A → B: manifest.csv must exist before B can do a real (non-CIFAKE-smoke-test) training run
- B → C: checkpoint must exist before C can run real robustness eval
- A → C: SID_Set manifest rows must be present and excluded from training before C's unseen-generator number means anything
- B, C → D: final numbers and one failure-case example needed before the demo video script is final
