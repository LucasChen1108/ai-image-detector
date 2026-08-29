# AI-Generated Image Detector — TechJam

Hybrid **CLIP semantics + frequency-domain** detector for distinguishing
AI-generated images from authentic ones, built for robustness under
real-world redistribution (JPEG recompression, blur, cropping, resizing)
and generalization to generators unseen at training time.

See [`docs/PLAN.md`](docs/PLAN.md) for the full team plan, architecture,
evaluation protocol, and 3-day timeline. See [`docs/ROLES.md`](docs/ROLES.md)
for who owns what.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch/prepare data (fill in dataset paths first — see scripts/download_data.sh)
bash scripts/download_data.sh

# 2. Train
bash run.sh train

# 3. Build the robustness test set (JPEG/blur/crop/unseen-generator variants)
bash run.sh build_testset

# 4. Evaluate (produces the compact robustness table + Final Score)
bash run.sh evaluate
```

## Structure

```
src/data/models.py        # CLIP backbone + frequency branch + fusion head
src/data/augmentations.py # training-time "redistribution simulation" transforms
src/data/datasets.py      # manifest-driven real/fake dataset loader
src/train.py               # training loop
src/evaluate.py            # per-condition AUC/Acc, robustness table, Final Score
src/calibrate.py           # temperature-scaling calibration
scripts/build_robustness_testset.py  # generates JPEG q90/70/50/30, blur, crop variants
scripts/download_data.sh   # dataset acquisition (fill in approved sources)
configs/baseline_clip.yaml # hyperparameters
docs/PLAN.md                # full technical + team plan
docs/ROLES.md               # workstream ownership
```

## Compliance checklist (competition rules)

- [ ] Backbone is open-source (CLIP, public weights) — done by design
- [ ] Custom architecture code released under MIT (this repo's LICENSE)
- [ ] Only public/licensed datasets, no test-label training
- [ ] Augmentation/generation scripts included for reproducibility
- [ ] Model stays under 2B params (CLIP ViT-B/32 ≈ 151M / ViT-L/14 ≈ 428M — both comply)
- [ ] Not a direct replication of an existing paper's exact method (fusion design is ours)
- [ ] Public GitHub repo + run script + Devpost description + 2–4 min demo video
