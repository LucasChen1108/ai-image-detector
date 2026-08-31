# Who owns what

- **Ngiam** — data & augmentation. Got WildFake/CIFAKE/SID_Set into `manifest.csv`, owns `src/data/augmentations.py` and `datasets.py`. Also the one who has to make sure SID_Set actually stays out of training, since Aarav's whole "unseen generator" number depends on that being true.
- **Letao** — semantic branch + training loop. Owns `src/models/clip_backbone.py` and `src/train.py` (ended up owning most of the cross-cutting pipeline bugs too, since a few bugs in `train.py` turned out to be the same bugs in `evaluate.py` and `calibrate.py`).
- **Aaron** — frequency branch + fusion. Owns `src/models/frequency_branch.py` and `src/models/detector.py` (fusion head + the calibration temperature parameter).
- **Aarav** — robustness + evaluation. Owns `scripts/build_robustness_testset.py`, `src/evaluate.py`, `src/calibrate.py`, and the error-analysis writeup.

Dependencies that actually mattered, in order: `manifest.csv` had to exist before anyone could do a real training run, a checkpoint had to exist before Aarav could evaluate anything for real, and Ngiam's SID_Set exclusion had to actually hold or the whole unseen-generator test would've meant nothing. We tried to flag these early rather than find out at the end that one was silently broken.
