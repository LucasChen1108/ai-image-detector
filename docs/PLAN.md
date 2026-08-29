# Team Plan — AI-Generated Image Detector (TechJam)

**Deadline:** ~3 days from kickoff (Sat Aug 29, 2026 → submission target Mon Aug 31 / Tue Sep 1, Asia/Shanghai). Confirm the exact hour against the official deadline and adjust the timeline below — this plan assumes ~65 working hours across 4 people.

**Interpretation note:** "use data from CLIP" is read here as *use CLIP's pretrained visual encoder as the semantic branch*, per the slide deck's own key insight ("best detectors combine high-level CLIP semantics + low-level frequency patches"). This is a modeling choice, not a dataset choice — datasets are separate (see below).

---

## 1. What we're building

A hybrid two-branch binary classifier:

- **Semantic branch:** frozen (or lightly fine-tuned) CLIP visual encoder → learns high-level "does this look real" cues — impossible lighting, warped hands, garbled text, texture inconsistencies.
- **Frequency branch:** small CNN over the log-magnitude FFT spectrum of the raw pixels → learns low-level generator fingerprints (GAN/diffusion up-sampling periodicity, missing sensor noise) that survive when semantics look flawless.
- **Fusion head:** concat both embeddings → MLP → single logit → calibrated probability (temperature scaling).

This directly targets the brief's "Key Insight: Go hybrid" and its stated reason: each branch survives different transformations, so fusing them is what buys robustness, not just accuracy.

Why not just fine-tune CLIP alone? Slide 1 warns explicitly: "don't just fine-tune a classifier — think about what your model is actually learning." A CLIP-only classifier can shortcut on semantic giveaways that later generators fix, and it has no mechanism at all for the frequency-domain fingerprints that are often the most durable signal. The frequency branch is cheap (a 4-layer CNN) and buys a second, differently-fragile signal source.

## 2. Dataset plan

| Role | Dataset | Why |
|---|---|---|
| Primary training | **WildFake** | Large-scale, multi-generator (spans several GAN + diffusion families) — training on generator diversity is the single biggest lever for the "generalization" axis the brief scores on. |
| Day-1 sanity check only | **CIFAKE** | Small, low-res, single-generator (Stable Diffusion vs CIFAR-10 reals). Fast to download and iterate on — use it Day 1 to prove the pipeline runs end-to-end in minutes before committing hours to the full WildFake run. Not used for final training or eval. |
| Held-out cross-generator eval | **SID_Set** | Kept **completely out of training**. This is what makes the "unseen generator" row in the robustness table honest — the brief explicitly calls this out as "the real generalization test." If it leaks into training, that number stops meaning anything. |

All three are on the competition's approved list. No proprietary/production data, no test-label training — enforced by keeping SID_Set out of every training/val manifest, not just the test split.

Data prep produces one `manifest.csv` (see `src/data/datasets.py`) with `path,label,generator,split` — the `generator` column is what makes both cross-generator holdout and per-generator error analysis possible later, so get it right early rather than backfilling it Day 2.

## 3. Augmentation = training-time redistribution simulation

Mental model from the brief: *"if a transformation can happen on a real feed, it must happen in your training pipeline."* Implemented in `src/data/augmentations.py`:

- JPEG recompression at random quality (real re-encode, not a blur approximation)
- Gaussian blur
- **Crop-then-resize instead of down-sampling** (SAFE, KDD 2025 insight — cropping preserves the high-frequency detail that down-sampling destroys)
- Color jitter + random rotation (SAFE insight — kills color/semantic shortcuts)
- Sensor-noise injection
- Simulated re-screenshot (down-resize → blur → recompress → upscale)

One caution baked in per the DDA (NeurIPS 2025) insight: apply the *same* compression family to both real and fake training images, so "has JPEG artifacts" never becomes a spurious proxy for "is fake." Align pixel-domain and frequency-domain statistics rather than only pixel-domain.

## 4. Evaluation protocol

Primary metric: **ROC AUC** (threshold-free, robust to class imbalance — per the brief).

Build a transformed test set (`scripts/build_robustness_testset.py`) with these conditions, matching the slide deck's own table:

| Condition |
|---|
| Clean |
| JPEG q90 / q70 / q50 / q30 |
| Blur σ=2 |
| Crop 80% |
| Unseen generator (SID_Set, never trained on) |

```
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

where `AUC_robust` is the mean AUC across the JPEG/blur/crop conditions. Unseen-generator AUC is reported **separately** in the table (it measures generalization, not post-processing robustness — averaging it into `AUC_robust` would hide whichever axis is actually weaker, which defeats the point of measuring both).

Deliverable per the brief: a compact robustness table + an error-analysis note — a short write-up of *where* the model fails (which generator, which corruption, false-positive vs false-negative pattern), not just the numbers. `src/evaluate.py` saves per-condition results to `docs/robustness_results.json`; Track C turns that into the written note on Day 3.

## 5. Trade-offs to name explicitly (the brief scores this)

- **Robustness vs. clean accuracy:** heavy augmentation will cost a little clean-set AUC. Expected and worth it — a model that's 99% on clean and collapses under JPEG q30 is worse for the brief's stated goal than one that's 95%/90% across the board. Report both, don't hide the trade.
- **Generalization vs. specialization:** the frozen-CLIP-backbone choice trades away a few points of in-distribution accuracy for held-out-generator robustness. This is deliberate — see §1.
- **Complexity vs. feasibility:** we are explicitly *not* building a multi-branch ensemble beyond two branches, and not chasing an FFT+DCT+PRNU triple-branch system. Two branches, frozen backbone, small fusion MLP — something that trains in hours on hackathon compute and has a working demo, per "ship what runs."

## 6. Rules compliance (check before submitting)

- CLIP backbone is public/open-source ✓ (open_clip, public weights)
- Custom fusion/frequency-branch code released under MIT (this repo's `LICENSE`) ✓
- Only public/licensed datasets (WildFake, CIFAKE, SID_Set), no proprietary data, no test-label training ✓
- Augmentation and robustness-test-set generation scripts included and runnable (`scripts/`) ✓
- Model parameter count: CLIP ViT-B/32 (~151M) or ViT-L/14 (~428M) + small heads — both far under the 2B cap ✓
- Not a direct replication of a single existing paper/model — the two-branch CLIP+FFT fusion with learned temperature calibration is our own combination, citing SAFE/DDA as *inspiration* for augmentation strategy, not as an architecture to copy ✓
- Submission = public GitHub repo + run script (`run.sh`) + Devpost description + 2–4 min YouTube demo — tracked in §8 below

## 7. One-day plan — main task only (Ngiam, Letao, Aaron, Aarav)

Scope for today is the core technical task: data, model, training, robustness evaluation, and the error-analysis note. Demo recording, repo/infra cleanup, and the Devpost write-up are **not** in this schedule — they get planned as a separate pass once the four of you confirm the pipeline and results below are actually done. (`docs/ROLES.md` has the same split with more detail.)

Scope is also smaller than a full run: a WildFake **subset** across 3-4 generators (not the full corpus), one training run (no hyperparameter search, no CLIP-vs-hybrid ablation), and a smaller robustness test slice. Note any cuts explicitly when you get to write-up time later — a scoped, honestly-reported result beats an overreaching one that broke at the end.

| Hours | Ngiam (Data/Aug) | Letao (Semantic branch + training) | Aaron (Frequency branch + fusion) | Aarav (Robustness/Eval) |
|---|---|---|---|---|
| 0–1 | All 4: confirm repo cloned, `pip install -r requirements.txt` runs, `bash run.sh` reachable, walk through the architecture together so every handoff below is understood before it happens | | | |
| 0–2 | Download CIFAKE (full, it's small) + a WildFake **subset**: 3-4 generator subfolders, ~1.5-2.5k images/class; download a small SID_Set slice (a few hundred images) kept completely separate; start building `manifest.csv` | Fix the dual-tensor TODO (`train.py`/`datasets.py` need to return both the CLIP-preprocessed tensor and a raw `[0,1]` tensor per sample) | Finish/review `frequency_branch.py` and the fusion logic in `detector.py`; sanity-test the forward pass on a dummy random-tensor batch so shape/dtype bugs surface before real data is ready | Review `evaluate.py` / `build_robustness_testset.py`; run them against a tiny dummy manifest + a randomly-initialized model to shake out eval-code bugs early, before real numbers matter |
| 2–3 | **Checkpoint 1:** hand off `manifest.csv` | Smoke-test 1 training epoch on CIFAKE together with Aaron — pipeline runs end to end, val AUC prints | (joint with Letao, see left) | Confirm eval scripts run cleanly against the CIFAKE smoke-test checkpoint |
| 3–6 | Verify the SID_Set slice manifest rows are present, correctly labeled, and **excluded** from every train/val split; help debug data issues if any surface | Real training run on the WildFake subset with augmentation on (target ~5-8 epochs); co-owns this with Aaron | Co-owns the training run with Letao; watches the frequency-branch/fusion behavior specifically for NaNs or dead gradients | Finish debugging the robustness-test-set builder and evaluator against the CIFAKE checkpoint so it's fully ready the moment a real checkpoint lands |
| 6 | | **Checkpoint 2:** best checkpoint saved, handed to Aarav | | |
| 6–7 | Free — help inspect failure cases | Free — help inspect failure cases | Free — help inspect failure cases | `calibrate.py` (temperature scaling) → `build_robustness_testset.py` on the held-out test slice → `evaluate.py` → real robustness table + Final Score |
| 7–8 | All 4: look at false positives/negatives together, write the error-analysis note (2-3 concrete failure examples + a one-line hypothesis each), lock `docs/robustness_results.json` | | | |

**End of Hour 8 = main task done:** working hybrid model, real robustness table, Final Score, error-analysis note. Stop here — plan demo recording, repo/infra polish, and the Devpost write-up as a separate follow-on pass once this is actually confirmed working, not squeezed into the same day's schedule preemptively.

If you're running short by Hour 6 (training not converged / checkpoint missing), fall back to the CIFAKE-only checkpoint from Checkpoint 1, note that honestly whenever you do write things up later, and spend the reclaimed hours making the robustness table and error analysis on *that* checkpoint as solid as possible.

## 8. GitHub repo setup (do this now, Day 1)

This scaffold has been generated locally at `ai-image-detector/` inside your TechJam folder and initialized as a local git repo. To get it on GitHub:

1. Go to github.com → New repository. Name it (e.g. `techjam-ai-image-detector`), leave it **empty** (no README/gitignore/license — we already have those), set visibility to **public** (competition requires a public repo).
2. Copy the repo URL it gives you (`https://github.com/<you>/<repo>.git`).
3. Run:
   ```bash
   cd ai-image-detector
   git remote add origin <PASTE_URL_HERE>
   git branch -M main
   git push -u origin main
   ```
4. Add your 3 teammates as collaborators (repo Settings → Collaborators), or have them fork+PR if you prefer that workflow.
5. Everyone clones and works from feature branches (`git checkout -b track-a-data`, etc.), PRs into `main` — with 3 days, keep review fast and don't block on it; a quick Slack/Discord "pushed X, reviewing now" beats a formal PR process here.

