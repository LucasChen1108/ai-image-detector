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

## 7. Three-day timeline (4 people)

### Day 1 — Sat Aug 29: Pipeline stands up end-to-end
All 4 work the core pipeline together; sub-tasks below are what each of you starts on, but nobody is siloed — check in with whoever's blocking you rather than working around it.

- **Track A (Data/Aug):** Get WildFake + CIFAKE + SID_Set downloaded and licensed correctly; build `manifest.csv` via `make_manifest_from_folders`; verify `augmentations.py` transforms visually on a handful of samples.
- **Track B (Model/Training):** Wire up `ClipSemanticBranch` + `FrequencyBranch` + `HybridDetector`; fix the `clip_and_raw` dual-tensor TODO in `src/train.py` (dataset currently returns one tensor — needs both the CLIP-preprocessed and raw-[0,1] tensors per sample); get one training epoch running on CIFAKE as a smoke test.
- **Track C (Eval):** Stand up `evaluate.py` against dummy/CIFAKE data to confirm the AUC table and Final Score formula compute correctly before real numbers matter.
- **Track D (Repo/Infra):** This scaffold (done) → push to GitHub (see §8), set up a shared results doc, confirm everyone can run `bash run.sh train` locally.

**End of Day 1 goal:** `bash run.sh train` completes an epoch on real (even if small) data and prints a val AUC.

### Day 2 — Sun Aug 30: Real training + robustness
- **Track A:** Finalize the full WildFake training manifest; hold SID_Set out completely; sanity-check the `generator` column is populated correctly (this is what makes cross-generator eval possible).
- **Track B:** Full training run on WildFake with augmentation on; save best checkpoint; run `calibrate.py`.
- **Track C:** Run `build_robustness_testset.py` on the held-out test split; run full `evaluate.py`; get the real robustness table and Final Score; start drafting the error-analysis note (which conditions/generators fail, and a hypothesis why).
- **Track D:** Ablation if time allows — CLIP-only vs. hybrid, augmentation on/off — feeds directly into the "how senior engineers think: trade-offs" section of the write-up. Keep the repo's README/PLAN in sync with whatever actually shipped.

**End of Day 2 goal:** Full robustness table with real numbers, checkpoint saved, error-analysis draft started.

### Day 3 — Mon Aug 31 (buffer into Tue if deadline allows): Polish, demo, submit
- **All 4:** Finalize error-analysis note and trade-offs write-up (§5 above — reuse it, don't rewrite from scratch).
- **Track D leads, all review:** Clean repo (remove dead code/TODOs that didn't get resolved, make sure `run.sh` actually runs top-to-bottom on a clean checkout), finalize README, confirm `requirements.txt` is accurate, tag a release commit.
- **Track C:** Freeze final numbers into `docs/robustness_results.json`, make sure the table in this doc / README matches what's actually reproducible.
- **All 4:** Write the Devpost description (approach, results table, trade-offs, what we'd do with more time). Script and record the 2–4 min demo video — show: (1) a real image classified correctly, (2) a fake image classified correctly, (3) the same fake surviving JPEG/blur and still being caught, (4) one honest failure case from the error analysis (judges notice teams who show a failure mode they understand vs. teams who only show wins).
- **Submit:** public repo link + run script confirmed working + Devpost + YouTube demo, before the deadline with buffer for upload/processing time.

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

