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

## 7. One-day compressed timeline (4 people)

Superseded the original 3-day plan — full sprint in a single day. Hour markers are relative to whenever you actually start (`Hour 0`); anchor that against your real deadline and compress/stretch the buffer at the end accordingly, not the middle. This scope is deliberately smaller than the 3-day version: a WildFake **subset** across 3-4 generators (not the full corpus), one training run (no hyperparameter search, no CLIP-vs-hybrid ablation), and a smaller robustness test slice. Note the cuts explicitly in the Devpost write-up as "future work" — judges read a scoped-and-honest MVP better than an overreaching one that broke at the end.

| Hours | Track A (Data/Aug) | Track B (Model/Training) | Track C (Robustness/Eval) | Track D (Repo/Infra/Demo) |
|---|---|---|---|---|
| 0–1 | All 4: confirm repo cloned, `pip install -r requirements.txt` runs, `bash run.sh` reachable, tracks confirmed | | | |
| 0–2 | Download CIFAKE (full, it's small) + a WildFake **subset**: pick 3-4 generator subfolders, cap ~1.5-2.5k images/class; download a small SID_Set slice (a few hundred images) kept completely separate | Fix the dual-tensor TODO (`train.py`/`datasets.py` need to return both the CLIP-preprocessed tensor and a raw `[0,1]` tensor per sample) | Idle until checkpoint 1 — read `evaluate.py` / `EVAL_CONDITIONS` to be ready | Push repo, add 3 collaborators, draft README/Devpost skeleton from `docs/PLAN.md`, pick demo example shots |
| 2–3 | Build `manifest.csv` (real/subset-fake/generator column correct); hand off | **Checkpoint 1:** smoke-test 1 epoch on CIFAKE — pipeline runs end to end, val AUC prints | Stand up `build_robustness_testset.py` + `evaluate.py` against the CIFAKE smoke-test checkpoint to catch bugs in the eval code before real numbers matter | Set up screen recording tool, outline demo script (real→correct, fake→correct, fake+JPEG/blur→still caught, one honest failure case) |
| 3–6 | Verify SID_Set slice manifest rows are present, correctly labeled, and **excluded** from every train/val split | Real training run on the WildFake subset with augmentation on (target ~5-8 epochs depending on compute); save best checkpoint by val AUC | Idle/on-call for Track B; help debug if training stalls | Continue README/Devpost draft; nothing to demo yet |
| 6 | | **Checkpoint 2:** best checkpoint saved, handed to Track C | | |
| 6–7 | Free — help with error analysis | `calibrate.py` (temperature scaling on val set) | `build_robustness_testset.py` on the (smaller) held-out test slice → `evaluate.py` → real robustness table + Final Score | Keep repo/README in sync as real numbers land |
| 7–8 | Inspect false positives/negatives together — write the error-analysis note (2-3 concrete failure examples + a one-line hypothesis each), lock `docs/robustness_results.json` | | | |
| 8–9 | All 4: repo cleanup (resolve/remove leftover TODOs, confirm `run.sh` works top-to-bottom on a **fresh clone**), tag a release commit, run the §6 compliance checklist | | | |
| 8–10 | | | | Record the 2-4 min demo video per the script above; edit/trim |
| 9–10 | | | | Write the Devpost description: architecture, real results table, trade-offs (§5, reuse it), explicit "what we cut for time / would add next" section |
| 10–11 | **Buffer.** Fix whatever broke in the fresh-clone test, re-render demo audio/captions if needed, don't start anything new here | | | |
| 11 | **Submit:** public repo confirmed, `run.sh` tested clean, Devpost live, demo video uploaded — with margin before the deadline, not at it | | | |

If you're running short by Hour 6 (training not converged / checkpoint missing), the fallback is: ship the CIFAKE-only checkpoint from Checkpoint 1, be upfront about it in the Devpost ("MVP trained on a smaller sanity-check set due to time; full WildFake run is the immediate next step"), and spend the reclaimed hours making the robustness table and error analysis on *that* checkpoint as solid as possible. A smaller, honestly-reported result beats a bigger claim you ran out of time to verify.

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

