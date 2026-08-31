# Team plan

Wrote this at the start before we'd built anything, keeping it here since it's roughly what we actually followed (not exactly — see the note at the bottom).

We're building a hybrid classifier: a CLIP branch for the high-level "does this look real" stuff, and a small CNN over the FFT spectrum for the low-level generator-fingerprint stuff, fused with a small MLP and a learned temperature so we get an actual calibrated probability out, not just a ranking score. The two branches see different kinds of image evidence, so they can keep working when an image gets damaged in different ways.

Why not just fine-tune CLIP and call it done? A CLIP-only model can pick up semantic giveaways that a new generator might remove, and it has no way to catch frequency-domain artifacts, which tend to last longer. The frequency branch is cheap to add (4 conv layers), so there wasn't really a reason not to.

## Data

- **WildFake** is our actual training set. It includes several generators, which matters because generalizing across generators is the hardest part of this project.
- **CIFAKE** we only used on day one, to prove the pipeline actually runs end to end before burning hours on a real WildFake run. Small, low-res, single generator — good for catching bugs fast, not for a real result.
- **SID_Set** is held out completely, never touched during training. This is what makes our "unseen generator" number in the results table actually mean something instead of being secretly not-unseen.

`manifest.csv` (built by `src/data/datasets.py`) has `path,label,generator,split` columns. Getting the `generator` column right early mattered a lot since it's what makes both the held-out eval and any later per-generator error analysis possible.

## Augmentation

If a transformation can happen to an image on a real feed — recompression, blur, someone screenshots a screenshot, cropping for a thumbnail — it should happen during training too, or the model picks up on signals that don't survive contact with the real world. So `src/data/augmentations.py` does JPEG recompression at random quality, blur, crop-then-resize instead of plain downsampling (cropping keeps the high-frequency detail that downsampling destroys), color jitter + rotation, sensor noise, and a simulated re-screenshot.

One thing we made a point of doing: apply the same compression family to both real and fake training images, so "has JPEG artifacts" never quietly becomes a shortcut for "is fake" on its own.

## Evaluation

Primary metric is ROC AUC — threshold-free, doesn't care about class imbalance. We build a transformed test set with JPEG at a few qualities, blur, crop, and several other conditions, then compute:

```
Overall score = 0.5 * AUC_clean + 0.5 * AUC_robust
```

Unseen-generator AUC gets reported separately, not folded into the robust average — it's measuring a different thing (generalization, not robustness to post-processing), and averaging them together would just hide whichever one is actually weaker.

## Trade-offs we tried to be upfront about

- Heavy augmentation costs a bit of clean-set accuracy. Worth it — a model that's great on clean images and falls apart under JPEG compression isn't really useful in practice.
- Freezing CLIP trades a couple points of in-distribution accuracy for hopefully-better generalization to generators we didn't train on. Generalization stayed rough anyway in the end (see `error_analysis.md`), but freezing was still the right call for the reasons above, not something we'd undo.
- Two branches, not some four-branch FFT+DCT+PRNU setup. Something that finishes training on available compute and has a working demo beats a fancier idea that doesn't run in time.

## How the day actually went

Rough split: Ngiam on the data pipeline (fetching WildFake/SID_Set, building the manifest), Letao on the semantic branch and training loop, Aaron on the frequency branch and fusion, Aarav on robustness eval and calibration. Full breakdown's in `ROLES.md`.

We ended up scoping this down more than once — the timeline got compressed early on, and we deliberately cut demo/repo-polish work out of the same-day plan so it wasn't competing with the actual pipeline for time. Good call in hindsight.

One honest note: if the results/timeline you see elsewhere in this repo don't exactly match what's outlined above, that's because plans don't survive contact with a real training run. This doc is what we planned going in — the main `README.md` and `docs/error_analysis.md` are what actually happened and what we actually found.
