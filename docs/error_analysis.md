# Error Analysis

## Robustness summary

Evaluated on the held-out `test` split (WildFake generators: ADM, StyleGAN3, VQVAE)
plus a cross-generator condition (`unseen_generator`) drawn from SID_Set, a diffusion
generator never present in training. Checkpoint: `checkpoints/best.pt`
(`freq_lr_multiplier: 1.0`, 10 epochs, WildFake only).

| Condition | Acc. | AUC |
|---|---|---|
| clean | 0.87 | 0.94 |
| jpeg_q90 | 0.84 | 0.93 |
| jpeg_q70 | 0.85 | 0.93 |
| jpeg_q50 | 0.82 | 0.91 |
| jpeg_q30 | 0.82 | 0.89 |
| blur_sigma2 | 0.71 | 0.82 |
| crop_80pct | 0.80 | 0.88 |
| unseen_generator | 0.48 | 0.51 |

**Final Score** (0.5·AUC_clean + 0.5·AUC_robust, robust = mean of the six
post-processing conditions above, excluding `unseen_generator`) = **0.9156**.

## Finding 1: robustness to post-processing is solid; cross-generator
## generalization is not

Accuracy degrades gracefully under JPEG recompression, blur, and cropping —
AUC stays above 0.82 even at JPEG q30 and blur σ=2, the two harshest
conditions tested. This is the direct result of training-time augmentation
(`src/data/augmentations.py`) explicitly simulating these corruptions per the
SAFE (KDD 2025) and DDA (NeurIPS 2025) insights cited in the challenge brief.

`unseen_generator` is a different story: AUC 0.51, accuracy 0.48 — chance
level. The model cannot distinguish real images from SID_Set's diffusion
outputs at all, despite performing well on WildFake's own generators (which
include ADM, itself a diffusion model). This means the failure isn't simply
"GANs vs diffusion" — it's specific to SID_Set's particular generator
fingerprint, consistent with the literature on detector overfitting to
generator-specific artifacts rather than a general "AI-generated-ness"
signal. This is expected in kind (the challenge brief names cross-generator
generalization as one of the two defining hard problems, alongside
robustness) but more severe in degree than we'd hoped for.

## Finding 2: the frequency branch's role in this failure

Since the hybrid model's premise is that a frequency-domain branch captures
signal a semantic (CLIP) branch misses, we built a dedicated diagnostic
(`scripts/check_frequency_branch.py`) with two checks: a **gradient check**
(does the frequency branch receive real learning signal?) and an **ablation**
(does zeroing its embedding before fusion change predictions?).

On the baseline checkpoint (`freq_lr_multiplier: 1.0`):

| | in-domain (val) | blur_sigma2 | unseen_generator |
|---|---|---|---|
| AUC with frequency branch | 0.9386 | 0.8189 | 0.5070 |
| AUC with frequency zeroed | 0.9383 | 0.8206 | 0.5027 |
| delta | +0.0002 | -0.0017 | +0.0044 |

All three deltas are near zero — the frequency branch is barely influencing
predictions anywhere we've checked, including blur, the one condition it
was specifically designed to help with (recovering high-frequency artifacts
that blur destroys). That rules out "it only fails to generalize
cross-generator" as the story; the more accurate statement is that the
fusion layer isn't leaning on this branch's output under any condition
tested so far. The gradient check explains why: its gradient norm (0.050)
ran ~65x smaller than the semantic projection head's (3.28) on a single
backward pass, consistent with a known "modality imbalance" pattern in
multi-branch fusion — an ~88M-parameter pretrained CLIP branch produces a
strong, low-loss signal almost immediately, so gradient descent naturally
routes most useful update magnitude through it while a small,
randomly-initialized CNN branch gets comparatively tiny updates each step.

### Experiment: does more capacity fix it?

We tested the natural hypothesis — give the frequency branch its own higher
learning rate (`freq_lr_multiplier: 5.0`, a separate AdamW param group) so
its small gradients still translate into meaningful updates over the same
10-epoch budget — and retrained.

| | in-domain (val) | unseen_generator |
|---|---|---|
| AUC with frequency branch | 0.9448 | 0.5226 |
| AUC with frequency zeroed | 0.9458 | 0.5296 |
| delta | -0.0010 | **-0.0071** |

The headline robustness table barely moved (clean AUC 0.93 vs 0.94,
unseen-generator AUC 0.52 vs 0.51) — because CLIP alone is already
near-ceiling on this data, the table is not a sensitive instrument for the
frequency branch's contribution either way. The ablation delta is the real
signal, and it flipped sign: with the LR boost, *zeroing* the frequency
branch now *improves* unseen-generator AUC. The branch stopped being inert,
but what it learned actively hurts cross-generator transfer — consistent
with it picking up WildFake-specific frequency artifacts (compression
signature, generator-specific noise patterns) rather than a signal that
transfers to a new generator family.

**Conclusion:** undertraining was not the (whole) explanation for the
frequency branch's low contribution. Giving it more effective learning
capacity didn't produce a more general representation — it produced a more
confidently WildFake-specific one. We reverted to `freq_lr_multiplier: 1.0`
(`checkpoints/best.pt`) as the safer, do-no-harm checkpoint; the boosted
checkpoint is retained at `checkpoints/best_v2_freq_lr5x_negative_result.pt`
for reference. This is a negative result, but a diagnosed one: we know
*why* the frequency branch isn't helping cross-generator generalization
(a representation-quality problem, not a training-budget problem), which
rules out the cheap fix and points toward what would actually be required —
generator-diverse training data, not more optimization pressure on the
existing branch.

## Trade-offs (per the brief's own framing)

- **Robustness vs. clean accuracy:** not really a trade-off here — the
  augmentation pipeline achieves both simultaneously (clean 0.94, JPEG q30
  0.89).
- **Generalization vs. specialization:** the core limitation of this
  submission. The model is well-specialized to WildFake's generator family
  and does not generalize to SID_Set's diffusion outputs. Diagnosed as a
  representation gap (verified via ablation), not a fusion/architecture bug.
- **Complexity vs. feasibility:** the frequency branch adds real complexity
  for a currently-negligible in-domain benefit and a measured *cost* on
  cross-generator generalization once given more learning capacity. Kept in
  the architecture (per the brief's hybrid-design guidance and because it
  does no harm at `freq_lr_multiplier: 1.0`), but its value is not yet
  demonstrated on this dataset.

## What we'd do with more time

Broaden training-generator diversity rather than tune optimization further —
per the brief's own SAFE/DDA-derived guidance ("augmentation + data
alignment > architecture tricks"), the fix for a representation gap is more
representative training data, not a different learning rate. Concretely:
add a modest slice of one or two additional diffusion-generator subsets
from [GenImage](https://github.com/GenImage-Dataset/GenImage) (a public
benchmark covering Stable Diffusion, GLIDE, VQDM, Midjourney, and Wukong,
in addition to BigGAN/ADM) into the *training* manifest — keeping SID_Set
untouched as the held-out eval set throughout, so the unseen-generator
number stays an honest measurement rather than becoming something the
model was partly trained on.

For blur specifically: the frequency branch doesn't appear to be carrying
blur robustness either (ablation delta -0.0017, Finding 2), so the higher-
leverage change there is tuning `RedistributionAugment` in
`src/data/augmentations.py` — e.g. weighting blur more heavily in the
per-sample op choice, or extending `blur_sigma_range` beyond the current
(0.3, 2.5) — rather than a frequency-branch architecture change.

Also worth adding before final submission: a false-positive-rate discussion
at a chosen calibrated-probability threshold (the brief's "Technical Goal"
slide names false positives explicitly, alongside robustness and
generalization, as an expected trade-off discussion). `src/calibrate.py`
now produces calibrated probabilities via temperature scaling; a short
precision/recall or false-positive-rate table at threshold=0.5 would close
this out cheaply once that checkpoint is calibrated.
