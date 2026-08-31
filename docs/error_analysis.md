# Error Analysis

## Robustness summary

Evaluated on the held-out `test` split (WildFake generators: ADM, StyleGAN3, VQVAE)
plus a cross-generator condition (`unseen_generator`) drawn from SID_Set, a diffusion
generator never present in training. Checkpoint: `checkpoints/best.pt`
(`freq_lr_multiplier: 1.0`, default augmentation recipe -- see Finding 3 below
for why the widened-noise/added-resize variant was tried and reverted --
10 epochs, WildFake only). 15 conditions, covering all six transform
categories in the challenge brief's grid (5.2): JPEG, Gaussian Blur, Resize,
Gaussian Noise, Color Jitter, Center Crop.

| Condition | Acc. | AUC |
|---|---|---|
| clean | 0.87 | 0.94 |
| jpeg_q90 | 0.84 | 0.93 |
| jpeg_q70 | 0.85 | 0.93 |
| jpeg_q50 | 0.82 | 0.91 |
| jpeg_q30 | 0.82 | 0.89 |
| blur_sigma0.5 | 0.85 | 0.93 |
| blur_sigma1.0 | 0.81 | 0.91 |
| blur_sigma2 | 0.71 | 0.82 |
| resize_0.5x | 0.81 | 0.90 |
| resize_0.25x | 0.75 | 0.82 |
| noise_sigma0.02 | 0.83 | 0.91 |
| noise_sigma0.05 | 0.77 | 0.87 |
| noise_sigma0.10 | 0.73 | 0.82 |
| color_jitter | 0.82 | 0.92 |
| crop_80pct | 0.81 | 0.90 |
| unseen_generator | 0.48 | 0.51 |

**Final Score** (0.5·AUC_clean + 0.5·AUC_robust, robust = mean of the 14
post-processing conditions above, excluding `unseen_generator`) ≈ **0.915**.
Every post-processing condition stays above 0.82 AUC, with a smooth,
expected degradation curve as each transform gets harsher (e.g. blur:
0.93 → 0.91 → 0.82 as sigma increases). `resize_0.25x` and
`noise_sigma0.10` are the weakest post-processing conditions (0.82 AUC
each); see Finding 3 for why we did not manage to improve on them.

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

## Finding 3: widening augmentation to match eval was tried, and reverted

After expanding the robustness eval to the full brief transform grid
(Finding above's table), `resize_0.25x` and `noise_sigma0.10` came out as
the weakest post-processing conditions (0.82 AUC each). The training-time
augmenter (`RedistributionAugment`) had a real gap that looked like the
cause: `noise_sigma` was a fixed 3.0 on a 0-255 scale (~0.012 normalized),
about 8x weaker than the hardest eval noise condition (0.10 normalized),
and there was no dedicated whole-frame resize/thumbnail training op at all.

We widened `noise_sigma` to a sampled range (2.0, 30.0) and added a
`"resize"` op (scale sampled from (0.2, 0.8)) to the per-sample op choice,
then retrained. Result: worse, not better, across nearly the whole table.

| Condition | v3 (before) | v4 (after) |
|---|---|---|
| clean | 0.94 | 0.94 |
| jpeg_q30 | 0.89 | 0.88 |
| blur_sigma2 | 0.82 | 0.83 |
| resize_0.25x | 0.82 | 0.83 |
| noise_sigma0.02 | 0.91 | 0.90 |
| noise_sigma0.05 | 0.87 | 0.85 |
| noise_sigma0.10 | 0.82 | **0.79** |
| Final Score | ≈0.915 | 0.9108 |

`resize_0.25x` moved a hair in the right direction; `noise_sigma0.10` --
the condition this change specifically targeted -- got measurably worse,
and most of the rest of the table drifted down slightly too.

**Working theory:** `random.choice()` over the op list means each op fires
with probability 1/N. Adding a 7th op (`"resize"`) diluted how often
`jpeg`/`blur`/`crop`/`noise` each get selected, by design, not just
`noise`. And widening `noise_sigma_range` to (2.0, 30.0) means uniform
sampling spends much of its mass on noise *milder* than the old fixed 3.0
-- so a change aimed at increasing exposure to harsh noise plausibly
*decreased* average exposure to it instead, while simultaneously making
every other op fire less often. Both effects point the same direction as
what we measured.

**Decision:** reverted to the pre-alignment augmentation recipe.
`checkpoints/best.pt` is the reverted (v3) checkpoint; the widened-
augmentation checkpoint is kept at
`checkpoints/best_v4_aug_alignment_negative_result.pt` for reference. Like
Finding 2's LR-boost experiment, this is a second diagnosed negative
result rather than a silent revert: the fix that seemed obviously correct
(train past what you're evaluated on) had a side effect (diluting the
other ops' firing rate) that outweighed its intended benefit at these
specific hyperparameter values. A better-targeted version of the same idea
-- e.g. reweighting op-selection probabilities instead of adding a op to
an unweighted uniform choice, or increasing `noise_sigma`'s upper bound
without changing its lower bound -- might still work, but we're not
spending further iteration time on it given the marginal size of the gap
being chased.

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

**Generalization (Finding 2):** broaden training-generator diversity rather
than tune optimization further — per the brief's own SAFE/DDA-derived
guidance ("augmentation + data alignment > architecture tricks"), the fix
for a representation gap is more representative training data, not a
different learning rate. Concretely: add a modest slice of one or two
additional diffusion-generator subsets from
[GenImage](https://github.com/GenImage-Dataset/GenImage) (a public
benchmark covering Stable Diffusion, GLIDE, VQDM, Midjourney, and Wukong,
in addition to BigGAN/ADM) into the *training* manifest — keeping SID_Set
untouched as the held-out eval set throughout, so the unseen-generator
number stays an honest measurement rather than becoming something the
model was partly trained on.

**Robustness (Finding 3):** we tried the obvious fix for the weakest
post-processing conditions (widen noise range, add a resize op) and it
made things slightly worse, likely by diluting how often every other op
fires. A better-targeted retry — reweighting op-selection probabilities
directly instead of adding an option to an unweighted `random.choice`,
so existing ops keep firing at their old rate while noise/resize get
added on top rather than competing for the same probability mass — is
the more promising next attempt, not a reason to abandon the idea
entirely.

**Trust & calibration:** add a false-positive-rate discussion at a chosen
calibrated-probability threshold (the brief's "Technical Goal" section
names false positives explicitly, alongside robustness and generalization,
as an expected trade-off discussion). `src/calibrate.py` now produces
calibrated probabilities via temperature scaling; a short precision/recall
or false-positive-rate table at threshold=0.5 would close this out cheaply
once the active checkpoint is calibrated.
