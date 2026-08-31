# Error Analysis

## Robustness summary

We ran this on our held-out `test` split (WildFake's ADM, StyleGAN3, VQVAE generators) plus `unseen_generator`, which is SID_Set -- a generator the model never saw during training. Checkpoint is `checkpoints/best.pt` (`freq_lr_multiplier: 1.0`, the default augmentation recipe -- see Finding 3 for why we tried a widened version and reverted it, 10 epochs, WildFake only). There are 15 conditions in total, covering JPEG, Gaussian blur, resize, Gaussian noise, color jitter, and center crop.

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

Overall score (0.5·AUC_clean + 0.5·AUC_robust, robust = mean of the 14 post-processing conditions above, not counting `unseen_generator`) comes out to about **0.915**. We're happy with this half of the results -- every post-processing condition stays above 0.82 AUC, and it degrades the way you'd expect as each transform gets harsher (blur goes 0.93 → 0.91 → 0.82 as sigma climbs, for example). `resize_0.25x` and `noise_sigma0.10` are the weakest at 0.82 each -- we tried to fix specifically those two and it didn't go well, see Finding 3.

## Representative false positives and false negatives

Ran `scripts/find_error_examples.py` (`bash run.sh find_errors`) on the same
420-image clean/no-transform condition above -- no post-processing, so these
are the model's core errors, not something a blur or JPEG pass introduced.
22 of 210 real images got scored as AI-generated (10.5% false positive
rate), and 34 of 210 AI images got scored as real (16.2% false negative
rate).

The false negatives aren't spread evenly across generators:

| Generator | False negatives | Rate |
|---|---|---|
| adm | 4 / 70 | 5.7% |
| stylegan3 | 16 / 70 | 22.9% |
| vqvae | 14 / 70 | 20.0% |

adm (a diffusion model, same family WildFake trains on most) gets caught
almost every time; stylegan3 and vqvae slip past roughly 4x as often. Worth
reading alongside the `unseen_generator` result below -- that's *also* a
generator the model handles badly (SID_Set). Together they point the same
way: the model's strength tracks specific generator fingerprints it saw a
lot of in training, not a clean "diffusion vs. GAN vs. VQ" category
boundary.

### False positives -- real photos scored as AI-generated

| | | | |
|---|---|---|---|
| ![close-up cat face](examples/fp_1_real_pred1.00.jpg) pred 1.00 | ![chain-link fence over debris](examples/fp_2_real_pred0.98.jpg) pred 0.98 | ![close-up dirt and soil](examples/fp_3_real_pred0.96.jpg) pred 0.96 | ![glossy studio headshot](examples/fp_4_real_pred0.95.jpg) pred 0.95 |

All four of the model's most confident false positives are close-up,
texture-heavy, or unusually polished shots: a macro cat-face crop, a
chain-link fence over debris, a dirt/soil close-up, and a glossy studio
headshot with soft bokeh and near-perfect skin. None of these are "normal"
snapshot compositions -- they're either extreme macro texture (fur, dirt,
wire mesh) or the kind of over-lit, over-smooth portrait that looks like
stock photography. Our read: the model has picked up something closer to
"looks too clean / too textured to be a candid photo" than an actual
generative-artifact signal, and unusual real photography trips that same
heuristic.

### False negatives -- AI images scored as real

| | | | |
|---|---|---|---|
| ![fox portrait, stylegan3](examples/fn_1_stylegan3_pred0.00.jpg) pred 0.00, stylegan3 | ![corporate headshot, stylegan3](examples/fn_2_stylegan3_pred0.01.jpg) pred 0.01, stylegan3 | ![baseball batter, vqvae](examples/fn_3_vqvae_pred0.01.jpg) pred 0.01, vqvae | ![portrait in teal jacket, stylegan3](examples/fn_4_stylegan3_pred0.01.jpg) pred 0.01, stylegan3 |

Close to the mirror image of the false positives: a fox portrait, two
corporate-style headshots, and a baseball action shot -- all ordinary,
mundane compositions with nothing visually "off." Three of the four are
StyleGAN3, which is more or less built for exactly this (photorealistic
human faces with natural framing and lighting), and it shows: the model
seems to key off surface polish/unusualness more than any deeper generative
signature, so an AI image that happens to look like a boring real photo
slips through with high confidence.

Full scored results (all 8 examples plus the per-generator breakdown) are
in `docs/error_examples.json`; the thumbnails themselves are in
`docs/examples/`.

## Finding 1: robustness to post-processing is solid, cross-generator
generalization is not

Accuracy degrades gracefully under JPEG recompression, blur, and cropping —
AUC stays above 0.82 even at JPEG q30 and blur σ=2, the two harshest
conditions tested. This is the direct result of training-time augmentation
(`src/data/augmentations.py`) explicitly simulating these corruptions per the
SAFE (KDD 2025) and DDA (NeurIPS 2025) insights.

`unseen_generator` is a different story: AUC 0.51, accuracy 0.48 — chance
level. The model cannot distinguish real images from SID_Set's diffusion
outputs at all, despite performing well on WildFake's own generators (which
include ADM, itself a diffusion model). This means the failure isn't simply
"GANs vs diffusion" — it's specific to SID_Set's particular generator
fingerprint, consistent with the literature on detector overfitting to
generator-specific artifacts rather than a general "AI-generated-ness"
signal. Cross-generator generalization is a separate problem from robustness,
and the failure here is more severe than we'd hoped for.

## Finding 2: the frequency branch's role in this failure

The whole point of the hybrid design is that the frequency branch is supposed to catch something the CLIP branch misses. So we built a diagnostic (`scripts/check_frequency_branch.py`) to actually check that, rather than just assuming it because the val AUC looked fine. Two checks: a gradient check (is the frequency branch even getting a real learning signal?) and an ablation (does zeroing it out before fusion actually change any predictions?).

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

So undertraining wasn't the (whole) explanation for why the frequency branch wasn't contributing. Giving it more room to learn didn't make it more general, it just made it more confidently wrong about WildFake specifically. We went back to `freq_lr_multiplier: 1.0` (that's what `checkpoints/best.pt` is) since it's the safer, do-no-harm option; the boosted checkpoint is still around at `checkpoints/best_v2_freq_lr5x_negative_result.pt` if anyone wants to poke at it. We're counting this as a real result even though it's negative -- we now actually know *why* the frequency branch isn't helping generalization (it's a representation problem, not a training-budget problem), which rules out the cheap fix and points at what would actually help instead: more generator-diverse training data, not more optimization pressure on the branch we already have.

## Finding 3: widening augmentation to match eval was tried, and reverted

After expanding the robustness eval to the full transform grid
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
| Overall score | ≈0.915 | 0.9108 |

`resize_0.25x` moved a hair in the right direction; `noise_sigma0.10` --
the condition this change specifically targeted -- got measurably worse,
and most of the rest of the table drifted down slightly too.

Our best guess why: `random.choice()` over the op list means each op fires with probability 1/N, so adding a 7th option (`"resize"`) quietly diluted how often `jpeg`/`blur`/`crop`/`noise` each get picked, not just `noise`. And widening `noise_sigma_range` to (2.0, 30.0) means most of what gets sampled from that range is actually milder than the old fixed 3.0 -- so a change meant to increase exposure to harsh noise may have decreased average exposure to it instead, on top of every other op also firing less often. Both of those point the same direction as what we actually measured, so we're fairly confident this is the real explanation and not just noise in the results.

We reverted to the pre-alignment augmentation recipe. `checkpoints/best.pt` is that reverted (v3) checkpoint; the widened-augmentation one is kept around at `checkpoints/best_v4_aug_alignment_negative_result.pt` in case it's useful later. Same as the LR-boost experiment in Finding 2, we're writing this up rather than quietly deleting it, because the fix that seemed obviously right (train on harder conditions than you're evaluated on) had a side effect we didn't anticipate that outweighed the intended benefit, at least at these specific numbers. A more careful version of the same idea -- reweighting how often each op gets picked instead of just adding a new option to an unweighted choice, or only raising `noise_sigma`'s upper bound without touching the lower one -- might still work. We just didn't have time to chase it further given how small the gap we were trying to close actually was.

## Trade-offs

- **Robustness vs. clean accuracy:** not really a trade-off here — the
  augmentation pipeline achieves both simultaneously (clean 0.94, JPEG q30
  0.89).
- **Generalization vs. specialization:** the core limitation of this
  model. The model is well-specialized to WildFake's generator family
  and does not generalize to SID_Set's diffusion outputs. Diagnosed as a
  representation gap (verified via ablation), not a fusion/architecture bug.
- **Complexity vs. feasibility:** the frequency branch adds real complexity
  for a currently-negligible in-domain benefit and a measured *cost* on
  cross-generator generalization once given more learning capacity. Kept in
  the architecture because it does no harm at `freq_lr_multiplier: 1.0`, but
  its value is not yet demonstrated on this dataset.

## What we'd do with more time

**Generalization (Finding 2):** broaden training-generator diversity rather
than tune optimization further. A representation gap needs more
representative training data, not a different learning rate. Concretely: add
a modest slice of one or two
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
calibrated-probability threshold. `src/calibrate.py` now produces
calibrated probabilities via temperature scaling; a short precision/recall
or false-positive-rate table at threshold=0.5 would close this out cheaply
once the active checkpoint is calibrated.
