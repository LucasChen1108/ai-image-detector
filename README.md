# PixelProof

PixelProof detects AI-generated images after common real-world transformations. Four of us built it over a weekend, so don't expect production polish — expect a model that actually works, honestly-reported numbers, and a couple of experiments that flopped and taught us something anyway.

(Repo is still named `ai-image-detector` on GitHub — we didn't want to risk breaking any links we'd already shared by renaming it mid-project. PixelProof is the project name everywhere else: Devpost, this README, the demo video.)

## What this actually is

The task: tell real photos apart from AI-generated ones, and don't just do it on clean images — do it after the image has been through the stuff that actually happens on the internet (compressed, blurred, cropped, resized into a thumbnail, whatever).

We went with a hybrid model: combine a high-level "does this look real" signal with a low-level "does this have the statistical fingerprint of a generator" signal, because they tend to survive different kinds of damage.

- **Semantic branch** — a frozen CLIP ViT-B/32 (openai weights, loaded through `open_clip`) with a small trainable projection head on top. Frozen on purpose — we're not fine-tuning it, just training a lightweight head, which keeps CLIP's broad "seen a lot of images" generalization intact instead of overfitting it to our specific training generators.
- **Frequency branch** — a small 4-layer CNN over the log-magnitude FFT spectrum of the raw image. The idea is this branch catches generator artifacts (weird periodic frequency patterns from up-sampling, missing sensor noise) that don't show up as anything a human — or CLIP — would notice by looking at the picture.
- **Fusion head** — concat both embeddings, small MLP, one logit out, with a learned temperature so the output is an actual calibrated probability and not just a score that happens to rank things correctly.

~88M parameters total (mostly the frozen CLIP backbone), comfortably under the 2B cap.

## Data

- **[WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)** — our actual training set. Multi-generator (ADM, StyleGAN3, VQVAE + real), which matters a lot when testing generalization.
- **[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)** — never touched during training. Held out completely so we have an honest "have you seen this exact generator before" test (spoiler: no, and it shows — see Results).
- **[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)** — used once, day one, just to prove the whole pipeline runs end to end before we committed hours to a real WildFake run. Not part of the final model.

One thing worth calling out because it took us a while to catch: WildFake's real images are basically all JPEG and its fakes are basically all PNG, which means a model can cheat by learning "is this a PNG" instead of "is this AI-generated." Our data-fetch script re-encodes everything to the same format/resolution specifically to kill that shortcut — see `scripts/wildfake_remote.py`'s `normalize()` if you want the details.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then grab the data:

```bash
bash scripts/download_data.sh
```

This pulls WildFake + SID_Set and writes `manifest/manifest.csv` (the `split` column is what keeps SID_Set out of training — `train`/`val` splits only, SID_Set rows are tagged `heldout` and `ManifestDataset` never loads that split).

## Running it

The main entrypoint is `run.sh`:

```bash
bash run.sh train           # trains the model, saves checkpoints/best.pt
bash run.sh build_testset   # generates the JPEG/blur/resize/noise/color-jitter/crop test variants
bash run.sh evaluate        # runs the full robustness table + overall score
bash run.sh calibrate       # fits temperature scaling for calibrated probabilities
```

The inference script takes a folder and writes a JSON file of predictions:

```bash
bash run.sh predict --input_dir path/to/some/images --out predictions.json
```

Output looks like `[{"image_path": "...", "pred": 0.87}, ...]` where `pred` is the probability the image is AI-generated.

## Results

Ran on our held-out `test` split (WildFake generators the model trained on) plus `unseen_generator`, which is SID_Set — a generator the model has never seen:

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
| **unseen_generator** | **0.48** | **0.51** |

Overall score (0.5 × AUC_clean + 0.5 × mean of the post-processing rows) ≈ **0.915**.

We're pretty happy with the robustness half of this — every single post-processing condition stays above 0.82 AUC even at the harshest settings we tested (heaviest blur, heaviest noise, most aggressive downscale). We covered the full transform grid instead of cherry-picking a subset.

`unseen_generator` is the number that's not good — 0.51 AUC is basically a coin flip. More on that below, because we actually spent a decent chunk of time on it and it's kind of the most interesting part of this project, even though (especially because) we didn't fix it.

## What doesn't work, and what we learned trying to fix it

Full writeup with all the numbers is in [`docs/error_analysis.md`](docs/error_analysis.md), but the short version:

The model straight up cannot tell real from fake on SID_Set — a generator family it never saw during training — even though it does fine on WildFake's own generators (which include ADM, itself a diffusion model, so it's not simply "GANs vs diffusion"). Cross-generator generalization is simply still a hard problem here, and there isn't a silver bullet.

We built a diagnostic (`scripts/check_frequency_branch.py`) to figure out whether the frequency branch was even doing anything, since the whole point of the hybrid design was that it'd catch stuff CLIP misses. Turns out: not really. Zeroing it out barely changes predictions anywhere we tested — in-domain, under blur, or on the unseen generator. We tried two separate fixes for this:

1. Giving the frequency branch its own higher learning rate, so it'd stop getting drowned out by the (much bigger, pretrained) CLIP branch during training. Result: it did start mattering more, but what it learned made cross-generator performance *worse*, not better — it looks like more training capacity just let it lock onto WildFake-specific artifacts harder, not learn anything that transfers.
2. Widening our noise/resize augmentation ranges to better match the harder eval conditions. Result: also made things slightly worse across the board, probably because adding another random augmentation option diluted how often the existing ones fire.

Both are documented as negative results rather than quietly deleted, because "we tried the obvious thing and it didn't work, here's why" is a real finding, and figuring out *why* something we changed made things worse taught us more about the model than just reporting whatever number train.py spat out first.

If we had more time, the thing we'd actually try is adding a second, generator-diverse dataset (like [GenImage](https://github.com/GenImage-Dataset/GenImage)) into training — not touching SID_Set, which needs to stay untouched for this eval to mean anything — since our own diagnostics point at this being a data problem, not an architecture or optimization problem.

## Repo layout

```
src/models/clip_backbone.py    # frozen CLIP + trainable projection head
src/models/frequency_branch.py # FFT log-magnitude spectrum + small CNN
src/models/detector.py         # fusion head, calibrated predict_proba()
src/data/datasets.py           # manifest-driven dataset loader
src/data/augmentations.py      # training-time augmentation + the eval-condition transforms
src/train.py                   # training loop
src/evaluate.py                # robustness table + overall score
src/calibrate.py               # temperature scaling
src/predict.py                 # the directory-in, JSON-out inference script
scripts/fetch_dataset.py       # pulls WildFake + SID_Set, writes manifest/manifest.csv
scripts/build_robustness_testset.py  # generates the JPEG/blur/resize/noise/crop test variants
scripts/check_frequency_branch.py    # gradient + ablation checks on the frequency branch
configs/baseline_clip.yaml     # all the hyperparameters live here
docs/error_analysis.md         # the full writeup behind the "what doesn't work" section above
docs/PLAN.md, docs/ROLES.md    # our original planning docs, kept for the record
```

## Who did what

- **Ngiam** — data pipeline. Wrote the WildFake/SID_Set fetch scripts (including the byte-range remote zip/parquet reading so we didn't have to download the whole dataset locally), the manifest building, and caught/fixed the JPEG-vs-PNG format shortcut before it could quietly wreck the model.
- **Letao** — semantic branch and the training loop, and ended up owning most of the pipeline-wide bug hunting (a dual-tensor wiring bug that fed the wrong pixel values into the frequency branch, a DataLoader pickling issue, an MPS float64 crash, a QuickGELU mismatch) since it turned out train.py's bugs were also evaluate.py's and calibrate.py's bugs.
- **Aaron** — frequency branch and the fusion head, plus the calibration temperature parameter.
- **Aarav** — robustness eval, the calibration script, and the error-analysis writeup.

## License

MIT, see [`LICENSE`](LICENSE).
