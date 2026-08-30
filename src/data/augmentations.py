"""
Training-time augmentation pipeline: "simulate redistribution during training."

Mental model (from the challenge brief): if a transformation can happen to a
real feed after upload (recompression, re-screenshotting, resizing, cropping
for a thumbnail, color correction), it MUST happen in the training pipeline,
or the model will learn signals that don't survive contact with the real world.

Design choices follow the two cited insights:
  - SAFE (KDD 2025): prefer CROPPING over down-sampling to preserve
    high-frequency artifacts; ColorJitter + RandomRotation to kill
    color/semantic shortcuts.
  - DDA (NeurIPS 2025): watch for frequency bias — don't let JPEG-ness itself
    become a spurious "fake" signal. Apply the *same* compression family to
    both real and fake images so the model can't shortcut on compression
    alone; align pixel-domain and frequency-domain statistics.
"""
import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip an image through JPEG at a given quality to inject real
    compression artifacts (not a synthetic blur approximation)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def random_crop_resize(img: Image.Image, min_frac: float = 0.7) -> Image.Image:
    """Crop (never just down-sample) to a random sub-region, then resize back.
    Cropping preserves local high-frequency detail that down-sampling erases —
    this is the SAFE-insight fix."""
    w, h = img.size
    frac = random.uniform(min_frac, 1.0)
    cw, ch = int(w * frac), int(h * frac)
    x0 = random.randint(0, max(0, w - cw))
    y0 = random.randint(0, max(0, h - ch))
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
    return cropped.resize((w, h), Image.BICUBIC)


def add_sensor_noise(img: Image.Image, sigma: float = 3.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_noise_01(img: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise with sigma expressed on a normalized [0,1]
    pixel scale, matching the challenge brief's parameterization (sigma =
    0.02 / 0.05 / 0.10) directly -- unlike add_sensor_noise above, which
    uses a raw 0-255 scale for the training-time augmenter. Real-world
    analog: low-light sensor noise."""
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def downscale_upscale(img: Image.Image, scale: float) -> Image.Image:
    """Resize down to `scale` of original resolution, then back up to the
    original size -- the brief's "Resize" condition (scale 0.5x / 0.25x
    then upscale), real-world analog: thumbnail generation. Distinct from
    random_crop_resize (which crops a sub-region) -- this shrinks the whole
    frame, destroying fine detail uniformly rather than at the edges."""
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def color_jitter_fixed(img: Image.Image, factor: float = 1.2) -> Image.Image:
    """Deterministic +/-20% brightness/contrast/saturation shift -- the
    brief's "Color Jitter" condition. Fixed (not random) because eval
    conditions must be reproducible for a fixed test set, unlike the
    training-time T.ColorJitter which is intentionally randomized per
    sample. Real-world analog: filter apps, auto-enhance."""
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(factor)
    img = ImageEnhance.Contrast(img).enhance(factor)
    img = ImageEnhance.Color(img).enhance(factor)  # PIL's "Color" = saturation
    return img


def simulate_rescreenshot(img: Image.Image) -> Image.Image:
    """Approximate a screenshot-of-a-screenshot: down-resize, slight blur,
    re-encode at moderate JPEG quality, upscale back."""
    w, h = img.size
    small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BICUBIC)
    small = gaussian_blur(small, sigma=0.6)
    small = jpeg_recompress(small, quality=random.randint(55, 80))
    return small.resize((w, h), Image.BICUBIC)


@dataclass
class RedistributionAugment:
    """Applied at TRAIN time only, with probability `p` per-sample, one
    randomly chosen transform per application (keeps clean/augmented mix
    balanced rather than stacking every corruption on every sample)."""
    p: float = 0.7
    jpeg_quality_range: tuple = (30, 95)
    blur_sigma_range: tuple = (0.3, 2.5)
    crop_min_frac: float = 0.7
    noise_sigma: float = 3.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        op = random.choice(["jpeg", "blur", "crop", "noise", "rescreenshot", "none"])
        if op == "jpeg":
            q = random.randint(*self.jpeg_quality_range)
            return jpeg_recompress(img, q)
        if op == "blur":
            s = random.uniform(*self.blur_sigma_range)
            return gaussian_blur(img, s)
        if op == "crop":
            return random_crop_resize(img, self.crop_min_frac)
        if op == "noise":
            return add_sensor_noise(img, self.noise_sigma)
        if op == "rescreenshot":
            return simulate_rescreenshot(img)
        return img


class TrainTransform:
    """Picklable train-time transform. This MUST be a class, not a closure —
    DataLoader with num_workers > 0 pickles the whole Dataset (including
    whatever transform it holds) to ship to worker processes, and Python's
    pickle cannot serialize a function defined inside another function (no
    module-level name to look it up by). A class instance pickles fine as
    long as its attributes do. Composed with the augmenter BEFORE
    resizing/normalizing so corruptions act on full-resolution pixels,
    matching what happens on a real feed."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.redistribution = RedistributionAugment()
        self.color_geo = T.Compose([
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            T.RandomApply([T.RandomRotation(10)], p=0.3),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self.redistribution(img)
        img = self.color_geo(img)
        return img


def build_train_transform(image_size: int = 224):
    return TrainTransform(image_size)


# Fixed, named conditions for the EVAL-time robustness test set. Keep this in
# sync with scripts/build_robustness_testset.py and src/evaluate.py so the
# robustness table columns line up with what the slide deck expects.
# Named per the challenge brief's transform grid (5.2): JPEG q=90/70/50/30,
# Gaussian Blur sigma=0.5/1.0/2.0, Resize 0.5x/0.25x-then-upscale, Gaussian
# Noise sigma=0.02/0.05/0.10, Color Jitter +/-20%, Center Crop 80%. The brief
# says "a subset" is acceptable, but we cover all six categories so the
# robustness table matches the official grid rather than only the slide
# deck's illustrative subset.
EVAL_CONDITIONS = {
    "clean": lambda img: img,
    "jpeg_q90": lambda img: jpeg_recompress(img, 90),
    "jpeg_q70": lambda img: jpeg_recompress(img, 70),
    "jpeg_q50": lambda img: jpeg_recompress(img, 50),
    "jpeg_q30": lambda img: jpeg_recompress(img, 30),
    "blur_sigma0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_sigma1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_sigma2": lambda img: gaussian_blur(img, 2.0),
    "resize_0.5x": lambda img: downscale_upscale(img, 0.5),
    "resize_0.25x": lambda img: downscale_upscale(img, 0.25),
    "noise_sigma0.02": lambda img: gaussian_noise_01(img, 0.02),
    "noise_sigma0.05": lambda img: gaussian_noise_01(img, 0.05),
    "noise_sigma0.10": lambda img: gaussian_noise_01(img, 0.10),
    "color_jitter": lambda img: color_jitter_fixed(img, 1.2),
    "crop_80pct": lambda img: random_crop_resize(img, 0.8),
}
