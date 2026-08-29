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


def build_train_transform(image_size: int = 224):
    """Composed with the augmenter BEFORE resizing/normalizing so corruptions
    act on full-resolution pixels, matching what happens on a real feed."""
    redistribution = RedistributionAugment()
    color_geo = T.Compose([
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        T.RandomApply([T.RandomRotation(10)], p=0.3),
        T.RandomHorizontalFlip(p=0.5),
    ])

    def _transform(img: Image.Image):
        img = redistribution(img)
        img = color_geo(img)
        return img

    return _transform


# Fixed, named conditions for the EVAL-time robustness test set. Keep this in
# sync with scripts/build_robustness_testset.py and src/evaluate.py so the
# robustness table columns line up with what the slide deck expects.
EVAL_CONDITIONS = {
    "clean": lambda img: img,
    "jpeg_q90": lambda img: jpeg_recompress(img, 90),
    "jpeg_q70": lambda img: jpeg_recompress(img, 70),
    "jpeg_q50": lambda img: jpeg_recompress(img, 50),
    "jpeg_q30": lambda img: jpeg_recompress(img, 30),
    "blur_sigma2": lambda img: gaussian_blur(img, 2.0),
    "crop_80pct": lambda img: random_crop_resize(img, 0.8),
}
