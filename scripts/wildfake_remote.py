"""
Remote partial-ZIP reader for ModelScope-hosted datasets.

WildFake is 1.29 TB and its download unit is a whole archive (GAN_based.zip
alone is 47 GB). We need ~4k specific images. ModelScope's CDN honours HTTP
range requests, and a ZIP's central directory lives at the END of the file —
so we can read the index remotely and then fetch only the byte ranges of the
members we actually want. Total transfer drops from ~89 GB to well under 1 GB.

Also holds the image normalization used when materializing images, so the
spec builder and the teammate-facing fetcher cannot drift apart — if these
two applied different resizing or JPEG quality, everyone would silently be
training on different pixels.
"""
import io

import requests
from PIL import Image

API = ("https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake"
       "/repo?Revision=master&FilePath={}")

TARGET_SIZE = 256
JPEG_QUALITY = 95


def resolve(path: str, timeout: int = 60):
    """Resolve a repo path to its CDN URL + size.

    The CDN URL embeds an expiring `auth_key`, so this must be called fresh
    each run rather than cached anywhere on disk.
    """
    r = requests.head(API.format(path), allow_redirects=False, timeout=timeout)
    r.raise_for_status()
    cdn = r.headers["Location"]
    size = int(requests.head(cdn, timeout=timeout).headers["content-length"])
    return cdn, size


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests.

    zipfile.ZipFile only needs seek/read/tell, so this is enough to hand it a
    6-50 GB remote archive and have it read just the central directory.
    """

    def __init__(self, url: str, size: int, retries: int = 4):
        self.url, self.size, self.pos = url, size, 0
        self.retries = retries
        self.session = requests.Session()

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        headers = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url, headers=headers, timeout=180)
                if r.status_code in (200, 206):
                    self.pos += len(r.content)
                    return r.content
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = repr(e)
        raise IOError(f"range read failed after {self.retries} tries: {last}")


def normalize(raw: bytes) -> bytes:
    """Center-crop to square, resize to TARGET_SIZE, re-encode as JPEG.

    Why this exists: in the raw corpus the reals are 100% JPEG at ~200px and
    ADM/DDIM are 100% PNG at 256px. Either of those — container format or
    resolution — is a perfect label predictor that has nothing to do with how
    the image was generated, and the frequency branch would learn it in one
    epoch. Forcing every image through identical crop/resize/encode removes
    both giveaways.

    Center-crop rather than a plain resize because the reals are often
    non-square while the fakes are square: stretching only one class to fit
    would leave a class-dependent resampling signature, which is the same
    problem wearing a different hat.

    Residual imperfection, worth stating in the write-up: the reals were
    already JPEG, so they end up double-compressed while the fakes are
    compressed once. That cannot be undone here — only reduced. The random
    JPEG recompression applied to BOTH classes in training augmentation
    (docs/PLAN.md §3) is what's meant to swamp the remainder.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
