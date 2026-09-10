"""Background texture bank, with split-disjointness enforced in code.

ChaLearn's degraded images were produced by compositing fingerprints onto background
textures. Reproducing that needs a texture source, and *which* source matters.

Cappelli's explicit criticism of DNUNets and FingerGAN is that they synthesised training
latents by compositing onto noise backgrounds **extracted from the test images**, leaking
test information into training. The literature review records this (section 9, finding
four) and `PROJECT_RULES.md` makes avoiding it a non-negotiable.

Two defences, both structural rather than procedural:

1. **The textures are not fingerprints.** We use the Describable Textures Dataset (DTD,
   5,640 images / 47 categories), the same source Kriangkhajorn et al. used for SFP. No
   fingerprint test image can leak through a texture that was never a fingerprint.
2. **Backgrounds are partitioned by split.** A texture used to build a training image can
   never appear in a validation or test image, because the partition is computed by hashing
   the filename -- deterministic, machine-independent, and impossible to get wrong by
   forgetting to pass a flag. `split="test"` simply cannot return a training texture.

Partition is by DTD *category*, not by file, so a training image never even sees a texture
from the same family as a test one -- a stricter guarantee than file-level splitting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
# Category-level split proportions; train gets whatever the other two do not.
_VAL_SHARE, _TEST_SHARE = 0.15, 0.15


def _stable_fraction(name: str) -> float:
    """Deterministic value in [0, 1) from a string -- stable across machines and runs,
    unlike hash(), which is salted per process."""
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_of_category(category: str) -> str:
    f = _stable_fraction(category)
    if f < _TEST_SHARE:
        return "test"
    if f < _TEST_SHARE + _VAL_SHARE:
        return "val"
    return "train"


@dataclass
class BackgroundBank:
    """Texture backgrounds for one split. Never returns another split's textures."""

    split: str
    paths: list[Path]
    categories: list[str]

    @classmethod
    def from_dtd(cls, dtd_root: str | Path, split: str) -> BackgroundBank:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        images = Path(dtd_root) / "dtd" / "images"
        if not images.is_dir():
            raise FileNotFoundError(
                f"DTD images not found at {images}. Expected the archive extracted so that "
                f"<dtd_root>/dtd/images/<category>/*.jpg exists."
            )
        categories = sorted(
            d.name for d in images.iterdir() if d.is_dir() and split_of_category(d.name) == split
        )
        paths = sorted(p for c in categories for p in (images / c).glob("*.jpg"))
        if not paths:
            raise RuntimeError(f"no DTD textures for split {split!r}")
        return cls(split=split, paths=paths, categories=categories)

    def __len__(self) -> int:
        return len(self.paths)

    def sample(self, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        """A greyscale texture in [0, 1] covering `shape` (height, width).

        Chooses a random crop at a random scale, then resizes to fit -- so the same
        texture file yields many distinct backgrounds without ever crossing a split.
        """
        height, width = shape
        for _ in range(8):  # a handful of DTD files are truncated; skip them
            path = self.paths[rng.integers(len(self.paths))]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size:
                break
        else:
            raise RuntimeError(f"could not read any texture from split {self.split!r}")

        # Random crop covering 40-100% of the shorter side, then fit to target.
        src_h, src_w = img.shape
        frac = rng.uniform(0.4, 1.0)
        crop = max(16, int(min(src_h, src_w) * frac))
        top = int(rng.integers(0, src_h - crop + 1))
        left = int(rng.integers(0, src_w - crop + 1))
        patch = img[top : top + crop, left : left + crop]

        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1, :]
        patch = np.rot90(patch, k=int(rng.integers(4)))

        out = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)

    def describe(self) -> str:
        return (
            f"DTD split={self.split}: {len(self.paths)} textures "
            f"across {len(self.categories)} categories"
        )


def split_report(dtd_root: str | Path) -> dict[str, dict[str, int]]:
    """Counts per split, for recording in the thesis and asserting disjointness in tests."""
    return {
        split: {
            "categories": len(bank.categories),
            "textures": len(bank.paths),
        }
        for split in SPLITS
        for bank in [BackgroundBank.from_dtd(dtd_root, split)]
    }
