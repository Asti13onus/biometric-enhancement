"""Abstention: declining to reconstruct where the evidence was destroyed (plan U8).

Every published enhancement method reconstructs the whole image whether or not evidence
survives there, because none of them has any way to know. Ours does: the confidence head is
trained against the wear model's evidence map, so it estimates how much ridge evidence was
actually present at each pixel.

Two things follow, and the second is the one that moves the metric:

1. **Where confidence is low, write background instead of predicted ridges.** Handing the
   extractor a plausible invention is worse than handing it nothing -- absence is honest,
   invention becomes a minutia.
2. **Discard minutiae that land in declined regions, and count them.** This is the direct
   intervention on precision, and precision is the binding constraint in this literature
   because a spurious minutia is a false identity feature.

The coverage threshold is a free parameter swept at evaluation time, never a constant baked
into the model. Sweeping it produces the thesis's defining figure: precision and EER against
the fraction of the image the method is willing to reconstruct, on which every competing
method is a single point at coverage 1.0.

Coverage is always reported **over the foreground**, not over the whole image. Padding and
background are not decisions the method made, and counting them would let a tightly cropped
image report high coverage for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AbstentionConfig", "apply_coverage", "coverage_fraction", "filter_minutiae",
    "filter_xyt_file", "write_xyt",
]

BACKGROUND = 1.0
"""Declined pixels are written as background -- light, i.e. valley, not ink."""


@dataclass(frozen=True)
class AbstentionConfig:
    threshold: float = 0.5
    """Confidence below this is declined. Swept at evaluation time."""
    background: float = BACKGROUND
    dilate_px: int = 0
    """Optionally grow the declined region, to keep minutiae off the boundary."""


def apply_coverage(
    ridge: np.ndarray, confidence: np.ndarray, config: AbstentionConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Blank the ridge map where confidence is low.

    Returns (image, coverage) where `coverage` is 1 for reconstructed pixels and 0 for
    declined ones.
    """
    config = config or AbstentionConfig()
    if ridge.shape != confidence.shape:
        raise ValueError(
            f"ridge shape {ridge.shape} != confidence shape {confidence.shape}"
        )
    coverage = (np.asarray(confidence, dtype=np.float32) >= config.threshold)
    if config.dilate_px > 0:
        import cv2

        kernel = np.ones((config.dilate_px * 2 + 1,) * 2, np.uint8)
        coverage = cv2.erode(coverage.astype(np.uint8), kernel).astype(bool)

    image = np.where(coverage, np.asarray(ridge, dtype=np.float32), config.background)
    return image.astype(np.float32), coverage


def coverage_fraction(
    coverage: np.ndarray, foreground: np.ndarray | None = None
) -> float:
    """Fraction of the *foreground* the method was willing to reconstruct."""
    coverage = np.asarray(coverage).astype(bool)
    if foreground is None:
        return float(coverage.mean())
    foreground = np.asarray(foreground).astype(bool)
    total = int(foreground.sum())
    if total == 0:
        return 0.0
    return float((coverage & foreground).sum() / total)


def filter_minutiae(minutiae, coverage: np.ndarray) -> tuple[list, int]:
    """Drop minutiae that fall in declined regions. Returns (kept, dropped count).

    A minutia outside the image bounds is dropped too: it cannot be verified against the
    coverage decision, and keeping it would mean reporting a feature the method never
    claimed to see.
    """
    coverage = np.asarray(coverage).astype(bool)
    height, width = coverage.shape
    kept = []
    for m in minutiae:
        if 0 <= m.y < height and 0 <= m.x < width and coverage[m.y, m.x]:
            kept.append(m)
    return kept, len(minutiae) - len(kept)


def filter_xyt_file(xyt: str | Path, keep: np.ndarray) -> int:
    """Filter a mindtct `.xyt` template in place against a top-origin `keep` mask.

    Returns the number of minutiae dropped. **mindtct's default `.xyt` measures y from
    the bottom of the image** (NIST internal representation; `-m1` would switch to ANSI
    top-origin, and we do not pass it). Measured on FVC2004: read bottom-origin, 100% of
    minutiae land on the finger; read top-origin, 90.6%. The mask is therefore flipped
    here rather than trusting every caller to remember.
    """
    from fpe.metrics.nbis import read_xyt

    minutiae = read_xyt(xyt)
    kept, dropped = filter_minutiae(minutiae, np.asarray(keep)[::-1])
    if dropped:
        write_xyt(kept, xyt)
    return dropped


def write_xyt(minutiae, path: str | Path) -> Path:
    """Write minutiae back in NBIS `.xyt` form so bozorth3 can read them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{m.x} {m.y} {m.theta} {m.quality}" for m in minutiae]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return path
