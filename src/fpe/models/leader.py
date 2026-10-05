"""LEADER as a second extraction chain (plan U11).

Spurious minutiae originate in extraction, so the thesis's claims must survive a change
of extractor. LEADER (pyfing's end-to-end minutiae network) replaces mindtct here,
emitting `.xyt` templates the existing bozorth3 chain matches unchanged. Comparisons are
strictly within-chain: LEADER templates against LEADER templates.

Conventions, converted explicitly: pyfing minutiae are top-origin pixels with directions
in radians from `arctan2` (pyfing/minutiae.py, head_atan2); mindtct's `.xyt` is
bottom-origin with degrees measured in the y-up frame. Hence y -> height - y and
theta -> -theta, and quality scales from [0, 1] to the integer [0, 100] bozorth3 expects.
"""

from __future__ import annotations

import math
from pathlib import Path

__all__ = ["LeaderExtractor"]


class LeaderExtractor:
    """Callable: image path -> cached `.xyt` template extracted by LEADER."""

    def __init__(self, cache_dir: str | Path, *, dpi: int = 500) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def __call__(self, image_path: str | Path) -> Path:
        from fpe.data.convert import cache_path, load_greyscale

        src = Path(image_path)
        out = cache_path(src, self.cache_dir, ".xyt")
        if out.is_file():
            return out

        image = load_greyscale(src)
        minutiae = self.extract(image)
        height = image.shape[0]
        lines = [
            f"{m.x} {height - m.y} {round(math.degrees(-m.direction)) % 360} "
            f"{max(0, min(100, round(m.quality * 100)))}"
            for m in minutiae
        ]
        tmp = out.with_suffix(".xyt.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
        tmp.replace(out)
        return out

    def extract(self, image) -> list:
        """pyfing Minutia records (x, y, direction, type, quality), CPU-pinned."""
        from fpe.models.pyfing_runtime import load_pyfing, pyfing_cpu

        pf = load_pyfing()
        with pyfing_cpu():
            return pf.minutiae_extraction(image, dpi=self.dpi, method="LEADER")
