"""NFIQ 2 quality scoring (ISO/IEC 29794-4).

Scores run 0-100, higher is better. The thesis reports the *distribution shift*
between conditions rather than a mean, because NFIQ 2's relationship to
matching performance is not monotonic (Schuch, Schulz & Busch, 2018) -- a
higher mean is suggestive, not sufficient.

One hard rule, from measuring the corpus: NFIQ 2 assumes 500 dpi input, so
scores are only comparable between images at the same effective resolution.
Never compare a score from a ~200 dpi set against one from a true 500 dpi set.

The binary is invoked in batch mode; per-image invocation costs more in
process spawn than in scoring.
"""
from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Nfiq2Score", "find_nfiq2", "score_images", "score_summary"]

ROOT = Path(__file__).resolve().parents[3]

FAILED = -1  # sentinel: NFIQ 2 could not score this image


@dataclass(frozen=True)
class Nfiq2Score:
    path: str
    score: int  # 0-100, or FAILED
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.score != FAILED


@lru_cache(maxsize=1)
def find_nfiq2() -> Path:
    """Locate the nfiq2 executable."""
    exe = "nfiq2.exe" if os.name == "nt" else "nfiq2"
    for d in (
        Path(os.environ["FPE_NFIQ2_BIN"]) if os.environ.get("FPE_NFIQ2_BIN") else None,
        ROOT / "tools" / "bin",
    ):
        if d is not None and (d / exe).is_file():
            return d / exe
    found = shutil.which("nfiq2")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"nfiq2 not found in tools/bin or PATH; set FPE_NFIQ2_BIN to its directory."
    )


def score_images(
    images: Sequence[str | Path], *, threads: int = 2, timeout: int = 3600
) -> list[Nfiq2Score]:
    """Score a batch of images. Unscoreable images come back as FAILED, not dropped.

    `threads` stays low by default: this machine has 7.8 GB of RAM and the
    binding constraint is memory, not cores.
    """
    if not images:
        return []
    paths = [Path(p) for p in images]

    with tempfile.TemporaryDirectory(prefix="fpe_nfiq2_") as tmp:
        batch = Path(tmp) / "batch.txt"
        batch.write_text(
            "\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8"
        )
        cmd = [str(find_nfiq2()), "-f", str(batch), "-F"]
        if threads > 1:
            cmd += ["-j", str(threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"nfiq2 failed (exit {proc.returncode})\nstderr: {proc.stderr.strip()}"
        )

    scored: dict[str, Nfiq2Score] = {}
    reader = csv.DictReader(io.StringIO(proc.stdout))
    for row in reader:
        name = (row.get("Filename") or "").strip().strip('"')
        raw = (row.get("QualityScore") or "").strip()
        err = (row.get("OptionalError") or "").strip().strip('"')
        try:
            value = int(raw)
        except ValueError:
            value = FAILED
        scored[name] = Nfiq2Score(
            path=name, score=value, error=None if err in ("", "NA") else err
        )

    # Preserve input order, and represent anything NFIQ 2 did not report.
    out: list[Nfiq2Score] = []
    for p in paths:
        key = p.as_posix()
        out.append(
            scored.get(key)
            or scored.get(str(p))
            or Nfiq2Score(path=key, score=FAILED, error="not reported by nfiq2")
        )
    return out


def score_summary(scores: Iterable[Nfiq2Score]) -> dict[str, float]:
    """Distribution summary suitable for a registry row."""
    all_scores = list(scores)  # materialise first; the argument may be a generator
    values = np.array([s.score for s in all_scores if s.ok], dtype=np.float64)
    if values.size == 0:
        return {"nfiq2_n": 0, "nfiq2_failed": len(all_scores)}
    q = np.percentile(values, [25, 50, 75])
    return {
        "nfiq2_n": int(values.size),
        "nfiq2_failed": len(all_scores) - int(values.size),
        "nfiq2_mean": float(values.mean()),
        "nfiq2_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "nfiq2_q25": float(q[0]),
        "nfiq2_median": float(q[1]),
        "nfiq2_q75": float(q[2]),
        # NFIQ 2 below 30 is the band commonly treated as operationally weak.
        "nfiq2_frac_below_30": float((values < 30).mean()),
    }
