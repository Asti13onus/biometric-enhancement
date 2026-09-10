"""Wrappers for NBIS `mindtct` (minutiae extraction) and `bozorth3` (matching).

These two binaries produce the thesis's headline number, so this module is
deliberately strict: it fails loudly rather than returning a plausible-looking
zero. A silent failure here would poison every downstream metric.

`mindtct` cannot read TIFF, BMP or PNG unless NBIS was built with PNG support,
and it cannot infer the dimensions of a raw file. `prepare_image` therefore
converts anything Pillow can open into a format this build accepts, and
records what it did.

Binary discovery order: the `FPE_NBIS_BIN` environment variable, then
`tools/bin/`, then the default build location.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Minutia",
    "find_binary",
    "run_mindtct",
    "read_xyt",
    "match_xyt",
    "match_many",
]

ROOT = Path(__file__).resolve().parents[3]

_SEARCH_DIRS = [
    Path(os.environ["FPE_NBIS_BIN"]) if os.environ.get("FPE_NBIS_BIN") else None,
    ROOT / "tools" / "bin",
    Path("E:/toolchains/nbis/bin"),
]


@dataclass(frozen=True)
class Minutia:
    """One minutia in NBIS `.xyt` terms: pixel position, angle, quality."""

    x: int
    y: int
    theta: int  # degrees, 0-359
    quality: int  # 0-100


@lru_cache(maxsize=None)
def find_binary(name: str) -> Path:
    """Locate an NBIS executable, or explain clearly where we looked."""
    exe = f"{name}.exe" if os.name == "nt" else name
    for d in _SEARCH_DIRS:
        if d is None:
            continue
        candidate = Path(d) / exe
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    searched = ", ".join(str(d) for d in _SEARCH_DIRS if d is not None)
    raise FileNotFoundError(
        f"NBIS binary '{name}' not found. Searched: {searched}, and PATH. "
        f"Set FPE_NBIS_BIN to the directory holding it."
    )


def run_mindtct(
    image: str | Path,
    out_root: str | Path,
    *,
    enhance: bool = False,
    timeout: int = 120,
) -> Path:
    """Extract minutiae; return the path to the `.xyt` template.

    `enhance` passes `-b`, mindtct's own contrast boost. Leave it off for
    baseline conditions -- it is an enhancement step, and enabling it silently
    would confound exactly the comparison this thesis makes.
    """
    image, out_root = Path(image), Path(out_root)
    if not image.is_file():
        raise FileNotFoundError(image)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(find_binary("mindtct"))]
    if enhance:
        cmd.append("-b")
    cmd += [str(image), str(out_root)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    xyt = out_root.with_suffix(".xyt")
    if proc.returncode != 0 or not xyt.is_file():
        raise RuntimeError(
            f"mindtct failed on {image} (exit {proc.returncode})\n"
            f"stdout: {proc.stdout.strip()}\nstderr: {proc.stderr.strip()}"
        )
    return xyt


def read_xyt(path: str | Path) -> list[Minutia]:
    """Parse an NBIS `.xyt` template."""
    out: list[Minutia] = []
    for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y, t = (int(parts[0]), int(parts[1]), int(parts[2]))
        q = int(parts[3]) if len(parts) > 3 else 0
        out.append(Minutia(x=x, y=y, theta=t, quality=q))
    return out


def match_xyt(probe: str | Path, gallery: str | Path, *, timeout: int = 120) -> int:
    """BOZORTH3 similarity for one pair. Higher is more similar."""
    cmd = [str(find_binary("bozorth3")), str(probe), str(gallery)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bozorth3 failed (exit {proc.returncode})\nstderr: {proc.stderr.strip()}"
        )
    text = proc.stdout.strip().splitlines()
    if not text:
        raise RuntimeError(f"bozorth3 produced no score for {probe} vs {gallery}")
    return int(text[-1].split()[-1])


def match_many(
    pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    mates_file: str | Path,
    timeout: int = 3600,
) -> np.ndarray:
    """Score many pairs in one bozorth3 invocation.

    Spawning a process per comparison dominates runtime once pair counts reach
    the tens of thousands, which ours do. bozorth3's `-M` mode reads a file of
    alternating probe/gallery paths and emits one score per line.
    """
    mates = Path(mates_file)
    mates.parent.mkdir(parents=True, exist_ok=True)
    with mates.open("w", encoding="utf-8", newline="\n") as fh:
        for probe, gallery in pairs:
            fh.write(f"{Path(probe).as_posix()}\n{Path(gallery).as_posix()}\n")

    # maxfiles defaults to 10,000; our impostor lists run to hundreds of
    # thousands of lines, and bozorth3 refuses rather than truncating.
    cmd = [
        str(find_binary("bozorth3")),
        "-A", "outfmt=s",
        "-A", f"maxfiles={max(2 * len(pairs) + 16, 10000)}",
        "-M", str(mates),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bozorth3 -M failed (exit {proc.returncode})\nstderr: {proc.stderr.strip()}"
        )
    scores = [int(line.split()[-1]) for line in proc.stdout.split("\n") if line.strip()]
    if len(scores) != len(pairs):
        raise RuntimeError(
            f"bozorth3 returned {len(scores)} scores for {len(pairs)} pairs"
        )
    return np.asarray(scores, dtype=np.int32)
