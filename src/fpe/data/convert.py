"""Convert dataset images into something NBIS can read.

Our images arrive as TIFF (FVC, Neurotechnology), BMP (FVS), PNG (Anguli) and
headerless raw (MINEX). NBIS reads ANSI/NIST, WSQ, lossless JPEG, JPEG and
IHead -- plus PNG only when built with PNG support.

Several sets also store greyscale content in a colour mode: Neurotechnology
uses palette (`P`), FVS and L3-SF use RGB, and SOCOFing's real images are
RGBA. Feeding those through unconverted would give NBIS three identical
channels or an indexed palette, so conversion to 8-bit greyscale is
unconditional.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "PreparedImage",
    "to_greyscale_png",
    "to_nbis_input",
    "prepare",
    "load_greyscale",
    "cache_path",
]

DEFAULT_PPI = 500


def load_greyscale(path: str | Path) -> np.ndarray:
    """Read any supported image as an 8-bit greyscale array."""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        return np.asarray(im, dtype=np.uint8)


def cache_path(source: str | Path, cache_dir: str | Path, suffix: str = ".png") -> Path:
    """A stable cache filename for one source image.

    Named by a hash of the source path so that images with colliding basenames
    across datasets -- FVC's `101_1.tif` exists in all twelve databases --
    cannot overwrite each other.
    """
    src = Path(source)
    digest = hashlib.sha256(src.as_posix().encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{src.stem}_{digest}{suffix}"


def to_greyscale_png(
    source: str | Path, cache_dir: str | Path, *, overwrite: bool = False
) -> Path:
    """Write `source` as an 8-bit greyscale PNG under `cache_dir`.

    PNG is lossless, so this introduces no compression artefact into the
    baseline -- which matters, because the baseline is what every enhancement
    result is measured against.
    """
    out = cache_path(source, cache_dir, ".png")
    if out.is_file() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = load_greyscale(source)
    Image.fromarray(arr, mode="L").save(out, format="PNG", optimize=False)
    return out


@dataclass(frozen=True)
class PreparedImage:
    """One source image, in the two forms the pipeline needs."""

    source: Path
    png: Path  # for NFIQ 2 and, later, the network
    jpl: Path  # for mindtct
    width: int
    height: int
    ppi: int


def to_nbis_input(
    source: str | Path,
    cache_dir: str | Path,
    *,
    ppi: int = DEFAULT_PPI,
    overwrite: bool = False,
) -> Path:
    """Write `source` as lossless JPEG, the format mindtct accepts.

    mindtct reads ANSI/NIST, WSQ, lossless JPEG, JPEG and IHead -- not TIFF,
    BMP or PNG, and it cannot infer the dimensions of a headerless raw file.
    Lossless JPEG is the only one of those that is both writable from here and
    free of compression artefacts; a WSQ intermediate would put lossy
    compression underneath every number in the thesis.

    Verified byte-identical on round-trip through cjpegl/djpegl.
    """
    from fpe.metrics.nbis import find_binary  # local import: avoids a cycle

    out = cache_path(source, cache_dir, ".jpl")
    if out.is_file() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = load_greyscale(source)
    height, width = arr.shape
    raw = out.with_suffix(".raw")
    arr.tofile(raw)
    try:
        # cjpegl names its output <input stem>.<outext>, beside the input.
        proc = subprocess.run(
            [
                str(find_binary("cjpegl")), "jpl", str(raw),
                "-raw_in", f"{width},{height},8,{ppi}",
            ],
            capture_output=True, text=True, timeout=120,
        )
        produced = raw.with_suffix(".jpl")
        if proc.returncode != 0 or not produced.is_file():
            raise RuntimeError(
                f"cjpegl failed on {source} (exit {proc.returncode})\n"
                f"stdout: {proc.stdout.strip()}\nstderr: {proc.stderr.strip()}"
            )
        if produced != out:
            produced.replace(out)
    finally:
        raw.unlink(missing_ok=True)
        out.with_suffix(".ncm").unlink(missing_ok=True)  # cjpegl comment file
    return out


def prepare(
    source: str | Path, cache_dir: str | Path, *, ppi: int = DEFAULT_PPI
) -> PreparedImage:
    """Produce both cached forms of one source image from a single decode."""
    source = Path(source)
    arr = load_greyscale(source)
    height, width = arr.shape
    return PreparedImage(
        source=source,
        png=to_greyscale_png(source, cache_dir),
        jpl=to_nbis_input(source, cache_dir, ppi=ppi),
        width=width,
        height=height,
        ppi=ppi,
    )
