"""The Anguli synthetic corpus: indexing, splits, and the supervision build (plan U2).

Layout on disk, verified against `data/processed/anguli_dev_5584`:

    Fingerprints/fp_N/M.png     clean binary master  -- the ridge-map target
    Impression_K/fp_N/M.png     greyscale impression -- what gets degraded and fed in
    Meta Info/fp_N/M.txt        pattern type and singular points only

Three facts about the layout that are easy to get wrong:

- **Finger numbers are globally unique**, not per-bucket: `fp_1` holds 1-1000 and `fp_6`
  holds 5001-5758. The bucket is a directory-size convention, not a namespace.
- **The numbering is sparse.** `scripts/reconcile_anguli.py` pruned torn records from the
  interrupted 20,000-finger run, so ranges have holes. Always enumerate files; never
  assume a contiguous range.
- **Empty bucket directories survive** from that run (`fp_7` through `fp_20`). They must be
  skipped rather than treated as missing data.

## Why the teachers run on the clean impression

The network needs five targets per sample, and three of them -- segmentation, orientation
and ridge frequency -- have no source: Anguli exports pattern type and singular points and
nothing else. So Cappelli's pretrained estimators are run over the **clean** impression and
their output is cached as the target. The student then has to reproduce, from a degraded
image, what the teacher could only manage on a clean one.

This is distillation from a validated teacher, and it has an honest cost that belongs in
the thesis: for those two heads the "ground truth" is another model's output rather than
human annotation, so they are evaluated as agreement-with-teacher, not absolute accuracy.
The ridge-map and confidence targets are unaffected -- the master is exact and the evidence
map is analytic.

## Splits

Assigned by hashing the finger id, so the split is decided by arithmetic rather than by a
file that has to be kept in sync between scripts. Two consequences that matter: the
assignment is identical in every process and every run, and a finger's master and all three
of its impressions land in the same split by construction, which is what stops a finger
being half-seen during training and half-used for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "AnguliSample",
    "SPLITS",
    "assign_split",
    "index_corpus",
    "supervision_path",
    "build_sample",
    "write_manifest",
]

# Keras resolves its backend at import time and defaults to TensorFlow, which is not
# installed here deliberately. pyfing is imported lazily inside build_sample, so setting
# this at module import is early enough -- and it makes the library work from any
# entrypoint rather than only from scripts that remember to set it.
os.environ.setdefault("KERAS_BACKEND", "torch")

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.80, 0.10, 0.10)
SPLIT_SALT = "fpe-anguli-v1"
"""Changing the salt reshuffles every split. Treat it as frozen once training starts."""

IMPRESSION_DIR = re.compile(r"^Impression_(\d+)$")
MASTER_DIR = "Fingerprints"


@dataclass(frozen=True)
class AnguliSample:
    """One impression, with its master and its split already resolved."""

    finger_id: str      # globally unique finger number, as a string
    bucket: str         # directory bucket, e.g. "fp_6"
    impression: int     # 1-based impression index
    master: Path        # clean binary ridge target
    image: Path         # greyscale impression -- the input before degradation
    split: str

    @property
    def key(self) -> str:
        return f"{self.bucket}/{self.finger_id}_{self.impression}"

    @property
    def relpath(self) -> str:
        """Path relative to the corpus root, in manifest form."""
        return f"Impression_{self.impression}/{self.bucket}/{self.finger_id}.png"


def assign_split(
    finger_id: str, ratios: Sequence[float] = DEFAULT_RATIOS, salt: str = SPLIT_SALT
) -> str:
    """Deterministically map a finger to a split.

    Uses blake2b rather than the built-in `hash()`, whose seed is randomised per process --
    the same finger would land in different splits in different runs, which is exactly the
    leakage this function exists to prevent.
    """
    digest = hashlib.blake2b(f"{salt}:{finger_id}".encode("utf-8"), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    for name, ratio in zip(SPLITS, ratios):
        cumulative += ratio
        if position < cumulative:
            return name
    return SPLITS[-1]


def index_corpus(
    root: str | Path,
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    impressions: int | None = None,
) -> list[AnguliSample]:
    """Every impression that has a master and a readable file, split assigned.

    A finger is included only when its master and the requested impressions all exist --
    a torn record would otherwise pair an input against a missing target and train the
    network on a lie.
    """
    root = Path(root)
    master_root = root / MASTER_DIR
    if not master_root.is_dir():
        raise FileNotFoundError(f"no {MASTER_DIR}/ under {root}")

    impression_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and IMPRESSION_DIR.match(d.name)),
        key=lambda d: int(IMPRESSION_DIR.match(d.name).group(1)),
    )
    if impressions is not None:
        impression_dirs = impression_dirs[:impressions]
    if not impression_dirs:
        raise FileNotFoundError(f"no Impression_* directories under {root}")

    out: list[AnguliSample] = []
    for bucket_dir in sorted(master_root.iterdir()):
        if not bucket_dir.is_dir():
            continue
        bucket = bucket_dir.name
        for master in sorted(bucket_dir.glob("*.png")):
            finger_id = master.stem
            paths = [d / bucket / master.name for d in impression_dirs]
            if not all(p.is_file() for p in paths):
                continue  # torn record; reconcile_anguli.py should have pruned it
            split = assign_split(finger_id, ratios)
            for index, path in enumerate(paths, start=1):
                out.append(AnguliSample(
                    finger_id=finger_id, bucket=bucket, impression=index,
                    master=master, image=path, split=split,
                ))
    return out


def supervision_path(sample: AnguliSample, out_root: str | Path) -> Path:
    """Where one sample's cached targets live. Sharded by bucket to keep directories small."""
    return Path(out_root) / sample.bucket / f"{sample.finger_id}_{sample.impression}.npz"


def build_sample(
    sample: AnguliSample, out_root: str | Path, *, dpi: int = 500, overwrite: bool = False
) -> tuple[Path, bool]:
    """Run the teachers on one clean impression and cache the targets.

    Returns (path, built) -- `built` is False when the cache already existed.

    The write is **atomic**: targets go to a temporary file which is then renamed into
    place. A rename cannot half-finish, so a file that exists is a file that is complete,
    and an interrupted run can be resumed by skipping what is already there. Without this,
    a build killed mid-write would leave a truncated array that loads without error and
    silently corrupts training.
    """
    from fpe.data.convert import load_greyscale
    from fpe.models.pyfing_runtime import load_pyfing, pyfing_cpu

    pf = load_pyfing()

    path = supervision_path(sample, out_root)
    if path.is_file() and not overwrite:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)

    image = load_greyscale(sample.image)
    with pyfing_cpu():
        mask = pf.fingerprint_segmentation(image, dpi=dpi, method="SUFS")
        orientation = pf.orientation_field_estimation(image, mask, dpi=dpi, method="SNFOE")
        period = pf.frequency_estimation(image, orientation, mask, dpi=dpi, method="SNFFE")

    meta = json.dumps({
        "finger_id": sample.finger_id, "bucket": sample.bucket,
        "impression": sample.impression, "split": sample.split, "dpi": dpi,
        "teachers": {"segmentation": "SUFS", "orientation": "SNFOE", "frequency": "SNFFE"},
        "master": sample.master.name,
    })

    tmp = path.with_suffix(".npz.tmp")
    # float16 is ample: orientation lives in [-pi/2, pi/2] and period in tens of pixels,
    # both far inside float16's ~3 significant digits. Halving the cache matters more for
    # read speed during training than the precision does for the targets.
    #
    # Write through an open handle, not a path: np.savez_compressed *appends* ".npz" to
    # any path that lacks it, so passing "x.npz.tmp" silently produces "x.npz.tmp.npz"
    # and the rename below finds nothing.
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            mask=(np.asarray(mask) > 0).astype(np.uint8),
            orientation=np.asarray(orientation, dtype=np.float16),
            period=np.asarray(period, dtype=np.float16),
            meta=np.array(meta),
        )
    os.replace(tmp, path)
    return path, True


def clear_partial_writes(out_root: str | Path) -> int:
    """Remove leftover temporary files from an interrupted run. Returns the count."""
    root = Path(out_root)
    if not root.is_dir():
        return 0
    removed = 0
    for tmp in root.rglob("*.npz.tmp"):
        tmp.unlink(missing_ok=True)
        removed += 1
    return removed


def pending(
    samples: Sequence[AnguliSample], out_root: str | Path
) -> Iterator[AnguliSample]:
    """Samples whose cache does not yet exist."""
    for sample in samples:
        if not supervision_path(sample, out_root).is_file():
            yield sample


MANIFEST_COLUMNS = [
    "dataset", "relpath", "sha256", "bytes", "width", "height",
    "declared_dpi", "mode", "subject", "finger", "impression",
    "quality", "gender", "severity",
]


def write_manifest(
    samples: Sequence[AnguliSample],
    manifest_path: str | Path,
    *,
    dataset: str = "anguli_dev_5584",
    split: str = "test",
    dpi: int = 500,
) -> int:
    """Write an evaluation manifest for one split, in the project's manifest format.

    The week-1 harness reads manifests, and Anguli has none -- without this there is no way
    to measure a trained model or to sweep the precision-coverage curve.

    **Only the named split is written.** Pointing the evaluator at training fingers would
    invalidate every number it produced, and the cheapest way to make that impossible is to
    leave them out of the file the evaluator reads.
    """
    import csv

    from PIL import Image

    chosen = [s for s in samples if s.split == split]
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for sample in sorted(chosen, key=lambda s: (s.bucket, int(s.finger_id), s.impression)):
            digest = hashlib.sha256()
            with sample.image.open("rb") as image_file:
                for chunk in iter(lambda: image_file.read(1 << 20), b""):
                    digest.update(chunk)
            with Image.open(sample.image) as im:
                width, height, mode = im.width, im.height, im.mode
            writer.writerow({
                "dataset": dataset,
                "relpath": sample.relpath,
                "sha256": digest.hexdigest(),
                "bytes": sample.image.stat().st_size,
                "width": width,
                "height": height,
                "declared_dpi": dpi,
                "mode": mode,
                "subject": sample.finger_id,
                "finger": sample.finger_id,
                "impression": sample.impression,
                "quality": "",
                "gender": "",
                "severity": "",
            })
    return len(chosen)
