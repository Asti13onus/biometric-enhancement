"""Real-print supervision for teacher distillation.

WAFEN trained on synthetic Anguli alone gives estimates on real prints clearly worse than
the pyfing networks that supervised it. On FVC2004 no WAFEN variant beats the unenhanced
control (session 10). The model has never seen a real sensor. This module labels real
prints from the **non-test** datasets with the same teachers -- SUFS, SNFOE, SNFFE -- plus
SNFEN's enhanced output as the ridge target, so the model can learn real texture while the
targets stay those of the 2026 state of the art.

Cache files have the same layout as the Anguli supervision (`fpe.data.anguli`) plus a
`ridge` array, so `WafenDataset` reads either. Writes are atomic and the build is
resumable, as there.

**Leakage.** FVC2004 is the test set and Neurotechnology CrossMatch an evaluation set. Both
are refused outright, not merely left off a default list: a test image used as a training
input is precisely Cappelli's criticism of DNUNets and FingerGAN (PROJECT_RULES.md rule 4).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from fpe.data.anguli import AnguliSample, assign_split, supervision_path
from fpe.eval.protocol import parse_manifest

__all__ = ["TRAIN_DATASETS", "HELD_OUT", "REAL_RATIOS", "index_real", "build_real_sample"]

TRAIN_DATASETS = ("fvc2000", "fvc2002", "neurotech_uareu")
HELD_OUT = frozenset({"fvc2004", "neurotech_crossmatch"})
REAL_RATIOS = (0.9, 0.1, 0.0)
"""Train / val / test by finger. No real test split: testing happens on held-out datasets."""


def index_real(
    manifest_dir: str | Path, data_root: str | Path,
    datasets: tuple[str, ...] = TRAIN_DATASETS,
) -> list[tuple[AnguliSample, int]]:
    """(sample, declared dpi) for every parseable print in the given datasets.

    `master` is None: real prints have no clean ridge master, their ridge target is the
    teacher's enhancement, stored in the cache.
    """
    leaked = HELD_OUT.intersection(datasets)
    if leaked:
        raise ValueError(f"refusing held-out dataset(s) {sorted(leaked)}: they are "
                         f"evaluation data and must never become training inputs")
    out = []
    for name in datasets:
        manifest = Path(manifest_dir) / f"{name}.csv"
        with manifest.open(newline="", encoding="utf-8") as fh:
            dpi = {row["relpath"]: int(row["declared_dpi"] or 500) for row in csv.DictReader(fh)}
        for imp in parse_manifest(manifest):
            sample = AnguliSample(
                finger_id=imp.finger_id, bucket=name, impression=int(imp.impression),
                master=None, image=Path(data_root) / name / imp.relpath,
                split=assign_split(imp.finger_id, REAL_RATIOS),
            )
            out.append((sample, dpi[imp.relpath]))
    return out


def build_real_sample(sample: AnguliSample, out_root: str | Path, *, dpi: int = 500,
                      overwrite: bool = False) -> tuple[Path, bool]:
    """Run the four teachers on one real print and cache the targets atomically."""
    from fpe.data.convert import load_greyscale
    from fpe.models.pyfing_runtime import load_pyfing, pyfing_cpu

    path = supervision_path(sample, out_root)
    if path.is_file() and not overwrite:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)

    pf = load_pyfing()
    image = load_greyscale(sample.image)
    with pyfing_cpu():
        mask = pf.fingerprint_segmentation(image, dpi=dpi, method="SUFS")
        orientation = pf.orientation_field_estimation(image, mask, dpi=dpi, method="SNFOE")
        period = pf.frequency_estimation(image, orientation, mask, dpi=dpi, method="SNFFE")
        enhanced = pf.fingerprint_enhancement(image, orientation, period, mask, dpi=dpi,
                                              method="SNFEN")
    fg = np.asarray(mask) > 0
    # SNFEN renders ridges near-white, so its intensity already means "ridge here" -- the
    # same sense as the inverted Anguli master. Outside the mask there is no ridge.
    ridge = np.where(fg, np.asarray(enhanced, dtype=np.uint8), 0).astype(np.uint8)

    meta = json.dumps({
        "finger_id": sample.finger_id, "dataset": sample.bucket,
        "impression": sample.impression, "split": sample.split, "dpi": dpi,
        "teachers": {"segmentation": "SUFS", "orientation": "SNFOE",
                     "frequency": "SNFFE", "ridge": "SNFEN"},
        "image": sample.image.name,
    })
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:  # handle, not path: savez would append ".npz"
        np.savez_compressed(
            fh, mask=fg.astype(np.uint8),
            orientation=np.asarray(orientation, dtype=np.float16),
            period=np.asarray(period, dtype=np.float16),
            ridge=ridge, meta=np.array(meta),
        )
    os.replace(tmp, path)
    return path, True
