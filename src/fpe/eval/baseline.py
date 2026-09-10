"""End-to-end verification benchmark: images in, EER out, one registry row.

This is the harness ADR 0002 says must exist before any model is trained. It
runs the full operational chain -- convert, extract minutiae, match, score --
so that every later condition (an enhancement method, a degradation setting)
is measured by swapping one stage and re-running.

Failure to acquire is measured here rather than treated as an error: an image
that yields no usable template is a real deployment outcome, and its rate is
one of the metrics the thesis reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from fpe.data.convert import to_greyscale_png, to_nbis_input
from fpe.eval.protocol import Impression, ImpostorMode, build_pairs, parse_manifest
from fpe.metrics.matching import bootstrap_eer
from fpe.metrics.nbis import match_many, read_xyt, run_mindtct
from fpe.metrics.nfiq2 import score_images, score_summary

__all__ = ["BaselineResult", "run_baseline"]

MIN_MINUTIAE = 1  # below this a template cannot be matched at all


@dataclass
class BaselineResult:
    metrics: dict[str, object]
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    failed: list[str]


def _template_for(
    im: Impression,
    *,
    data_root: Path,
    cache_dir: Path,
    work_dir: Path,
    preprocess: Callable[[Path], Path] | None,
) -> tuple[Path | None, int, Path | None]:
    """Return (xyt path, minutia count, prepared image).

    xyt is None when the image failed to enrol; the prepared image is still
    returned when it exists, so quality can be scored even for failures.
    """
    source = data_root / im.relpath
    png: Path | None = None
    try:
        png = to_greyscale_png(source, cache_dir)
        if preprocess is not None:
            png = preprocess(png)
        # mindtct cannot read PNG; lossless JPEG is byte-identical to it.
        jpl = to_nbis_input(png, cache_dir)
        xyt = run_mindtct(jpl, work_dir / jpl.stem)
    except (OSError, RuntimeError, ValueError):
        return None, 0, png
    count = len(read_xyt(xyt))
    if count < MIN_MINUTIAE:
        return None, count, png
    return xyt, count, png


def run_baseline(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    work_dir: str | Path,
    subset: str | None = None,
    preprocess: Callable[[Path], Path] | None = None,
    impostor: ImpostorMode = "all",
    n_resamples: int = 1000,
    seed: int = 0,
    progress: Callable[[str], None] = lambda _msg: None,
) -> BaselineResult:
    """Run the whole chain on one dataset and return its metrics.

    `preprocess` is where an enhancement method plugs in: it receives a
    prepared greyscale PNG and returns the path to an enhanced one. `None` is
    the no-enhancement control.
    """
    data_root = Path(data_root)
    work_dir = Path(work_dir)
    cache_dir = work_dir / "prepared"
    tmpl_dir = work_dir / "templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    impressions = parse_manifest(manifest_path, subset=subset)
    if not impressions:
        raise ValueError(f"no parseable impressions in {manifest_path} (subset={subset})")

    templates: dict[str, Path] = {}
    prepared_images: list[Path] = []
    minutia_counts: list[int] = []
    failed: list[str] = []
    for n, im in enumerate(impressions, start=1):
        xyt, count, prepared = _template_for(
            im,
            data_root=data_root,
            cache_dir=cache_dir,
            work_dir=tmpl_dir,
            preprocess=preprocess,
        )
        if prepared is not None:
            prepared_images.append(prepared)
        if xyt is None:
            failed.append(im.relpath)
        else:
            templates[im.relpath] = xyt
            minutia_counts.append(count)
        if n % 100 == 0 or n == len(impressions):
            progress(f"  templates {n}/{len(impressions)} ({len(failed)} failed)")

    usable = [im for im in impressions if im.relpath in templates]
    pairs = build_pairs(usable, impostor=impostor)
    if not pairs.genuine or not pairs.impostor:
        raise ValueError(
            f"insufficient pairs after enrolment: {len(pairs.genuine)} genuine, "
            f"{len(pairs.impostor)} impostor"
        )
    progress(f"  matching {len(pairs.genuine)} genuine, {len(pairs.impostor)} impostor")

    def score(pair_list: Sequence[tuple[Impression, Impression]], tag: str) -> np.ndarray:
        return match_many(
            [(templates[a.relpath], templates[b.relpath]) for a, b in pair_list],
            mates_file=work_dir / f"mates_{tag}.lis",
        )

    genuine = score(pairs.genuine, "genuine")
    impostor_scores = score(pairs.impostor, "impostor")

    rates = bootstrap_eer(genuine, impostor_scores, n_resamples=n_resamples, seed=seed)

    progress(f"  scoring quality on {len(prepared_images)} images")
    try:
        quality = score_summary(score_images(prepared_images))
    except (FileNotFoundError, RuntimeError) as exc:
        # Quality is a reported metric, not a gate -- a missing NFIQ 2 must not
        # lose the EER we just spent minutes computing.
        progress(f"  NFIQ 2 unavailable: {exc}")
        quality = {}

    counts = np.asarray(minutia_counts, dtype=np.float64)
    metrics: dict[str, object] = {
        **rates.as_dict(),
        **quality,
        "n_images": len(impressions),
        "n_enrolled": len(usable),
        "fta_rate": len(failed) / len(impressions),
        "minutiae_mean": float(counts.mean()) if counts.size else 0.0,
        "minutiae_median": float(np.median(counts)) if counts.size else 0.0,
        "genuine_score_mean": float(genuine.mean()),
        "impostor_score_mean": float(impostor_scores.mean()),
    }
    return BaselineResult(
        metrics=metrics,
        genuine_scores=genuine,
        impostor_scores=impostor_scores,
        failed=failed,
    )
