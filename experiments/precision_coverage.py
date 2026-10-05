"""Minutiae precision and the precision-coverage curve (plan U9).

On the held-out synthetic test split, deterministically wear-degraded at severity 0.5:
pseudo ground truth is mindtct on each finger's clean master, and every condition is
scored by Cappelli's correspondence protocol (14 px, pi/9; type-agnostic and type-exact)
plus verification EER over the test pairs. The WAFEN curve sweeps the block-abstention
threshold; every baseline is a single point at coverage 1.0.

All figures from this script are **synthetic** pseudo-GT numbers and are labelled so;
they are not comparable to published SD27 results.

    python experiments/precision_coverage.py            # full run, resumable (~2.5 h)
    python experiments/precision_coverage.py --limit-fingers 8 --dry-run   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fpe.data.anguli import index_corpus, supervision_path  # noqa: E402
from fpe.data.convert import load_greyscale, to_nbis_input  # noqa: E402
from fpe.degradation.wear import WearConfig, WearDegradation  # noqa: E402
from fpe.eval.abstain import filter_xyt_file  # noqa: E402
from fpe.eval.protocol import Impression, build_pairs  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402
from fpe.metrics.matching import bootstrap_eer  # noqa: E402
from fpe.metrics.minutiae import match_minutiae, parse_min_file  # noqa: E402
from fpe.metrics.nbis import match_many, run_mindtct  # noqa: E402

CORPUS = ROOT / "data" / "processed" / "anguli_dev_5584"
SUPERVISION = ROOT / "data" / "processed" / "supervision"
DEGRADED = ROOT / "data" / "processed" / "degraded_test"
WORK = ROOT / "data" / "work" / "u9"
SEVERITY = 0.5
SWEEP = (0.10, 0.15, 0.20, 0.30, 0.40)
"""Block-abstention (16 px) thresholds. The sweep *is* the protocol (R15): the curve is
the deliverable, not a search for one good point, and every point lands in the registry."""
MAX_IMPOSTOR = 20_000


def seed_of(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big")


def degraded_path(sample) -> Path:
    return DEGRADED / sample.bucket / f"{sample.finger_id}_{sample.impression}.png"


def build_degraded(samples, progress) -> None:
    """Deterministic full-size wear degradation, written once and reused by every run."""
    from PIL import Image

    wear = WearDegradation(WearConfig())
    todo = [s for s in samples if not degraded_path(s).is_file()]
    progress(f"degrading {len(todo)}/{len(samples)} test impressions (severity {SEVERITY})")
    for n, s in enumerate(todo, 1):
        with np.load(supervision_path(s, SUPERVISION), allow_pickle=False) as d:
            mask = d["mask"].astype(np.float32)
            orientation = d["orientation"].astype(np.float32)
        image = load_greyscale(s.image).astype(np.float32) / 255.0
        out = wear(image, seed=seed_of(s.key), severity=SEVERITY,
                   mask=mask, orientation=orientation).image
        path = degraded_path(s)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".png.tmp")
        Image.fromarray(np.round(np.clip(out, 0, 1) * 255).astype(np.uint8)).save(
            tmp, "PNG")
        tmp.replace(path)
        if n % 200 == 0:
            progress(f"  {n}/{len(todo)}")


def template_for(image_path: Path, tmpl_dir: Path, stem: str) -> Path | None:
    """mindtct via the lossless-JPEG conversion it requires; cached by .min existence."""
    xyt = tmpl_dir / f"{stem}.xyt"
    if xyt.with_suffix(".min").is_file():
        return xyt
    try:
        jpl = to_nbis_input(image_path, tmpl_dir / "jpl")
        return run_mindtct(jpl, tmpl_dir / stem)
    except (OSError, RuntimeError, ValueError):
        return None


def build_gt(samples, progress) -> dict[str, Path]:
    """One pseudo-GT template per finger: mindtct on the clean master."""
    gt_dir = WORK / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    masters = {s.finger_id: s.master for s in samples}
    out = {}
    for n, (fid, master) in enumerate(sorted(masters.items()), 1):
        xyt = template_for(master, gt_dir, f"f{fid}")
        if xyt is not None:
            out[fid] = xyt.with_suffix(".min")
        if n % 200 == 0:
            progress(f"  gt {n}/{len(masters)}")
    progress(f"ground truth: {len(out)}/{len(masters)} fingers")
    return out


def run_condition(name, samples, gt, enhancer, abstain, progress, dry_run):
    cond_dir = WORK / name.replace("@", "_t")
    tmpl = cond_dir / "tmpl"
    tmpl.mkdir(parents=True, exist_ok=True)

    per = {"precision": [], "recall": [], "f1": [],
           "tx_precision": [], "tx_recall": [], "tx_f1": []}
    templates: dict[str, Path] = {}
    dropped = 0
    for n, s in enumerate(samples, 1):
        src = degraded_path(s)
        img = src if enhancer is None else enhancer(src)
        stem = f"{s.bucket}_{s.finger_id}_{s.impression}"
        xyt = template_for(img, tmpl, stem)
        if xyt is None or s.finger_id not in gt:
            continue
        min_path = xyt.with_suffix(".min")
        detected = parse_min_file(min_path)
        if abstain is not None and enhancer is not None:
            keep = load_greyscale(enhancer._keep_path(Path(img))) > 127
            kept = [m for m in detected
                    if 0 <= m.y < keep.shape[0] and 0 <= m.x < keep.shape[1]
                    and keep[m.y, m.x]]  # .min is top-origin: direct indexing
            dropped += len(detected) - len(kept)
            detected = kept
            filter_xyt_file(xyt, keep)
        truth = parse_min_file(gt[s.finger_id])
        c = match_minutiae(detected, truth)
        tx = match_minutiae(detected, truth, type_exact=True)
        for k, v in [("precision", c.precision), ("recall", c.recall), ("f1", c.f1),
                     ("tx_precision", tx.precision), ("tx_recall", tx.recall),
                     ("tx_f1", tx.f1)]:
            per[k].append(v)
        templates[s.key] = xyt
        if n % 200 == 0:
            progress(f"  {name}: {n}/{len(samples)}")

    impressions = [Impression("anguli", s.key, s.finger_id, str(s.impression), "anguli")
                   for s in samples if s.key in templates]
    pairs = build_pairs(impressions, max_impostor=MAX_IMPOSTOR, seed=0)
    genuine = match_many([(templates[a.relpath], templates[b.relpath])
                          for a, b in pairs.genuine], mates_file=cond_dir / "gen.lis")
    impostor = match_many([(templates[a.relpath], templates[b.relpath])
                           for a, b in pairs.impostor], mates_file=cond_dir / "imp.lis")
    rates = bootstrap_eer(genuine, impostor, n_resamples=1000, seed=0)

    coverage = 1.0
    if abstain is not None and enhancer is not None:
        am = enhancer.abstention_metrics
        coverage = am.get("coverage_mean", 1.0)
    metrics = {
        **{k: float(np.mean(v)) for k, v in per.items()},
        "coverage": coverage, "minutiae_dropped_total": float(dropped),
        "n_images": float(len(templates)), "n_genuine": float(len(pairs.genuine)),
        "n_impostor": float(len(pairs.impostor)), **rates.as_dict(),
    }
    progress(f"  {name}: precision {metrics['precision']:.3f}  recall "
             f"{metrics['recall']:.3f}  EER {metrics['eer']:.4f}  "
             f"coverage {coverage:.1%}")
    if not dry_run:
        log_run(experiment="precision-coverage",
                dataset=f"anguli_dev_5584 test (wear severity {SEVERITY}, synthetic "
                        f"pseudo-GT from clean masters)",
                method=name,
                metrics=metrics,
                config={"tau_d_px": 14, "tau_theta": "pi/9", "severity": SEVERITY,
                        "abstain": abstain, "block": 16 if abstain else None,
                        "max_impostor": MAX_IMPOSTOR, "sweep": list(SWEEP)},
                notes="U9: every figure is synthetic pseudo-GT; not comparable to "
                      "published SD27 numbers.")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "data" / "work" / "wafen_da" / "best.pt")
    ap.add_argument("--limit-fingers", type=int, default=None)
    ap.add_argument("--skip-snfen", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    samples = [s for s in index_corpus(CORPUS) if s.split == "test"
               and supervision_path(s, SUPERVISION).is_file()]
    if args.limit_fingers:
        keep = sorted({s.finger_id for s in samples})[:args.limit_fingers]
        samples = [s for s in samples if s.finger_id in set(keep)]
    print(f"{len(samples)} test impressions, {len({s.finger_id for s in samples})} fingers")

    build_degraded(samples, print)
    gt = build_gt(samples, print)

    from fpe.models.wafen_enhancer import WafenEnhancer

    results = {}
    results["none"] = run_condition("none", samples, gt, None, None, print,
                                    args.dry_run)
    if not args.skip_snfen:
        from fpe.models.pyfing_baseline import PyfingEnhancer

        results["SNFEN"] = run_condition(
            "SNFEN", samples, gt,
            PyfingEnhancer("SNFEN", cache_dir=WORK / "SNFEN"), None, print,
            args.dry_run)
    results["WAFEN@1.0"] = run_condition(
        "WAFEN@1.0", samples, gt,
        WafenEnhancer(args.checkpoint, cache_dir=WORK / "WAFEN_t0"), None, print,
        args.dry_run)
    for t in SWEEP:
        results[f"WAFEN@{t:.2f}"] = run_condition(
            f"WAFEN@{t:.2f}", samples, gt,
            WafenEnhancer(args.checkpoint, cache_dir=WORK / f"WAFEN_t{t:.2f}",
                          abstain_threshold=t, block=16),
            t, print, args.dry_run)

    import json

    (WORK / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"summary -> {WORK / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
