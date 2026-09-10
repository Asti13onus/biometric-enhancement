"""Baseline verification benchmark: NBIS mindtct + bozorth3, no enhancement.

This is the Phase 0 gate. It runs the whole operational chain on a real
dataset and writes one row to the results registry.

    python experiments/baseline_nbis.py --dataset fvc2004 --subset db1_b

Thin entrypoint by convention -- the logic lives in `src/fpe/eval/baseline.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.eval.baseline import run_baseline  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="manifest stem, e.g. fvc2004")
    ap.add_argument("--subset", default=None, help="path prefix filter, e.g. db1_b")
    ap.add_argument("--data-root", default=None, help="defaults to data/raw/<dataset>")
    ap.add_argument("--work-dir", default=None, help="defaults to a scratch dir on E:")
    ap.add_argument(
        "--impostor", choices=["all", "first"], default="all",
        help="'all' uses every cross-finger impression pair; 'first' is the "
             "official FVC rule and yields far fewer pairs",
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--method", default="none", choices=["none", "SNFEN", "GBFEN"],
        help="'none' is the no-enhancement control; the others are pyfing's "
             "released implementations of the 2026 state of the art",
    )
    ap.add_argument("--notes", default=None)
    ap.add_argument("--dry-run", action="store_true", help="do not write to the registry")
    args = ap.parse_args()

    manifest = ROOT / "docs" / "datasets" / "manifests" / f"{args.dataset}.csv"
    if not manifest.is_file():
        ap.error(f"no manifest at {manifest.relative_to(ROOT)}")
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / "raw" / args.dataset
    work_dir = Path(args.work_dir) if args.work_dir else ROOT / "data" / "work" / (
        f"{args.dataset}_{args.subset or 'all'}_{args.method}"
    )

    label = f"{args.dataset}/{args.subset}" if args.subset else args.dataset
    print(f"baseline: {label}  method={args.method}")

    enhancer = None
    if args.method != "none":
        from fpe.models.pyfing_baseline import PyfingEnhancer

        enhancer = PyfingEnhancer(args.method, cache_dir=work_dir)

    result = run_baseline(
        manifest_path=manifest,
        data_root=data_root,
        work_dir=work_dir,
        subset=args.subset,
        preprocess=enhancer,
        impostor=args.impostor,
        n_resamples=args.bootstrap,
        seed=args.seed,
        progress=print,
    )

    m = result.metrics
    if enhancer is not None:
        m.update(enhancer.inference_seconds)
    print(
        f"\n  EER {m['eer']:.4f}  "
        f"95% CI [{m['eer_ci_low']:.4f}, {m['eer_ci_high']:.4f}]\n"
        f"  enrolled {m['n_enrolled']}/{m['n_images']}  FTA {m['fta_rate']:.3%}  "
        f"minutiae/img {m['minutiae_mean']:.1f}\n"
        f"  pairs {m['n_genuine']} genuine / {m['n_impostor']} impostor"
    )
    if "nfiq2_mean" in m:
        print(f"  NFIQ 2 mean {m['nfiq2_mean']:.1f}  median {m['nfiq2_median']:.0f}")
    if "inference_mean_s" in m:
        print(f"  enhancement {m['inference_mean_s']*1000:.0f} ms/image on CPU "
              f"(n={m['inference_n']}, warm-up excluded)")

    if args.dry_run:
        print("\n  --dry-run: registry not written")
        return 0

    log_run(
        experiment="baseline-nbis",
        dataset=label,
        method=args.method,
        metrics=m,
        config={
            "extractor": "mindtct",
            "matcher": "bozorth3",
            "impostor_mode": args.impostor,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        manifest_path=manifest,
        notes=args.notes,
    )
    print(f"\n  registry row appended -> results/registry/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
