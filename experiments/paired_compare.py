"""Paired-bootstrap comparison of two conditions' EER on the same pairs (plan U11).

Runs both conditions through the same harness (enhanced images and NFIQ 2 caches make
re-runs cheap), verifies the pair sets are identical, and bootstraps the EER
*difference* by resampling pairs once per draw -- the powered test the marginal
intervals could not provide. Optionally swaps the extraction chain to LEADER, testing
whether a result survives a different extractor.

    python experiments/paired_compare.py --a none --b SNFEN
    python experiments/paired_compare.py --a none --b WAFEN --checkpoint-b data/work/wafen_da/best.pt
    python experiments/paired_compare.py --a none --b SNFEN --extractor leader
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.eval.baseline import run_baseline  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402
from fpe.metrics.matching import paired_bootstrap_eer_difference  # noqa: E402


def make_enhancer(method: str, checkpoint: str | None, work_dir: Path):
    if method == "none":
        return None
    if method == "WAFEN":
        from fpe.models.wafen_enhancer import WafenEnhancer

        return WafenEnhancer(checkpoint or ROOT / "data/work/wafen_da/best.pt",
                             cache_dir=work_dir)
    from fpe.models.pyfing_baseline import PyfingEnhancer

    return PyfingEnhancer(method, cache_dir=work_dir)


def condition_label(method: str, checkpoint: str | None) -> str:
    if method == "WAFEN" and checkpoint:
        return f"WAFEN[{Path(checkpoint).parent.name}]"
    return method


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fvc2004")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--a", required=True, choices=["none", "SNFEN", "GBFEN", "WAFEN"])
    ap.add_argument("--b", required=True, choices=["none", "SNFEN", "GBFEN", "WAFEN"])
    ap.add_argument("--checkpoint-a", default=None)
    ap.add_argument("--checkpoint-b", default=None)
    ap.add_argument("--extractor", choices=["mindtct", "leader"], default="mindtct")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = ROOT / "docs" / "datasets" / "manifests" / f"{args.dataset}.csv"
    data_root = ROOT / "data" / "raw" / args.dataset
    chain = "" if args.extractor == "mindtct" else "_leader"

    results = {}
    for side, method, ckpt in (("a", args.a, args.checkpoint_a),
                               ("b", args.b, args.checkpoint_b)):
        label = condition_label(method, ckpt)
        work = ROOT / "data" / "work" / (
            f"{args.dataset}_{args.subset or 'all'}_{label.replace('[', '_').rstrip(']')}"
            + chain)
        kwargs = {}
        if args.extractor == "leader":
            from fpe.models.leader import LeaderExtractor

            kwargs["extract"] = LeaderExtractor(work / "leader_templates")
        print(f"condition {side.upper()}: {label} ({args.extractor})")
        results[side] = (label, run_baseline(
            manifest_path=manifest, data_root=data_root, work_dir=work,
            subset=args.subset,
            preprocess=make_enhancer(method, ckpt, work),
            progress=lambda m: print(m), **kwargs))

    (la, ra), (lb, rb) = results["a"], results["b"]
    if ra.failed or rb.failed:
        both = set(ra.failed) | set(rb.failed)
        print(f"NOTE: {len(both)} image(s) failed to enrol in at least one condition; "
              f"pair sets may differ and the comparison would be invalid.")
    d = paired_bootstrap_eer_difference(
        ra.genuine_scores, ra.impostor_scores, rb.genuine_scores, rb.impostor_scores,
        n_resamples=args.resamples)

    verdict = ("resolved: B better" if d.excludes_zero and d.difference < 0 else
               "resolved: B worse" if d.excludes_zero else
               "not resolved (interval includes zero)")
    print(f"\n  {la}: EER {d.eer_a:.4f}   {lb}: EER {d.eer_b:.4f}")
    print(f"  paired difference (B - A): {d.difference:+.4f} "
          f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}]  ->  {verdict}")

    if not args.dry_run:
        log_run(
            experiment="paired-compare",
            dataset=f"{args.dataset}/{args.subset}" if args.subset else args.dataset,
            method=f"{lb} vs {la} ({args.extractor})",
            metrics={**d.as_dict(), "verdict": verdict},
            config={"a": la, "b": lb, "extractor": args.extractor,
                    "resamples": args.resamples,
                    "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b},
            manifest_path=manifest,
        )
        print("  registry row appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
