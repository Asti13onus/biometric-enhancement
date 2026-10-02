"""Generate thesis tables from the results registry.

The registry is the only source of numbers (PROJECT_RULES.md non-negotiable 2), so
tables are queried from it rather than typed. Re-running this after new
experiments regenerates every table.

    python scripts/make_tables.py                  # print to stdout
    python scripts/make_tables.py --write          # write results/tables/*.md

Where two conditions have overlapping confidence intervals, that is stated
rather than left for the reader to infer from the numbers -- on our smaller
evaluation sets it is usually the honest conclusion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.eval.registry import read_runs  # noqa: E402

METHOD_ORDER = {"none": 0, "GBFEN": 1, "SNFEN": 2}
METHOD_LABEL = {"none": "No enhancement"}

CHECKPOINT_LABEL = {
    "wafen": "WAFEN (arm 1: corrected synthesis)",
    "wafen_ft": "WAFEN distilled (arm 1 + teacher labels)",
    "wafen_mn": "WAFEN minutia loss (arm 1)",
    "wafen_pc": "WAFEN pair-consistent (arm 1)",
    "wafen_da": "WAFEN aligned (arm 2: domain alignment)",
    "wafen_da20": "WAFEN aligned, 20 ep (arm 2)",
}


def _method_label(row: dict) -> str:
    """One row per *variant*, not per method name.

    Every WAFEN configuration was registered as method "WAFEN"; the checkpoint and the
    rendering flags in `config` are what distinguish the thesis's arms and variants.
    Without this, "last row wins" silently replaces arm one's number with arm two's.
    """
    method = row.get("method") or "?"
    cfg = row.get("config") or {}
    if not method.startswith("WAFEN"):
        return method
    stem = Path(cfg.get("checkpoint", "wafen/best.pt")).parent.name
    label = CHECKPOINT_LABEL.get(stem, f"{method} [{stem}]")
    if method == "WAFEN+SNFEN":
        label = f"WAFEN mask+ori -> SNFEN [{stem}]"
    if cfg.get("blend"):
        label += " + confidence blend"
    if cfg.get("abstain_threshold") is not None:
        label += (f" + abstention t={cfg['abstain_threshold']}"
                  + (f" block={cfg['block']}" if cfg.get("block") else ""))
    return label


def _latest_per_condition(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Last row wins for each (dataset, variant): the registry is append-only,
    so a re-run supersedes rather than replaces."""
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("experiment") != "baseline-nbis":
            continue
        latest[(r.get("dataset") or "?", _method_label(r))] = r
    return latest


def baseline_table(rows: list[dict]) -> str:
    latest = _latest_per_condition(rows)
    if not latest:
        return "_No baseline runs in the registry yet._\n"

    lines = [
        "| Dataset | Method | EER | 95% CI | Minutiae/img | NFIQ 2 mean | FTA | CPU ms/img |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset in sorted({d for d, _ in latest}):
        methods = sorted(
            (m for d, m in latest if d == dataset),
            key=lambda m: (METHOD_ORDER.get(m, 99), m),
        )
        for method in methods:
            m = latest[(dataset, method)]["metrics"]
            ms = m.get("inference_mean_s")
            lines.append(
                f"| {dataset} | {METHOD_LABEL.get(method, method)} "
                f"| {m['eer']:.4f} "
                f"| [{m['eer_ci_low']:.4f}, {m['eer_ci_high']:.4f}] "
                f"| {m.get('minutiae_mean', float('nan')):.1f} "
                f"| {m.get('nfiq2_mean', float('nan')):.1f} "
                f"| {m.get('fta_rate', 0):.1%} "
                f"| {f'{ms * 1000:.0f}' if ms else '—'} |"
            )
    return "\n".join(lines) + "\n"


def significance_notes(rows: list[dict]) -> str:
    """State plainly where intervals overlap, per dataset, against the control."""
    latest = _latest_per_condition(rows)
    out: list[str] = []
    for dataset in sorted({d for d, _ in latest}):
        control = latest.get((dataset, "none"))
        if control is None:
            continue
        c = control["metrics"]
        for method in sorted({m for d, m in latest if d == dataset and m != "none"}):
            t = latest[(dataset, method)]["metrics"]
            overlap = (
                t["eer_ci_low"] <= c["eer_ci_high"] and c["eer_ci_low"] <= t["eer_ci_high"]
            )
            direction = "lower" if t["eer"] < c["eer"] else "higher"
            verdict = (
                "intervals overlap — not resolved at this sample size"
                if overlap
                else f"intervals disjoint — {direction} EER"
            )
            out.append(
                f"- **{dataset}**, {method} vs. no enhancement: "
                f"{t['eer']:.4f} vs. {c['eer']:.4f} ({direction}); {verdict}."
            )
    return ("\n".join(out) + "\n") if out else ""


def _eer_cell(m: dict) -> str:
    return f"{m['eer']:.4f} [{m['eer_ci_low']:.4f}, {m['eer_ci_high']:.4f}]"


def _latest(rows, experiment):
    """Append-only registry: the last row wins per (dataset, method)."""
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("experiment") == experiment:
            latest[(r["dataset"], r["method"])] = r
    return latest


def benchmark_table(rows: list[dict]) -> str:
    """U12: held-out sets, the severity axis, and MINEX position strata."""
    latest = _latest(rows, "benchmark")
    if not latest:
        return "_No benchmark rows yet._\n"
    methods = sorted({m for _, m in latest},
                     key=lambda m: (0 if m == "none" else 1 if m == "SNFEN" else 2, m))

    def block(title, datasets):
        if not datasets:
            return ""
        head = "| Dataset | " + " | ".join(METHOD_LABEL.get(m, m) for m in methods) + " |"
        out = [f"### {title}", "", head, "|---|" + "---|" * len(methods)]
        for d in datasets:
            cells = [(_eer_cell(latest[(d, m)]["metrics"]) if (d, m) in latest
                      else "absent") for m in methods]
            out.append(f"| {d} | " + " | ".join(cells) + " |")
        return "\n".join(out) + "\n\n"

    all_ds = sorted({d for d, _ in latest})
    plain = [d for d in all_ds if "wear" not in d and "/" not in d]
    wear = sorted((d for d in all_ds if "wear" in d),
                  key=lambda d: float(d.split("wear")[1].strip(" )")))
    strata = [d for d in all_ds if d.startswith("minex/")]
    return (block("Held-out real sets (EER)", plain)
            + block("Wear-severity axis, CrossMatch (EER)", wear)
            + block("MINEX finger-position strata (EER)", strata))


def precision_coverage_table(rows: list[dict]) -> str:
    """U9. Synthetic pseudo-GT throughout; never comparable to SD27 numbers."""
    latest = {r["method"]: r for r in rows
              if r.get("experiment") == "precision-coverage"}
    if not latest:
        return "_No precision-coverage rows yet._\n"
    out = ["| Condition | Coverage | Precision | Recall | F1 | Type-exact P | EER |",
           "|---|---|---|---|---|---|---|"]
    order = sorted(latest, key=lambda k: (k.startswith("WAFEN@"),
                                          -latest[k]["metrics"]["coverage"], k))
    for name in order:
        m = latest[name]["metrics"]
        out.append(f"| {name} | {m['coverage']:.1%} | {m['precision']:.3f} "
                   f"| {m['recall']:.3f} | {m['f1']:.3f} | {m['tx_precision']:.3f} "
                   f"| {m['eer']:.4f} |")
    return ("\n".join(out) + "\n\n_Synthetic pseudo-GT (clean-master mindtct), wear "
            "severity 0.5; not comparable to published SD27 numbers._\n")


def paired_table(rows: list[dict]) -> str:
    """U11: the powered paired-bootstrap verdicts."""
    seen: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("experiment") == "paired-compare":
            seen[(r["dataset"], r["method"])] = r
    if not seen:
        return "_No paired comparisons yet._\n"
    out = ["| Dataset | Comparison | EER difference (B - A) | 95% CI | Verdict |",
           "|---|---|---|---|---|"]
    for (d, name), r in sorted(seen.items()):
        m = r["metrics"]
        out.append(f"| {d} | {name} | {m['eer_difference']:+.4f} "
                   f"| [{m['diff_ci_low']:+.4f}, {m['diff_ci_high']:+.4f}] "
                   f"| {m['verdict']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write results/tables/")
    args = ap.parse_args()

    # The Windows console defaults to cp1252 and mangles the dashes below.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = read_runs()
    if not rows:
        print("registry is empty", file=sys.stderr)
        return 1

    body = (
        "# Baseline results\n\n"
        "Generated by `scripts/make_tables.py` from `results/registry/runs.jsonl`. "
        "Do not edit by hand.\n\n"
        "## Verification performance\n\n"
        + baseline_table(rows)
        + "\n## Against the no-enhancement control\n\n"
        + significance_notes(rows)
        + "\n## Benchmark (U12)\n\n"
        + benchmark_table(rows)
        + "\n## Minutiae precision and coverage (U9)\n\n"
        + precision_coverage_table(rows)
        + "\n## Paired comparisons (U11)\n\n"
        + paired_table(rows)
    )
    print(body)

    if args.write:
        out = ROOT / "results" / "tables" / "baselines.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
