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
