"""Summarise the committed dataset manifests into one table.

Feeds docs/datasets/DATASET_DOSSIER.md. Every per-dataset number in that document
comes from here, not from prose -- PROJECT_RULES.md non-negotiable 2.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "docs" / "datasets" / "manifests"
OUT = ROOT / "docs" / "datasets" / "manifest_summary.csv"

FIELDS = [
    "dataset", "images", "megabytes", "distinct_sizes", "size_range",
    "declared_dpi", "subjects", "greyscale_pct",
]


def summarise(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    dims = Counter((r["width"], r["height"]) for r in rows)
    dpi = sorted({r["declared_dpi"] for r in rows if r["declared_dpi"]})
    subjects = {r["subject"] for r in rows if r.get("subject")}
    modes = Counter(r["mode"] for r in rows)
    px = sorted({(int(w) * int(h), w, h) for w, h in dims})
    grey = modes.get("L", 0) / len(rows) * 100 if rows else 0.0
    return {
        "dataset": path.stem,
        "images": len(rows),
        "megabytes": round(sum(int(r["bytes"]) for r in rows) / 1e6, 1),
        "distinct_sizes": len(dims),
        "size_range": f"{px[0][1]}x{px[0][2]}" if len(px) == 1
                      else f"{px[0][1]}x{px[0][2]} to {px[-1][1]}x{px[-1][2]}",
        "declared_dpi": "/".join(dpi) if dpi else "unstated",
        "subjects": len(subjects) if subjects else "",
        "greyscale_pct": round(grey, 1),
    }


def main() -> int:
    manifests = sorted(MANIFESTS.glob("*.csv"))
    if not manifests:
        print(f"no manifests under {MANIFESTS}", file=sys.stderr)
        return 1
    summaries = [summarise(p) for p in manifests]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    widths = {f: max(len(f), *(len(str(s[f])) for s in summaries)) for f in FIELDS}
    print(" | ".join(f.ljust(widths[f]) for f in FIELDS))
    print("-+-".join("-" * widths[f] for f in FIELDS))
    for s in summaries:
        print(" | ".join(str(s[f]).ljust(widths[f]) for f in FIELDS))
    total = sum(s["images"] for s in summaries)
    print(f"\n{total:,} images, {sum(s['megabytes'] for s in summaries):,.0f} MB across "
          f"{len(summaries)} manifested datasets -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
