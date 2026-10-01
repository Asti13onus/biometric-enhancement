"""The thesis-defining figure: precision and EER against coverage (plan U9).

Reads the registry's precision-coverage rows (never typed numbers), draws the WAFEN
sweep as a curve and every baseline as a single point at coverage 1.0.

    python scripts/make_precision_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, SUB, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#d95926"


def main() -> int:
    rows: dict[str, dict] = {}
    for line in (ROOT / "results" / "registry" / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("experiment") == "precision-coverage":
            rows[r["method"]] = r["metrics"]  # later rows supersede
    if not rows:
        print("no precision-coverage rows in the registry yet")
        return 1

    sweep = sorted((m["coverage"], m) for name, m in rows.items()
                   if name.startswith("WAFEN@"))
    points = [(name, m) for name, m in rows.items() if not name.startswith("WAFEN@")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, key, title in ((a, "precision", "Minutiae precision vs coverage"),
                           (b, "eer", "Verification EER vs coverage (lower is better)")):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=SUB, labelsize=8)
        ax.grid(color="#e8e7e2", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("foreground coverage", fontsize=8.5, color=SUB)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")

        xs = [c for c, _ in sweep]
        ys = [m[key] for _, m in sweep]
        ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, markersize=5,
                label="WAFEN (abstaining)")
        for name, m in points:
            marker = "s" if name == "none" else "D"
            color = SUB if name == "none" else ORANGE
            ax.scatter([m["coverage"]], [m[key]], marker=marker, s=42, color=color,
                       zorder=3)
            ax.annotate(name if name != "none" else "no enhancement",
                        (m["coverage"], m[key]), textcoords="offset points",
                        xytext=(-6, 7), fontsize=7.5, color=SUB, ha="right")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=SUB, loc="lower left")

    fig.suptitle("Synthetic pseudo-GT (clean-master mindtct); wear severity 0.5 - "
                 "not comparable to published SD27 numbers", fontsize=7.5, color=SUB,
                 y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = ROOT / "results" / "figures" / "precision_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
