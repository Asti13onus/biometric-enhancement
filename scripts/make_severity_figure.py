"""The severity-crossover figure (U12/AE3): EER vs wear severity per method.

Reads the registry's CrossMatch benchmark rows (clean + the wear grid) and draws one
curve per method with bootstrap intervals. The story it carries: enhancement is harmful
on clean prints, breakeven at light wear, and dramatically beneficial at heavy wear --
the quality-gating rule for deployment, measured on one sensor with everything fixed.

    python scripts/make_severity_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB = "#0b0b0b", "#52514e"
SERIES = {"none": ("#52514e", "No enhancement"),
          "SNFEN": ("#2a78d6", "SNFEN (4 networks)"),
          "WAFEN[wafen_da]": ("#d95926", "WAFEN aligned (ours, 1 network)")}


def main() -> int:
    rows: dict[tuple[float, str], dict] = {}
    for line in (ROOT / "results" / "registry" / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("experiment") != "benchmark":
            continue
        d = r["dataset"]
        if d == "neurotech_crossmatch":
            rows[(0.0, r["method"])] = r["metrics"]
        elif d.startswith("neurotech_crossmatch (wear"):
            rows[(float(d.split("wear")[1].strip(" )")), r["method"])] = r["metrics"]
    severities = sorted({s for s, _ in rows})

    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=SUB, labelsize=8.5)
    ax.grid(color="#e8e7e2", linewidth=0.7)
    ax.set_axisbelow(True)

    for method, (color, label) in SERIES.items():
        pts = [(s, rows[(s, method)]) for s in severities if (s, method) in rows]
        xs = [s for s, _ in pts]
        ys = [m["eer"] for _, m in pts]
        lo = [m["eer"] - m["eer_ci_low"] for _, m in pts]
        hi = [m["eer_ci_high"] - m["eer"] for _, m in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color=color, linewidth=2,
                    markersize=5, capsize=3, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=color, va="center")

    ax.set_xlabel("wear severity (0 = as captured)", fontsize=9, color=SUB)
    ax.set_ylabel("verification EER", fontsize=9, color=SUB)
    ax.set_title("When does enhancement pay? CrossMatch under increasing wear",
                 fontsize=10.5, color=INK, loc="left")
    ax.set_xlim(-0.03, 0.92)
    ax.legend(frameon=False, fontsize=8, labelcolor=SUB, loc="upper left")
    fig.text(0.01, 0.01, "Same 408 prints and subjects at every severity; wear applied "
             "by the thesis's degradation model; 95% bootstrap intervals.",
             fontsize=7, color=SUB)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "results" / "figures" / "severity_crossover.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
