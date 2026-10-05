"""Figure: the wear degradation model across a severity sweep, with evidence maps.

This is the visual audit of `fpe.degradation.wear` and the thesis figure for
Contribution 1. It deliberately reuses the layout of
`scripts/figure_latent_degradation.py` so the two degradation models can be read side by
side -- that comparison is the thesis's central experiment.

The second row is the part no published method can draw: because we synthesised the
damage, we know per pixel how much ridge evidence survived. That row is the supervision
target for the network's confidence head.

    python scripts/figure_wear_degradation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.degradation.wear import WearConfig, WearDegradation  # noqa: E402

OUT = ROOT / "results" / "figures" / "wear_degradation_severity.png"
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEED = 20260911


def main() -> int:
    source = ROOT / "data" / "raw" / "fvc2004" / "db1_b" / "101_1.tif"
    if not source.is_file():
        print(f"missing {source.relative_to(ROOT)}; run scripts/download_group_a.py first")
        return 1

    clean = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    degrade = WearDegradation(WearConfig())

    n = len(SEVERITIES)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.4))
    for col, severity in enumerate(SEVERITIES):
        result = degrade(clean, seed=SEED, severity=severity)

        axes[0, col].imshow(result.image, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"severity {severity:.1f}", fontsize=9)

        axes[1, col].imshow(result.evidence, cmap="magma", vmin=0, vmax=1)
        axes[1, col].set_xlabel(f"evidence {result.evidence.mean():.2f}", fontsize=8)

        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("degraded", fontsize=10)
    axes[1, 0].set_ylabel("evidence surviving", fontsize=10)
    fig.suptitle(
        "Wear degradation model: severity sweep and the evidence map it emits",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
