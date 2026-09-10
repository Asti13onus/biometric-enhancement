"""Figure: the latent-style degradation baseline arm, clean vs degraded.

This is the visual audit of `fpe.degradation.latent`. It goes in the thesis as the
depiction of the *control* condition -- the crime-scene-latent degradation model that
Contribution 1 argues is wrong for worn fingerprints. The companion figure for the wear
model will use the same layout, so the two can be read side by side.

    python scripts/figure_latent_degradation.py
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

from fpe.degradation.backgrounds import BackgroundBank  # noqa: E402
from fpe.degradation.latent import LatentConfig, LatentDegradation  # noqa: E402

OUT = ROOT / "results" / "figures" / "latent_degradation_examples.png"
N = 6


def main() -> int:
    sources = sorted((ROOT / "data" / "raw" / "fvc2004" / "db1_b").glob("*.tif"))
    if not sources:
        print("no FVC2004 DB1_B images found; run scripts/download_group_a.py first")
        return 1
    picks = [sources[i] for i in np.linspace(0, len(sources) - 1, N).astype(int)]

    bank = BackgroundBank.from_dtd(ROOT / "data" / "raw" / "dtd", "train")
    degrade = LatentDegradation(LatentConfig(), bank)

    fig, axes = plt.subplots(2, N, figsize=(2.1 * N, 5.0))
    for col, path in enumerate(picks):
        clean = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        degraded, record = degrade(clean, seed=4000 + col)

        axes[0, col].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(path.name, fontsize=8)
        axes[1, col].imshow(degraded, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_xlabel("\n".join(record.applied), fontsize=5.5)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("clean\n(FVC2004 DB1_B)", fontsize=9)
    axes[1, 0].set_ylabel("latent-style\ndegraded", fontsize=9)
    fig.suptitle(
        "Baseline arm: latent-style degradation (ChaLearn artefact list, our implementation)\n"
        f"backgrounds from DTD '{bank.split}' split -- {len(bank)} textures, "
        f"{len(bank.categories)} categories, disjoint from val/test",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
