"""Measure how closely each degradation model resembles real degradation (plan U3).

    python experiments/validate_wdm.py

Two measurements, both run for our wear model and for the conventional latent model that
the field currently trains on. The question is never "is this indistinguishable from real"
-- it is **which arm is closer**, measured the same way for both.

Whatever comes out is reported. Acceptance example AE4 commits the thesis to publishing an
unfavourable realism result rather than re-running until it flatters, so the registry write
happens regardless of the numbers.

## Why the classifier compares within one corpus

The obvious design -- degrade Anguli prints, then ask a classifier to tell them from real
degraded prints -- was tried first and is **worthless**. Measured control: clean Anguli
against clean FVC scores **1.000** accuracy before any degradation is applied at all. The
classifier separates the two corpora, not the two degradation models, and degrading the
images actually *lowers* its accuracy because the damage partly masks the corpus signature.

So the classifier arm never crosses corpora. Real prints are pooled and split by measured
NFIQ 2 quality **within each sensor subset**, which keeps sensors balanced on both sides.
The clean half is then degraded by each model and tested against the genuinely degraded
half. Same sensors, same subjects, same acquisition pipeline -- what remains is the
difference between synthetic damage and real damage, which is the thing being measured.

The NFIQ 2 distribution test is a different question and keeps the synthetic corpus: "does
the data we actually train on have a quality distribution like real worn prints?" That is
legitimately about the training corpus, so it is reported as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.data.convert import load_greyscale  # noqa: E402
from fpe.degradation.backgrounds import BackgroundBank  # noqa: E402
from fpe.degradation.latent import LatentConfig, LatentDegradation  # noqa: E402
from fpe.degradation.validate import (  # noqa: E402
    classifier_accuracy, ks_compare, patch_features, sample_patches,
)
from fpe.degradation.wear import WearConfig, WearDegradation  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402
from fpe.metrics.nfiq2 import score_images  # noqa: E402

FIGURE = ROOT / "results" / "figures" / "degradation_model_comparison.png"
REAL_DEGRADED = [("fvc2004", "db1_b"), ("fvc2004", "db3_b")]
POOL = [("fvc2000", "*"), ("fvc2002", "*"), ("fvc2004", "*"),
        ("neurotech_crossmatch", None), ("neurotech_uareu", None)]
PATCHES_PER_IMAGE = 4
SEVERITY_RANGE = (0.3, 1.0)
ARMS = ("latent", "wear")


def collect(dataset: str, subset: str | None) -> dict[str, list[Path]]:
    """Image paths grouped by sensor subset, so quality splits stay sensor-balanced."""
    root = ROOT / "data" / "raw" / dataset
    if not root.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    if subset is None:
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in {".tif", ".bmp"})
        if files:
            out[dataset] = files
        return out
    for child in sorted(root.iterdir()):
        if child.is_dir():
            files = sorted(p for p in child.glob("*")
                           if p.suffix.lower() in {".tif", ".bmp"})
            if files:
                out[f"{dataset}/{child.name}"] = files
    return out


def split_by_quality(rng: np.random.Generator, cache: Path
                     ) -> tuple[list[Path], list[Path]]:
    """Split the real pool into a clean half and a degraded half, per sensor subset.

    Images are converted to greyscale PNG first. NFIQ 2 silently fails on
    Neurotechnology's palette-mode TIFFs -- it reports nothing rather than erroring -- and
    scoring raw files dropped all 928 of them out of the pool without a word. The week-1
    harness never hit this because it scores converted images; this script did.
    """
    from fpe.data.convert import to_greyscale_png

    groups: dict[str, list[Path]] = {}
    for dataset, subset in POOL:
        groups.update(collect(dataset, None if subset is None else subset))

    clean: list[Path] = []
    degraded: list[Path] = []
    for name, paths in sorted(groups.items()):
        prepared = [to_greyscale_png(p, cache / name.replace("/", "_")) for p in paths]
        scores = score_images(prepared)
        usable = [(s.score, Path(s.path)) for s in scores if s.ok]
        if len(usable) < 8:
            # Loud, not silent: a whole sensor vanishing changes what the result means.
            print(f"  {name:28s} SKIPPED -- only {len(usable)}/{len(paths)} scored")
            continue
        if len(usable) < len(paths):
            print(f"  {name:28s} note: {len(paths) - len(usable)} image(s) unscored")
        usable.sort(key=lambda t: t[0])
        middle = len(usable) // 2
        degraded.extend(p for _, p in usable[:middle])   # lowest quality
        clean.extend(p for _, p in usable[middle:])      # highest quality
        print(f"  {name:28s} {len(usable):4d} images, "
              f"NFIQ 2 median {usable[middle][0]:3d}")
    rng.shuffle(clean)
    rng.shuffle(degraded)
    return clean, degraded


def degrade_all(sources, model, work: Path, rng, is_wear: bool) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for n, path in enumerate(sources):
        image = load_greyscale(path)
        seed = int(rng.integers(0, 2**31 - 1))
        if is_wear:
            result = model(image, seed=seed,
                           severity=float(rng.uniform(*SEVERITY_RANGE))).image
        else:
            result, _ = model(image, seed=seed)
        destination = work / f"{n:05d}.png"
        Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(destination)
        out.append(destination)
    return out


def features_for(paths, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        [patch_features(p) for path in paths
         for p in sample_patches(load_greyscale(path), PATCHES_PER_IMAGE, rng)],
        dtype=np.float64,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-synthetic", type=int, default=160)
    ap.add_argument("--n-real", type=int, default=400,
                    help="clean real prints to degrade for the classifier arm")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--work-dir", type=Path,
                    default=ROOT / "data" / "work" / "wdm_validation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    wear = WearDegradation(WearConfig())
    latent = LatentDegradation(
        LatentConfig(), BackgroundBank.from_dtd(ROOT / "data" / "raw" / "dtd", "train")
    )

    # -- arm 1: does our synthetic training corpus look like real worn prints? ------
    real_degraded = [p for d, s in REAL_DEGRADED
                     for p in sorted((ROOT / "data" / "raw" / d / s).glob("*.tif"))]
    anguli_root = ROOT / "data" / "processed" / "anguli_dev_5584" / "Impression_1"
    anguli = sorted(p for b in sorted(anguli_root.iterdir()) if b.is_dir()
                    for p in sorted(b.glob("*.png")))
    anguli = anguli[::max(1, len(anguli) // args.n_synthetic)][:args.n_synthetic]
    print(f"NFIQ 2 arm: {len(anguli)} synthetic sources vs {len(real_degraded)} real degraded")

    synthetic = {
        arm: degrade_all(anguli, wear if arm == "wear" else latent,
                         args.work_dir / f"syn_{arm}", rng, arm == "wear")
        for arm in ARMS
    }
    nfiq_real = np.array([s.score for s in score_images(real_degraded) if s.ok], float)
    nfiq = {a: np.array([s.score for s in score_images(v) if s.ok], float)
            for a, v in synthetic.items()}

    # -- arm 2: within-corpus, so the corpus signature cannot be what is measured ----
    print("\nsplitting the real pool by measured quality (per sensor):")
    clean_real, degraded_real = split_by_quality(rng, args.work_dir / "prepared")
    clean_real = clean_real[:args.n_real]
    print(f"  -> {len(clean_real)} clean sources, {len(degraded_real)} genuinely degraded")

    within = {
        arm: degrade_all(clean_real, wear if arm == "wear" else latent,
                         args.work_dir / f"real_{arm}", rng, arm == "wear")
        for arm in ARMS
    }
    features_degraded = features_for(degraded_real, args.seed)

    print(f"\n{'arm':8s} {'KS vs real':>11s} {'NFIQ2':>8s} {'classifier acc':>18s}"
          f"   (0.50 = indistinguishable)")
    results = {}
    for arm in ARMS:
        ks, p = ks_compare(nfiq[arm], nfiq_real)
        accuracy, low, high = classifier_accuracy(
            features_for(within[arm], args.seed), features_degraded, seed=args.seed
        )
        results[arm] = dict(
            ks_statistic=ks, ks_pvalue=p,
            nfiq2_mean_generated=float(nfiq[arm].mean()),
            nfiq2_mean_real=float(nfiq_real.mean()),
            classifier_accuracy=accuracy,
            classifier_ci_low=low, classifier_ci_high=high,
            n_generated=len(nfiq[arm]), n_real=len(nfiq_real),
            n_clean_sources=len(clean_real), n_real_degraded=len(degraded_real),
        )
        print(f"{arm:8s} {ks:11.4f} {nfiq[arm].mean():8.1f} "
              f"{accuracy:12.3f} [{low:.2f},{high:.2f}]")
    print(f"{'real':8s} {'-':>11s} {nfiq_real.mean():8.1f}")

    closer = min(results, key=lambda a: results[a]["ks_statistic"])
    harder = min(results, key=lambda a: abs(results[a]["classifier_accuracy"] - 0.5))
    print(f"\ncloser to real by NFIQ 2 distribution: {closer}")
    print(f"harder to distinguish from real damage: {harder}")

    make_figure(real_degraded, within, clean_real)

    if args.dry_run:
        print("\n--dry-run: registry not written")
        return 0
    for arm, metrics in results.items():
        log_run(
            experiment="wdm-realism", method=arm,
            dataset="real pool split by NFIQ 2; anguli_dev_5584 for the NFIQ 2 arm",
            metrics=metrics,
            config={"severity_range": list(SEVERITY_RANGE), "seed": args.seed,
                    "patches_per_image": PATCHES_PER_IMAGE,
                    "classifier_design": "within-corpus quality split"},
            notes="AE4: reported regardless of outcome. Cross-corpus classifier design "
                  "was abandoned: clean synthetic vs clean real scores 1.000 before any "
                  "degradation, so it measures the corpora, not the models.",
        )
    print("\nregistry rows appended -> results/registry/runs.jsonl")
    return 0


def make_figure(real_degraded, within, clean_real, n: int = 6) -> None:
    rows = [
        ("real, clean", [load_greyscale(p) for p in clean_real[:n]]),
        ("real, degraded", [load_greyscale(p) for p in real_degraded[:n]]),
        ("latent model", [load_greyscale(p) for p in within["latent"][:n]]),
        ("wear model (ours)", [load_greyscale(p) for p in within["wear"][:n]]),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.1 * n, 2.4 * len(rows)))
    for r, (label, images) in enumerate(rows):
        for c, image in enumerate(images):
            axes[r, c].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Real degradation against two degradation models, same source corpus",
                 fontsize=11)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140, bbox_inches="tight")
    print(f"wrote {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
