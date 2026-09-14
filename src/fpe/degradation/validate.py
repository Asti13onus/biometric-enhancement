"""Does the wear model actually look like real wear? (plan U3)

Realism here is **measured, not asserted** -- and the measurement is reported whichever way
it falls. A wear model that turns out to be distinguishable from real degradation is a
finding about the model, not a reason to re-run until the number flatters it.

Two independent measurements, and one comparison that matters more than either:

1. **NFIQ 2 distribution distance.** A two-sample Kolmogorov-Smirnov statistic between the
   quality scores of degraded synthetic prints and real degraded prints. Whole images, as
   NFIQ 2 intends.
2. **Domain classifier accuracy.** A deliberately small classifier trained to tell
   generated from real. Near chance (0.5) is the good outcome; well above chance is a
   reported limitation.

**The comparison of interest is not "is the wear model indistinguishable from real".** It
almost certainly is not. It is whether the wear model is *closer to real than the latent
model is*, because the latent model is what the field currently trains on. Both arms are
measured the same way and reported side by side.

## Two honest caveats, both of which weaken the classifier result

- **Resolution and size confound it.** Anguli images are 275x400 and FVC2004 DB1 is
  640x480. A classifier handed whole images separates on size alone and learns nothing
  about realism. So the classifier sees only fixed-size foreground patches. Any residual
  difference in *ridge scale* between the corpora still leaks into the features, and that
  is not fully controllable without resampling both to a measured common ridge period.
- **A strong classifier would cheat.** Given enough capacity it will find sensor noise,
  JPEG history, or border artefacts rather than anything about wear physiology. The
  classifier here is logistic regression over a couple of dozen hand-chosen features
  precisely so that a high accuracy means something interpretable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import optimize, stats

__all__ = [
    "RealismResult",
    "ks_compare",
    "patch_features",
    "sample_patches",
    "classifier_accuracy",
    "wilson_interval",
]

PATCH = 256
N_HIST_BINS = 16
N_FFT_BANDS = 8


@dataclass(frozen=True)
class RealismResult:
    """One arm of the comparison: a degradation model measured against real prints."""

    label: str
    ks_statistic: float
    ks_pvalue: float
    nfiq2_mean_generated: float
    nfiq2_mean_real: float
    classifier_accuracy: float
    classifier_ci_low: float
    classifier_ci_high: float
    n_generated: int
    n_real: int
    n_patches: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def ks_compare(generated: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and p-value. Lower statistic means closer distributions."""
    a = np.asarray(generated, dtype=np.float64).ravel()
    b = np.asarray(real, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")
    result = stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)


# -- features -------------------------------------------------------------------------

def patch_features(patch: np.ndarray) -> np.ndarray:
    """A compact, interpretable description of one image patch.

    Deliberately hand-chosen rather than learned: if the classifier separates real from
    generated, we want to be able to say *on what*.
    """
    img = np.asarray(patch, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    hist, _ = np.histogram(img, bins=N_HIST_BINS, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64) / max(img.size, 1)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    laplacian = cv2.Laplacian(img, cv2.CV_32F)

    # Radial spectrum: where the energy sits across spatial frequencies is the most
    # direct numerical expression of "do these ridges look like those ridges".
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean())))
    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - h / 2, xx - w / 2)
    max_radius = min(h, w) / 2
    edges = np.linspace(0, max_radius, N_FFT_BANDS + 1)
    total = spectrum.sum() + 1e-8
    bands = [
        float(spectrum[(radius >= lo) & (radius < hi)].sum() / total)
        for lo, hi in zip(edges[:-1], edges[1:])
    ]

    return np.concatenate([
        hist,
        [img.mean(), img.std(), magnitude.mean(), magnitude.std(), laplacian.var()],
        bands,
    ]).astype(np.float64)


def sample_patches(
    image: np.ndarray, n: int, rng: np.random.Generator, size: int = PATCH
) -> list[np.ndarray]:
    """Fixed-size patches with actual ridge content in them.

    Fixed size is what stops the classifier separating on image dimensions; the variance
    floor is what stops it being handed blank background.
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        img = cv2.resize(img, (max(size, int(round(w * scale))),
                               max(size, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    floor = float(np.asarray(img, dtype=np.float32).std()) * 0.5
    out: list[np.ndarray] = []
    for _ in range(n * 12):
        if len(out) >= n:
            break
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        patch = img[y:y + size, x:x + size]
        if float(np.asarray(patch, dtype=np.float32).std()) >= floor:
            out.append(patch)
    return out


# -- a deliberately small classifier ----------------------------------------------------

def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """L2-regularised logistic regression by L-BFGS. Deterministic, no learning rate."""
    design = np.hstack([x, np.ones((x.shape[0], 1))])

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ weights
        # log(1 + exp(z)) written stably.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + l2 * np.sum(weights[:-1] ** 2))
        probability = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (probability - y) / len(y)
        gradient[:-1] += 2 * l2 * weights[:-1]
        return loss, gradient

    result = optimize.minimize(
        objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 500},
    )
    return result.x


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves sensibly near 0 and 1, unlike the normal one."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


MIN_SAMPLES_PER_FEATURE = 20
"""Below this ratio the cross-validated estimate is biased *downwards* and unusable.

Measured on identical distributions, where the true accuracy is 0.5 by construction:
30 samples per class scored 0.317, 75 scored 0.493, 150 scored 0.440, 300 scored 0.470.
The small-sample estimates are not noisy around 0.5, they are systematically below it --
logistic regression separates the training fold perfectly when samples are few relative to
features, and then generalises worse than chance.

This matters because a spuriously low number is the *flattering* direction here: it would
read as "the wear model is indistinguishable from real" when it only means "too few
patches were sampled". So the function refuses rather than returning it. Patches are
cheap; sample more.
"""


def classifier_accuracy(
    generated: np.ndarray, real: np.ndarray, *, folds: int = 5, seed: int = 0
) -> tuple[float, float, float]:
    """Cross-validated accuracy at telling generated patches from real ones.

    0.5 is chance and is the *good* outcome. Returns (accuracy, ci_low, ci_high).

    Raises when given too few samples for the estimate to be trustworthy -- see
    `MIN_SAMPLES_PER_FEATURE`.
    """
    x = np.vstack([np.asarray(generated, dtype=np.float64),
                   np.asarray(real, dtype=np.float64)])
    y = np.concatenate([np.ones(len(generated)), np.zeros(len(real))])
    if len(generated) == 0 or len(real) == 0:
        raise ValueError("both classes must be non-empty")

    required = MIN_SAMPLES_PER_FEATURE * x.shape[1]
    if len(y) < required:
        raise ValueError(
            f"{len(y)} samples for {x.shape[1]} features is too few; the cross-validated "
            f"accuracy is biased below chance in this regime and would read as a "
            f"flattering result. Need at least {required} samples "
            f"({required // 2} per class)."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    assignments = np.arange(len(y)) % folds

    correct = 0
    for fold in range(folds):
        train, test = assignments != fold, assignments == fold
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        centre = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-9] = 1.0
        weights = _fit_logistic((x[train] - centre) / scale, y[train])
        design = np.hstack([(x[test] - centre) / scale, np.ones((test.sum(), 1))])
        predicted = (design @ weights > 0).astype(np.float64)
        correct += int((predicted == y[test]).sum())

    accuracy = correct / len(y)
    low, high = wilson_interval(correct, len(y))
    return float(accuracy), low, high
