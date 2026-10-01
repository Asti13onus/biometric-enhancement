"""Biometric error rates from genuine and impostor score distributions.

Convention throughout: **higher score means more similar**, which is what
BOZORTH3 returns. A dissimilarity metric must be negated before it gets here.

EER is reported with a bootstrap confidence interval rather than as a point
estimate, because several of our evaluation sets are small -- an FVC "B"
subset yields only 280 genuine pairs, and a point EER on that is not
meaningful on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["ErrorRates", "PairedDifference", "compute_eer", "bootstrap_eer",
           "paired_bootstrap_eer_difference", "far_at_frr", "frr_at_far"]


@dataclass(frozen=True)
class ErrorRates:
    """Summary of a verification experiment."""

    eer: float
    threshold: float
    n_genuine: int
    n_impostor: int
    eer_ci_low: float | None = None
    eer_ci_high: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_scores(genuine, impostor) -> tuple[np.ndarray, np.ndarray]:
    gen = np.asarray(genuine, dtype=np.float64).ravel()
    imp = np.asarray(impostor, dtype=np.float64).ravel()
    if gen.size == 0 or imp.size == 0:
        raise ValueError(
            f"need both genuine and impostor scores (got {gen.size} and {imp.size})"
        )
    if not (np.isfinite(gen).all() and np.isfinite(imp).all()):
        raise ValueError("scores contain NaN or inf")
    return gen, imp


def _rates(gen: np.ndarray, imp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FMR and FNMR over every threshold that can change a decision.

    Accepting when score >= threshold:
      FMR(t)  = fraction of impostor scores >= t
      FNMR(t) = fraction of genuine  scores <  t
    """
    thresholds = np.unique(np.concatenate([gen, imp]))
    # Sweep one step past the top so the all-reject corner is representable.
    step = np.diff(thresholds).min() if thresholds.size > 1 else 1.0
    thresholds = np.append(thresholds, thresholds[-1] + step)

    gen_sorted = np.sort(gen)
    imp_sorted = np.sort(imp)
    # searchsorted "left" counts strictly-below, which is exactly FNMR's numerator.
    fnmr = np.searchsorted(gen_sorted, thresholds, side="left") / gen.size
    fmr = 1.0 - np.searchsorted(imp_sorted, thresholds, side="left") / imp.size
    return thresholds, fmr, fnmr


def compute_eer(genuine, impostor) -> ErrorRates:
    """Equal error rate, interpolated at the FMR/FNMR crossing."""
    gen, imp = _as_scores(genuine, impostor)
    thresholds, fmr, fnmr = _rates(gen, imp)

    diff = fmr - fnmr
    # FMR falls and FNMR rises with the threshold, so diff crosses zero once.
    idx = int(np.argmin(np.abs(diff)))
    if diff[idx] == 0 or idx + 1 >= diff.size:
        eer = float((fmr[idx] + fnmr[idx]) / 2)
        threshold = float(thresholds[idx])
    else:
        # Linearly interpolate between the two thresholds bracketing the crossing.
        j = idx + 1 if diff[idx] > 0 else idx - 1
        j = int(np.clip(j, 0, diff.size - 1))
        lo, hi = (idx, j) if idx < j else (j, idx)
        denom = diff[lo] - diff[hi]
        w = 0.5 if denom == 0 else diff[lo] / denom
        eer = float((1 - w) * (fmr[lo] + fnmr[lo]) / 2 + w * (fmr[hi] + fnmr[hi]) / 2)
        threshold = float((1 - w) * thresholds[lo] + w * thresholds[hi])

    return ErrorRates(
        eer=eer, threshold=threshold, n_genuine=int(gen.size), n_impostor=int(imp.size)
    )


def bootstrap_eer(
    genuine, impostor, *, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> ErrorRates:
    """EER with a percentile bootstrap interval.

    Genuine and impostor sets are resampled independently, which is the usual
    treatment and keeps each set's size fixed.
    """
    gen, imp = _as_scores(genuine, impostor)
    point = compute_eer(gen, imp)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        g = rng.choice(gen, size=gen.size, replace=True)
        m = rng.choice(imp, size=imp.size, replace=True)
        draws[i] = compute_eer(g, m).eer

    tail = (1.0 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return ErrorRates(
        eer=point.eer,
        threshold=point.threshold,
        n_genuine=point.n_genuine,
        n_impostor=point.n_impostor,
        eer_ci_low=float(low),
        eer_ci_high=float(high),
    )


def frr_at_far(genuine, impostor, far: float) -> float:
    """FNMR at the operating point whose FMR is at most `far`."""
    gen, imp = _as_scores(genuine, impostor)
    thresholds, fmr, fnmr = _rates(gen, imp)
    ok = np.flatnonzero(fmr <= far)
    if ok.size == 0:
        return 1.0
    return float(fnmr[ok[0]])


def far_at_frr(genuine, impostor, frr: float) -> float:
    """FMR at the operating point whose FNMR is at most `frr`."""
    gen, imp = _as_scores(genuine, impostor)
    thresholds, fmr, fnmr = _rates(gen, imp)
    ok = np.flatnonzero(fnmr <= frr)
    if ok.size == 0:
        return 1.0
    return float(fmr[ok[-1]])


@dataclass(frozen=True)
class PairedDifference:
    """EER difference B - A with a paired-bootstrap interval. Negative favours B."""

    eer_a: float
    eer_b: float
    difference: float
    ci_low: float
    ci_high: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_dict(self) -> dict[str, object]:
        return {"eer_a": self.eer_a, "eer_b": self.eer_b,
                "eer_difference": self.difference,
                "diff_ci_low": self.ci_low, "diff_ci_high": self.ci_high,
                "excludes_zero": self.excludes_zero, "n_resamples": self.n_resamples}


def paired_bootstrap_eer_difference(
    genuine_a, impostor_a, genuine_b, impostor_b, *,
    n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0,
) -> PairedDifference:
    """Bootstrap the EER *difference* by resampling pairs, not scores.

    The marginal intervals of two conditions evaluated on the same pairs share the
    sampling noise of those pairs; comparing them by interval overlap throws that
    pairing away and is far too weak to resolve differences of a point or two of EER.
    Here each resample draws one set of genuine indices and one set of impostor indices
    and applies them to *both* conditions, so the pair-sampling noise cancels inside
    the difference. Inputs must therefore be aligned: element i of `genuine_a` and
    `genuine_b` is the same physical pair under the two conditions.
    """
    ga, ia = _as_scores(genuine_a, impostor_a)
    gb, ib = _as_scores(genuine_b, impostor_b)
    if ga.size != gb.size or ia.size != ib.size:
        raise ValueError(
            f"paired testing needs identical pair sets: genuine {ga.size} vs {gb.size}, "
            f"impostor {ia.size} vs {ib.size}. Restrict both conditions to the pairs "
            f"enrolled under both before calling this."
        )

    point_a = compute_eer(ga, ia).eer
    point_b = compute_eer(gb, ib).eer

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        g_idx = rng.integers(0, ga.size, size=ga.size)
        i_idx = rng.integers(0, ia.size, size=ia.size)
        draws[i] = compute_eer(gb[g_idx], ib[i_idx]).eer - compute_eer(ga[g_idx], ia[i_idx]).eer

    tail = (1.0 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return PairedDifference(eer_a=point_a, eer_b=point_b,
                            difference=point_b - point_a,
                            ci_low=float(low), ci_high=float(high),
                            n_resamples=n_resamples)
