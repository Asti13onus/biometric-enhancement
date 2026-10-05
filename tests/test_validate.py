"""Tests for realism validation (plan U3).

The trap this suite guards against: a measurement that *always* returns a flattering
answer. If the domain classifier cannot detect separation even when the two sets are
trivially different, then "accuracy near chance" proves nothing about the wear model and
the whole validation is theatre. So the harness is checked against cases where the right
answer is known in both directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpe.degradation.validate import (
    classifier_accuracy, ks_compare, patch_features, sample_patches, wilson_interval,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def ridged(h: int, w: int, period: float, rng: np.random.Generator, noise: float = 0.05):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 + 0.4 * np.sin(xx * (2 * np.pi / period))
    base = base + rng.normal(0, noise, base.shape)
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


# -- KS ---------------------------------------------------------------------------

def test_ks_is_zero_against_itself():
    sample = np.arange(100, dtype=float)
    statistic, p = ks_compare(sample, sample)
    assert statistic == 0.0
    assert p == pytest.approx(1.0)


def test_ks_approaches_one_for_disjoint_samples():
    statistic, p = ks_compare(np.zeros(200), np.ones(200))
    assert statistic == pytest.approx(1.0)
    assert p < 1e-6


def test_ks_is_small_for_samples_from_the_same_distribution(rng):
    statistic, p = ks_compare(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert statistic < 0.1
    assert p > 0.01


def test_ks_ranks_closer_distributions_lower(rng):
    """The comparison the thesis actually makes: which arm is closer to real."""
    real = rng.normal(50, 10, 2000)
    close = rng.normal(52, 10, 2000)
    far = rng.normal(75, 10, 2000)
    assert ks_compare(close, real)[0] < ks_compare(far, real)[0]


def test_ks_rejects_empty_input():
    with pytest.raises(ValueError):
        ks_compare(np.array([]), np.ones(5))


# -- features ---------------------------------------------------------------------

def test_features_are_finite_and_fixed_length(rng):
    a = patch_features(ridged(256, 256, 9.0, rng))
    b = patch_features(np.zeros((256, 256), np.uint8))
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_features_separate_different_ridge_periods(rng):
    """Ridge scale is the thing the spectral bands exist to capture."""
    fine = patch_features(ridged(256, 256, 5.0, rng))
    coarse = patch_features(ridged(256, 256, 18.0, rng))
    assert not np.allclose(fine, coarse)


def test_features_are_invariant_to_dtype_scaling(rng):
    img = ridged(256, 256, 9.0, rng)
    np.testing.assert_allclose(
        patch_features(img), patch_features(img.astype(np.float32) / 255.0), atol=1e-6
    )


# -- patch sampling ---------------------------------------------------------------

def test_patches_are_the_requested_size(rng):
    patches = sample_patches(ridged(400, 275, 9.0, rng), 6, rng, size=128)
    assert len(patches) == 6
    assert all(p.shape == (128, 128) for p in patches)


def test_small_images_are_upscaled_not_rejected(rng):
    patches = sample_patches(ridged(100, 90, 9.0, rng), 3, rng, size=128)
    assert len(patches) == 3
    assert all(p.shape == (128, 128) for p in patches)


def test_patches_avoid_blank_background(rng):
    """Half ridges, half blank: sampling should prefer the half with content."""
    image = np.full((400, 400), 255, np.uint8)
    image[:200] = ridged(200, 400, 9.0, rng)
    patches = sample_patches(image, 8, rng, size=64)
    assert all(np.asarray(p, dtype=np.float32).std() > 1.0 for p in patches)


# -- the classifier ---------------------------------------------------------------

N_EVAL = 300  # per class; below ~20 samples per feature the estimate is biased downwards


def test_classifier_detects_separation_when_it_exists(rng):
    """The load-bearing test. If this fails, a near-chance result proves nothing."""
    generated = np.array([patch_features(ridged(256, 256, 5.0, rng)) for _ in range(N_EVAL)])
    real = np.array([patch_features(ridged(256, 256, 18.0, rng)) for _ in range(N_EVAL)])
    accuracy, low, _ = classifier_accuracy(generated, real, seed=1)
    assert accuracy > 0.9, f"harness cannot see an obvious difference (acc {accuracy:.2f})"
    assert low > 0.5


def test_classifier_reports_near_chance_for_identical_distributions(rng):
    both = [patch_features(ridged(256, 256, 9.0, rng)) for _ in range(2 * N_EVAL)]
    accuracy, low, high = classifier_accuracy(
        np.array(both[:N_EVAL]), np.array(both[N_EVAL:]), seed=2
    )
    assert 0.35 < accuracy < 0.65, f"chance-level data scored {accuracy:.2f}"
    assert low <= 0.5 <= high, f"interval [{low:.3f}, {high:.3f}] excludes chance"


def test_classifier_refuses_too_few_samples(rng):
    """A spuriously low accuracy is the flattering direction, so it must not be returned."""
    few = np.array([patch_features(ridged(256, 256, 9.0, rng)) for _ in range(30)])
    with pytest.raises(ValueError, match="too few"):
        classifier_accuracy(few[:15], few[15:])


def test_classifier_is_deterministic(rng):
    generated = np.array([patch_features(ridged(256, 256, 6.0, rng)) for _ in range(N_EVAL)])
    real = np.array([patch_features(ridged(256, 256, 12.0, rng)) for _ in range(N_EVAL)])
    first = classifier_accuracy(generated, real, seed=3)
    second = classifier_accuracy(generated, real, seed=3)
    assert first == second


def test_classifier_rejects_an_empty_class(rng):
    features = np.array([patch_features(ridged(256, 256, 9.0, rng)) for _ in range(5)])
    with pytest.raises(ValueError):
        classifier_accuracy(features, np.empty((0, features.shape[1])))


# -- interval -----------------------------------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high


def test_wilson_interval_stays_in_range_at_the_extremes():
    for successes in (0, 100):
        low, high = wilson_interval(successes, 100)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_narrows_with_more_data():
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
