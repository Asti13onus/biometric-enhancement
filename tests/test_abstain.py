"""Tests for abstention (plan U8), covering acceptance example AE1.

The behaviour being protected is narrow and easy to get subtly wrong: declining must
actually remove minutiae from the template that reaches the matcher, the two extreme
thresholds must behave sensibly rather than crashing, and coverage must be measured over
the foreground so a tight crop cannot claim high coverage for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpe.eval.abstain import (
    AbstentionConfig, apply_coverage, coverage_fraction, filter_minutiae, write_xyt,
)
from fpe.metrics.nbis import Minutia, read_xyt

SIZE = 64


@pytest.fixture
def ridge() -> np.ndarray:
    field = np.zeros((SIZE, SIZE), np.float32)
    field[:, ::3] = 1.0
    return field


@pytest.fixture
def confidence() -> np.ndarray:
    """High confidence on the left half, destroyed on the right."""
    field = np.full((SIZE, SIZE), 0.9, np.float32)
    field[:, SIZE // 2:] = 0.1
    return field


def grid_minutiae(step: int = 8) -> list[Minutia]:
    return [Minutia(x=x, y=y, theta=0, quality=50)
            for y in range(4, SIZE, step) for x in range(4, SIZE, step)]


# -- AE1: minutiae do not survive in destroyed regions ------------------------------

def test_no_minutia_survives_in_a_declined_region(ridge, confidence):
    """Covers AE1."""
    _, coverage = apply_coverage(ridge, confidence)
    kept, dropped = filter_minutiae(grid_minutiae(), coverage)
    assert dropped > 0, "nothing was dropped from the destroyed half"
    assert all(m.x < SIZE // 2 for m in kept), "a minutia survived in the declined half"
    assert len(kept) + dropped == len(grid_minutiae())


def test_declined_pixels_become_background_not_invented_ridges(ridge, confidence):
    image, coverage = apply_coverage(ridge, confidence)
    assert np.all(image[~coverage] == 1.0), "declined pixels are not background"
    np.testing.assert_allclose(image[coverage], ridge[coverage])


# -- threshold extremes ---------------------------------------------------------------

def test_threshold_zero_declines_nothing(ridge, confidence):
    image, coverage = apply_coverage(ridge, confidence, AbstentionConfig(threshold=0.0))
    assert coverage.all()
    np.testing.assert_allclose(image, ridge)
    _, dropped = filter_minutiae(grid_minutiae(), coverage)
    assert dropped == 0


def test_threshold_above_one_declines_everything_without_crashing(ridge, confidence):
    """An empty template is a failure to acquire, which is a result, not an exception."""
    image, coverage = apply_coverage(ridge, confidence, AbstentionConfig(threshold=1.01))
    assert not coverage.any()
    assert np.all(image == 1.0)
    kept, dropped = filter_minutiae(grid_minutiae(), coverage)
    assert kept == []
    assert dropped == len(grid_minutiae())


def test_coverage_decreases_monotonically_as_the_threshold_rises(ridge):
    confidence = np.tile(np.linspace(0, 1, SIZE, dtype=np.float32), (SIZE, 1))
    fractions = [
        coverage_fraction(apply_coverage(ridge, confidence,
                                         AbstentionConfig(threshold=t))[1])
        for t in np.linspace(0, 1, 11)
    ]
    assert all(b <= a + 1e-9 for a, b in zip(fractions, fractions[1:])), fractions
    assert fractions[0] > fractions[-1]


# -- coverage accounting ----------------------------------------------------------------

def test_coverage_is_measured_over_the_foreground(ridge, confidence):
    """Otherwise a tightly cropped image claims high coverage for free."""
    _, coverage = apply_coverage(ridge, confidence)
    foreground = np.zeros((SIZE, SIZE), bool)
    foreground[:, :SIZE // 2] = True          # the finger is only the left half

    assert coverage_fraction(coverage) == pytest.approx(0.5, abs=0.02)
    assert coverage_fraction(coverage, foreground) == pytest.approx(1.0, abs=1e-6)


def test_empty_foreground_reports_zero_rather_than_dividing_by_zero(ridge, confidence):
    _, coverage = apply_coverage(ridge, confidence)
    assert coverage_fraction(coverage, np.zeros((SIZE, SIZE), bool)) == 0.0


# -- determinism and bounds ---------------------------------------------------------------

def test_a_minutia_exactly_on_the_threshold_is_kept(ridge):
    confidence = np.full((SIZE, SIZE), 0.5, np.float32)
    _, coverage = apply_coverage(ridge, confidence, AbstentionConfig(threshold=0.5))
    assert coverage.all(), "confidence equal to the threshold must count as covered"


def test_out_of_bounds_minutiae_are_dropped(ridge, confidence):
    _, coverage = apply_coverage(ridge, confidence)
    outside = [Minutia(x=-1, y=5, theta=0, quality=9),
               Minutia(x=5, y=SIZE + 3, theta=0, quality=9)]
    kept, dropped = filter_minutiae(outside, coverage)
    assert kept == [] and dropped == 2


def test_shape_disagreement_is_rejected(ridge):
    with pytest.raises(ValueError, match="confidence shape"):
        apply_coverage(ridge, np.zeros((8, 8), np.float32))


def test_dilation_shrinks_the_covered_region(ridge, confidence):
    plain = apply_coverage(ridge, confidence)[1]
    eroded = apply_coverage(ridge, confidence, AbstentionConfig(dilate_px=2))[1]
    assert eroded.sum() < plain.sum()
    assert not (eroded & ~plain).any(), "erosion added coverage somewhere"


# -- round trip through the template format --------------------------------------------

def test_filtered_minutiae_round_trip_through_xyt(tmp_path, ridge, confidence):
    _, coverage = apply_coverage(ridge, confidence)
    kept, _ = filter_minutiae(grid_minutiae(), coverage)
    path = write_xyt(kept, tmp_path / "t.xyt")
    reloaded = read_xyt(path)
    assert len(reloaded) == len(kept)
    assert [(m.x, m.y, m.theta, m.quality) for m in reloaded] == \
           [(m.x, m.y, m.theta, m.quality) for m in kept]


def test_empty_template_writes_a_readable_file(tmp_path):
    path = write_xyt([], tmp_path / "empty.xyt")
    assert path.is_file()
    assert read_xyt(path) == []


def test_xyt_filter_reads_y_from_the_bottom_as_mindtct_writes_it(tmp_path):
    """mindtct's default .xyt is bottom-origin; the keep mask is top-origin.

    Measured on FVC2004: read bottom-origin, 100% of minutiae land on the finger; read
    top-origin, 90.6%. Getting this wrong drops real minutiae and keeps invented ones.
    """
    from fpe.eval.abstain import filter_xyt_file

    keep = np.ones((100, 50), dtype=bool)
    keep[:20] = False                      # decline the top 20 rows of the image
    xyt = write_xyt([Minutia(10, 90, 0, 50),   # y=90 from bottom -> row 9: declined
                     Minutia(10, 5, 0, 50)],   # y=5 from bottom -> row 94: kept
                    tmp_path / "t.xyt")
    assert filter_xyt_file(xyt, keep) == 1
    assert [m.y for m in read_xyt(xyt)] == [5]


def test_block_confidence_is_the_foreground_mean_of_each_block():
    """Background must not dilute a block, and every pixel in a block gets one value."""
    from fpe.eval.abstain import block_confidence

    conf = np.zeros((32, 32), np.float32)
    conf[:16, :16] = 0.8
    conf[0, 0] = 0.0                              # one speck: must not decline the block
    fg = np.ones((32, 32), bool)
    fg[16:, 16:] = False                          # background block
    out = block_confidence(conf, fg, 16)
    assert out.shape == conf.shape
    assert np.allclose(out[:16, :16], (0.8 * 255) / 256)
    assert np.allclose(out[16:, 16:], 0.0)
    assert len(np.unique(out[:16, :16])) == 1


def test_block_confidence_handles_sides_not_divisible_by_the_block():
    from fpe.eval.abstain import block_confidence

    conf = np.full((20, 37), 0.5, np.float32)
    out = block_confidence(conf, np.ones_like(conf, bool), 16)
    assert out.shape == (20, 37) and np.allclose(out, 0.5)
