"""Tests for the wear degradation model (plan U1).

The properties that matter most here are reproducibility and severity monotonicity.
Every downstream claim in the thesis rests on the evidence map being an honest account of
what the model destroyed, so these tests check the *contract*, not the pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fpe.degradation.wear import WearConfig, WearDegradation


RIDGE_PERIOD = 9.0


@pytest.fixture
def synthetic_print() -> np.ndarray:
    """A curved sinusoidal ridge pattern -- enough structure for orientation estimation."""
    h, w = 200, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (xx + 0.004 * (yy - h / 2) ** 2) * (2 * np.pi / RIDGE_PERIOD)
    ridges = 0.5 + 0.45 * np.sin(phase)
    return (ridges * 255).astype(np.uint8)


def only(effect: str) -> WearConfig:
    """Config with exactly one mechanism enabled."""
    flags = {
        "enable_attenuation": False,
        "enable_creases": False,
        "enable_fragmentation": False,
        "enable_contact": False,
    }
    flags[f"enable_{effect}"] = True
    return WearConfig(**flags)


EFFECTS = ["attenuation", "creases", "fragmentation", "contact"]


# -- reproducibility --------------------------------------------------------------

def test_same_seed_reproduces_exactly(synthetic_print):
    wdm = WearDegradation()
    a = wdm(synthetic_print, seed=7, severity=0.6)
    b = wdm(synthetic_print, seed=7, severity=0.6)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.evidence, b.evidence)
    assert a.record.to_dict() == b.record.to_dict()


def test_different_seeds_differ(synthetic_print):
    wdm = WearDegradation()
    a = wdm(synthetic_print, seed=1, severity=0.6)
    b = wdm(synthetic_print, seed=2, severity=0.6)
    assert not np.array_equal(a.image, b.image)


# -- the identity guarantee -------------------------------------------------------

def test_severity_zero_is_the_identity(synthetic_print):
    result = WearDegradation()(synthetic_print, seed=3, severity=0.0)
    expected = synthetic_print.astype(np.float32) / 255.0
    np.testing.assert_allclose(result.image, expected, atol=1e-6)
    assert np.all(result.evidence == 1.0)
    assert result.record.applied == []


# -- each mechanism in isolation ---------------------------------------------------

@pytest.mark.parametrize("effect", EFFECTS)
def test_each_effect_alone_changes_image_and_evidence(synthetic_print, effect):
    wdm = WearDegradation(only(effect))
    result = wdm(synthetic_print, seed=11, severity=1.0)
    clean = synthetic_print.astype(np.float32) / 255.0
    assert not np.allclose(result.image, clean), f"{effect} left the image untouched"
    assert result.evidence.min() < 1.0, f"{effect} destroyed no evidence"
    assert len(result.record.applied) == 1


@pytest.mark.parametrize("effect", EFFECTS)
def test_disabled_effect_is_inert(synthetic_print, effect):
    flags = {f"enable_{e}": (e != effect) for e in EFFECTS}
    result = WearDegradation(WearConfig(**flags))(synthetic_print, seed=11, severity=1.0)
    assert all(not name.startswith(effect[:6]) for name in result.record.applied)


# -- evidence map contract ---------------------------------------------------------

@pytest.mark.parametrize("severity", [round(0.1 * i, 1) for i in range(11)])
def test_evidence_map_shape_dtype_and_range(synthetic_print, severity):
    result = WearDegradation()(synthetic_print, seed=5, severity=severity)
    assert result.evidence.shape == synthetic_print.shape
    assert result.evidence.dtype == np.float32
    assert result.evidence.min() >= 0.0 and result.evidence.max() <= 1.0
    assert result.image.dtype == np.float32
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0


def test_severity_monotonicity_per_seed(synthetic_print):
    """Raising severity must never restore evidence -- for the same seed, exactly."""
    wdm = WearDegradation()
    for seed in range(4):
        means = [
            float(wdm(synthetic_print, seed=seed, severity=s).evidence.mean())
            for s in np.linspace(0.0, 1.0, 11)
        ]
        diffs = np.diff(means)
        assert (diffs <= 1e-6).all(), f"seed {seed}: evidence rose with severity {means}"
        assert means[-1] < means[0], f"seed {seed}: severity 1.0 destroyed nothing"


# -- mechanism-specific structure ---------------------------------------------------

def test_creases_are_elongated_not_speckle(synthetic_print):
    """A crease is a fold across the print, not scattered pixels."""
    result = WearDegradation(only("creases"))(synthetic_print, seed=2, severity=1.0)
    damaged = (result.evidence < 0.5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(damaged, connectivity=8)
    assert count > 1, "no crease was drawn"

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    width = stats[largest, cv2.CC_STAT_WIDTH]
    height = stats[largest, cv2.CC_STAT_HEIGHT]
    area = stats[largest, cv2.CC_STAT_AREA]

    # Elongation: the component spans much of the image in at least one direction,
    # while filling only a small part of its own bounding box.
    span = max(width / synthetic_print.shape[1], height / synthetic_print.shape[0])
    fill = area / max(width * height, 1)
    assert span > 0.5, f"crease spans only {span:.2f} of the image"
    assert fill < 0.5, f"component fills {fill:.2f} of its bbox -- a blob, not a crease"


def test_fragmentation_is_anisotropic(synthetic_print):
    """Dry-skin breaks run along the ridge; isotropic blobs would be wrong.

    The ridges here are near-vertical, so break components should be taller than wide.
    """
    cfg = WearConfig(
        enable_attenuation=False, enable_creases=False, enable_contact=False,
        enable_fragmentation=True,
    )
    result = WearDegradation(cfg)(synthetic_print, seed=4, severity=0.5)
    damaged = (result.evidence < 0.5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(damaged, connectivity=8)
    assert count > 2, "fragmentation produced almost nothing to measure"

    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    big = stats[1:, cv2.CC_STAT_AREA] >= 8  # ignore single-pixel speckle
    assert big.sum() >= 5, "too few sizeable break components to judge shape"
    aspect = np.median(heights[big] / np.maximum(widths[big], 1.0))
    assert aspect > 1.3, f"breaks are not elongated along the ridge (aspect {aspect:.2f})"


def test_contact_loss_is_strongest_at_the_periphery(synthetic_print):
    result = WearDegradation(only("contact"))(synthetic_print, seed=6, severity=1.0)
    h, w = synthetic_print.shape
    centre = result.evidence[h // 3:2 * h // 3, w // 3:2 * w // 3].mean()
    border = np.concatenate([
        result.evidence[:8].ravel(), result.evidence[-8:].ravel(),
        result.evidence[:, :8].ravel(), result.evidence[:, -8:].ravel(),
    ]).mean()
    assert centre > border, "contact loss should spare the core, not the edges"


# -- behaviour on real data ---------------------------------------------------------

def test_runs_on_a_real_print():
    """Synthetic test images can hide assumptions; run on an actual FVC image."""
    from fpe.data.convert import load_greyscale

    path = "data/raw/fvc2004/db1_b/101_1.tif"
    try:
        image = load_greyscale(path)
    except (OSError, FileNotFoundError):
        pytest.skip(f"{path} not present")

    result = WearDegradation()(image, seed=9, severity=0.7)
    assert result.image.shape == image.shape
    assert result.evidence.shape == image.shape
    assert result.image.dtype == np.float32
    assert 0.0 <= result.evidence.min() <= result.evidence.max() <= 1.0
    assert result.evidence.mean() < 1.0


# -- input validation ----------------------------------------------------------------

@pytest.mark.parametrize("severity", [-0.1, 1.1])
def test_severity_out_of_range_is_rejected(synthetic_print, severity):
    with pytest.raises(ValueError, match="severity"):
        WearDegradation()(synthetic_print, seed=0, severity=severity)


def test_colour_input_is_rejected(synthetic_print):
    colour = np.stack([synthetic_print] * 3, axis=-1)
    with pytest.raises(ValueError, match="greyscale"):
        WearDegradation()(colour, seed=0, severity=0.5)


# -- foreground mask ------------------------------------------------------------------

def test_mask_leaves_background_untouched(synthetic_print):
    """Creases and dry skin exist only where there is skin.

    Without this, the evidence map claims destroyed evidence over blank background and
    the confidence head learns to distrust empty paper.
    """
    h, w = synthetic_print.shape
    mask = np.zeros((h, w), np.uint8)
    mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
    background = mask == 0

    result = WearDegradation()(synthetic_print, seed=8, severity=1.0, mask=mask)
    clean = synthetic_print.astype(np.float32) / 255.0

    np.testing.assert_allclose(result.image[background], clean[background], atol=1e-6)
    assert np.all(result.evidence[background] == 1.0)
    assert result.evidence[mask > 0].min() < 1.0, "no damage inside the mask"


def test_mask_shape_is_validated(synthetic_print):
    with pytest.raises(ValueError, match="mask shape"):
        WearDegradation()(synthetic_print, seed=0, severity=0.5,
                          mask=np.ones((5, 5), np.uint8))


def test_no_mask_matches_an_all_ones_mask(synthetic_print):
    wdm = WearDegradation()
    without = wdm(synthetic_print, seed=12, severity=0.8)
    with_full = wdm(synthetic_print, seed=12, severity=0.8,
                    mask=np.ones(synthetic_print.shape, np.uint8))
    np.testing.assert_allclose(without.image, with_full.image, atol=1e-6)
    np.testing.assert_allclose(without.evidence, with_full.evidence, atol=1e-6)
