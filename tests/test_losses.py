"""Tests for the WAFEN losses (plan U6).

Two properties carry thesis arguments and are checked numerically rather than by eye:
the ridge loss must genuinely punish false ridges harder than missed ones, and the
orientation loss must be blind to the pi wrap. If either is wrong the model still trains
and still produces numbers -- it just optimises for the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fpe.models.losses import (
    LossWeights, WafenLoss, masked_l1, orientation_loss, segmentation_loss, tversky_loss,
)
from fpe.models.wafen import Wafen, WafenConfig


def blob(value: float, size: int = 16) -> torch.Tensor:
    return torch.full((1, 1, size, size), value)


# -- Tversky: the precision argument ------------------------------------------------

def test_perfect_prediction_scores_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    assert tversky_loss(target.clone(), target).item() == pytest.approx(0.0, abs=1e-4)


def test_fully_inverted_prediction_scores_one():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    assert tversky_loss(1 - target, target).item() == pytest.approx(1.0, abs=1e-3)


def _equal_area_errors():
    """A target plus two predictions that are wrong by the same area, in opposite ways."""
    target = torch.zeros(1, 1, 20, 20)
    target[..., 4:12, 4:12] = 1.0          # 64 ridge pixels
    over = target.clone()
    over[..., 12:14, 4:12] = 1.0           # +16 pixels of invented ridge
    under = target.clone()
    under[..., 10:12, 4:12] = 0.0          # -16 pixels of missed ridge
    return target, over, under


def test_alpha_decides_which_error_is_worse():
    """The whole reason for alpha = 0.7: a spurious ridge becomes a spurious minutia,
    which is a false identity feature, while a missing one is merely absent.

    Note the two errors are equal in *area* but not in true-positive count -- inventing
    ridge leaves TP intact, erasing it does not -- so at alpha = 0.5 the missed ridge is
    already the costlier error. alpha = 0.7 must be strong enough to reverse that.
    """
    target, over, under = _equal_area_errors()

    symmetric = (tversky_loss(over, target, alpha=0.5).item(),
                 tversky_loss(under, target, alpha=0.5).item())
    assert symmetric[0] < symmetric[1], "baseline: missed should cost more at alpha=0.5"

    asymmetric = (tversky_loss(over, target, alpha=0.7).item(),
                  tversky_loss(under, target, alpha=0.7).item())
    assert asymmetric[0] > asymmetric[1], (
        f"alpha=0.7 failed to reverse the preference: invented {asymmetric[0]:.4f} "
        f"vs missed {asymmetric[1]:.4f}"
    )


def test_alpha_half_is_symmetric_under_swapping_prediction_and_target():
    """At alpha = 0.5 Tversky reduces to Dice, which is swap-invariant. Above 0.5 it is
    not -- that asymmetry is the entire mechanism."""
    target, over, _ = _equal_area_errors()
    assert tversky_loss(over, target, alpha=0.5).item() == pytest.approx(
        tversky_loss(target, over, alpha=0.5).item(), abs=1e-5
    )
    assert tversky_loss(over, target, alpha=0.7).item() != pytest.approx(
        tversky_loss(target, over, alpha=0.7).item(), abs=1e-3
    )


def test_tversky_ignores_everything_outside_the_mask():
    target = torch.zeros(1, 1, 10, 10)
    target[..., :5, :] = 1.0
    mask = torch.zeros(1, 1, 10, 10)
    mask[..., :5, :] = 1.0

    clean = tversky_loss(target.clone(), target, mask)
    polluted = target.clone()
    polluted[..., 5:, :] = 1.0             # nonsense, but entirely outside the mask
    assert tversky_loss(polluted, target, mask).item() == pytest.approx(
        clean.item(), abs=1e-5
    )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_outside_the_open_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        tversky_loss(blob(0.5), blob(1.0), alpha=alpha)


# -- orientation: the wrap ------------------------------------------------------------

def encode(angle: float, size: int = 8) -> torch.Tensor:
    return torch.cat([
        torch.full((1, 1, size, size), float(np.cos(2 * angle))),
        torch.full((1, 1, size, size), float(np.sin(2 * angle))),
    ], dim=1)


def test_orientation_loss_is_zero_when_correct():
    assert orientation_loss(encode(0.6), torch.full((1, 1, 8, 8), 0.6)).item() == \
        pytest.approx(0.0, abs=1e-5)


def test_orientation_loss_is_zero_when_off_by_pi():
    """theta and theta + pi are the same ridge and must cost nothing."""
    loss = orientation_loss(encode(0.6 + np.pi), torch.full((1, 1, 8, 8), 0.6))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_orientation_loss_is_maximal_at_a_right_angle():
    loss = orientation_loss(encode(0.6 + np.pi / 2), torch.full((1, 1, 8, 8), 0.6))
    assert loss.item() == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("offset", [0.1, 0.4, 0.8])
def test_orientation_loss_grows_with_angular_error(offset):
    target = torch.full((1, 1, 8, 8), 0.3)
    small = orientation_loss(encode(0.3 + offset / 2), target).item()
    large = orientation_loss(encode(0.3 + offset), target).item()
    assert large > small


def test_orientation_loss_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="2 channels"):
        orientation_loss(blob(1.0), blob(0.0))


# -- masked L1 --------------------------------------------------------------------------

def test_masked_l1_is_zero_on_an_exact_match():
    x = torch.rand(1, 1, 8, 8)
    assert masked_l1(x, x.clone()).item() == pytest.approx(0.0, abs=1e-7)


def test_masked_l1_ignores_masked_out_regions():
    prediction = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    target[..., 4:, :] = 99.0              # wild disagreement, all outside the mask
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0
    assert masked_l1(prediction, target, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_masked_l1_survives_an_all_zero_mask():
    value = masked_l1(blob(1.0), blob(0.0), torch.zeros(1, 1, 16, 16))
    assert torch.isfinite(value)


# -- segmentation -------------------------------------------------------------------

def test_segmentation_loss_is_near_zero_when_confident_and_right():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1.0
    prediction = target * 0.999 + (1 - target) * 0.001
    assert segmentation_loss(prediction, target).item() < 0.01


def test_segmentation_loss_is_finite_at_saturation():
    target = torch.ones(1, 1, 8, 8)
    assert torch.isfinite(segmentation_loss(torch.zeros(1, 1, 8, 8), target))


# -- degenerate inputs -----------------------------------------------------------------

@pytest.mark.parametrize("prediction,target", [
    (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0),
])
def test_tversky_is_finite_for_degenerate_uniform_inputs(prediction, target):
    value = tversky_loss(blob(prediction), blob(target))
    assert torch.isfinite(value) and 0.0 <= value.item() <= 1.0 + 1e-6


def test_tversky_is_finite_with_an_empty_mask():
    assert torch.isfinite(tversky_loss(blob(1.0), blob(1.0), torch.zeros(1, 1, 16, 16)))


# -- the combined objective ------------------------------------------------------------

def make_targets(size: int = 32) -> dict[str, torch.Tensor]:
    mask = torch.zeros(1, 1, size, size)
    mask[..., 4:-4, 4:-4] = 1.0
    ridge = torch.zeros(1, 1, size, size)
    ridge[..., ::3] = 1.0
    return {
        "mask": mask, "ridge": ridge,
        "orientation": torch.full((1, 1, size, size), 0.3),
        "period": torch.full((1, 1, size, size), 9.0),
        "evidence": torch.full((1, 1, size, size), 0.7),
    }


def test_combined_loss_reports_every_head():
    net = Wafen(WafenConfig(base_channels=4, depth=3))
    total, parts = WafenLoss()(net(torch.randn(1, 1, 32, 32)), make_targets())
    assert set(parts) == {"ridge", "confidence", "orientation", "period", "segmentation"}
    assert torch.isfinite(total)
    assert all(np.isfinite(v) for v in parts.values())


def test_combined_loss_backpropagates_to_every_parameter():
    net = Wafen(WafenConfig(base_channels=4, depth=3))
    total, _ = WafenLoss()(net(torch.randn(1, 1, 32, 32)), make_targets())
    total.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"


def test_zero_weight_removes_a_head_from_the_total():
    net = Wafen(WafenConfig(base_channels=4, depth=3))
    output = net(torch.randn(1, 1, 32, 32))
    targets = make_targets()
    full, parts = WafenLoss()(output, targets)
    without = WafenLoss(LossWeights(ridge=0.0))(output, targets)[0]
    assert full.item() - without.item() == pytest.approx(parts["ridge"], abs=1e-4)
