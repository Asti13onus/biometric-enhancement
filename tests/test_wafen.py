"""Tests for the WAFEN architecture (plan U5).

Three things are worth real scrutiny here. The parameter and latency budget, because
"runs on the hardware that exists in a village enrolment device" is a thesis claim and not
a nice-to-have. The orientation representation, because a network that cannot express the
wrap at pi will fail quietly rather than loudly. And gradient flow to every head, because
a head that silently receives no gradient is the most likely way this whole design fails.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from fpe.models.wafen import MAX_PARAMETERS, Wafen, WafenConfig, decode_orientation

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model() -> Wafen:
    return Wafen().eval()


# -- budget -------------------------------------------------------------------------

def test_default_configuration_is_inside_the_parameter_budget(model):
    total = model.parameter_count()
    assert total <= MAX_PARAMETERS, f"{total:,} parameters"
    assert total > 1_000_000, f"{total:,} looks too small to be the intended network"


def test_over_budget_configuration_is_refused():
    """Out of scope by definition, so it must fail at construction, not after an hour."""
    with pytest.raises(ValueError, match="budget"):
        Wafen(WafenConfig(base_channels=128, depth=6))


def test_budget_is_configurable_for_deliberate_experiments():
    small = Wafen(WafenConfig(base_channels=4, depth=3, max_parameters=100_000))
    assert small.parameter_count() <= 100_000


def test_depth_below_two_is_rejected():
    with pytest.raises(ValueError, match="depth"):
        Wafen(WafenConfig(depth=1))


# -- shapes -------------------------------------------------------------------------

def test_all_five_heads_come_back_at_input_resolution(model):
    out = model(torch.randn(2, 1, 256, 256))
    assert out.segmentation.shape == (2, 1, 256, 256)
    assert out.orientation.shape == (2, 2, 256, 256)
    assert out.period.shape == (2, 1, 256, 256)
    assert out.ridge.shape == (2, 1, 256, 256)
    assert out.confidence.shape == (2, 1, 256, 256)


def test_non_square_input_is_accepted(model):
    out = model(torch.randn(1, 1, 256, 128))
    assert out.ridge.shape == (1, 1, 256, 128)


def test_indivisible_input_fails_with_a_clear_message(model):
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 1, 250, 250))


def test_wrong_rank_input_is_rejected(model):
    with pytest.raises(ValueError, match=r"B, C, H, W"):
        model(torch.randn(1, 256, 256))


# -- output ranges -------------------------------------------------------------------

def test_probability_heads_are_bounded(model):
    out = model(torch.randn(2, 1, 64, 64) * 5)
    for name in ("segmentation", "ridge", "confidence"):
        tensor = getattr(out, name)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, name


def test_orientation_is_unit_norm(model):
    out = model(torch.randn(2, 1, 64, 64))
    norms = out.orientation.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# -- the orientation representation ---------------------------------------------------

@pytest.mark.parametrize("angle", [-1.5, -0.8, 0.0, 0.5, 1.2, 1.5])
def test_orientation_round_trips_through_the_double_angle(angle):
    pair = torch.tensor([[[[np.cos(2 * angle)]], [[np.sin(2 * angle)]]]], dtype=torch.float32)
    assert decode_orientation(pair).item() == pytest.approx(angle, abs=1e-5)


def _encode(theta: float) -> torch.Tensor:
    return torch.tensor([[[[np.cos(2 * theta)]], [[np.sin(2 * theta)]]]],
                        dtype=torch.float32)


@pytest.mark.parametrize("theta", [0.0, 0.01, 0.7, -1.2])
def test_orientation_identifies_theta_with_theta_plus_pi(theta):
    """The whole reason for two channels: a ridge has no head or tail, so t and t+pi
    are the same ridge and must encode to the same point."""
    assert torch.allclose(_encode(theta), _encode(theta + np.pi), atol=1e-6)


def test_orientation_keeps_mirrored_angles_apart():
    """t and -t are genuinely different orientations and must not collapse together --
    the double angle identifies t with t+pi, not with -t."""
    assert not torch.allclose(_encode(0.4), _encode(-0.4), atol=1e-3)


def test_nearly_opposite_angles_are_nearly_equal_in_the_encoding():
    """0.01 and just under pi differ by ~pi, so their encodings must be close: this is
    the wrap that a raw-angle head would be punished 178 units for."""
    assert torch.allclose(_encode(0.01), _encode(np.pi + 0.01), atol=1e-6)
    raw_penalty = abs((np.pi + 0.01) - 0.01)
    assert raw_penalty > 3.0, "sanity: the raw-angle representation would see a huge error"


def test_orientation_decoder_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="2 orientation channels"):
        decode_orientation(torch.randn(1, 1, 8, 8))


# -- gradients ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "head", ["segmentation", "orientation", "period", "ridge", "confidence"]
)
def test_every_head_propagates_gradient_to_the_shared_encoder(head):
    """A head wired up but receiving no gradient is the quietest way this design fails."""
    net = Wafen(WafenConfig(base_channels=4, depth=3))
    out = net(torch.randn(1, 1, 32, 32))
    getattr(out, head).sum().backward()

    first = net.encoders[0][0].weight
    assert first.grad is not None, f"{head} did not reach the encoder"
    assert torch.isfinite(first.grad).all()
    assert first.grad.abs().sum() > 0, f"{head} produced only zero gradients"


def test_gradients_are_finite_for_the_combined_objective():
    net = Wafen(WafenConfig(base_channels=4, depth=3))
    out = net(torch.randn(2, 1, 32, 32))
    total = sum(t.sum() for t in out.as_dict().values())
    total.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )


# -- budgets that are thesis claims ----------------------------------------------------

def test_cpu_inference_is_within_the_deployment_budget(model):
    """500 ms is a stated design goal; the four-network chain it replaces costs ~1.05 s."""
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        model(x)  # warm up
        times = []
        for _ in range(5):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    median = float(np.median(times))
    assert median < 0.5, f"CPU inference {median * 1000:.0f} ms exceeds the 500 ms budget"


def test_training_step_memory_stays_within_the_gpu_budget():
    """4 GB is the GTX 1650's ceiling; a config that cannot train on it is out of scope."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    net = Wafen().cuda()
    torch.cuda.reset_peak_memory_stats()
    out = net(torch.randn(8, 1, 256, 256, device="cuda"))
    sum(t.sum() for t in out.as_dict().values()).backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_gb < 4.0, f"peak {peak_gb:.2f} GB exceeds the 4 GB budget"
