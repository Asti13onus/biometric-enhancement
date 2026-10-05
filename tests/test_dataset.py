"""Tests for the training dataset (plan U4).

The load-bearing test is the rotation convention. Resampling an orientation map without
transforming the *values* produces targets with the right shape, the right range and
perfect spatial alignment that are nonetheless wrong everywhere -- the model would train
without complaint and simply never learn orientation. So the convention is checked against
a structure tensor recomputed on the rotated image, not against this module's own
arithmetic, which would agree with itself whatever sign it used.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fpe.data.anguli import AnguliSample
from fpe.data.dataset import AugmentConfig, WafenDataset, rotate_orientation
from fpe.degradation.wear import WearConfig, _ridge_orientation

SIZE = 256
PERIOD = 9.0


def striped(angle_deg: float = 0.0, size: int = SIZE) -> np.ndarray:
    """Straight ridges at a known orientation."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = np.deg2rad(angle_deg)
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    return ((0.5 + 0.45 * np.sin(projection * (2 * np.pi / PERIOD))) * 255).astype(np.uint8)


@pytest.fixture
def corpus(tmp_path):
    """A one-sample corpus on disk: impression, master, and cached targets."""
    impression_dir = tmp_path / "Impression_1" / "fp_1"
    master_dir = tmp_path / "Fingerprints" / "fp_1"
    supervision = tmp_path / "supervision"
    for d in (impression_dir, master_dir, supervision / "fp_1"):
        d.mkdir(parents=True, exist_ok=True)

    image = striped(0.0)
    Image.fromarray(image).save(impression_dir / "1.png")
    master = (image > 128).astype(np.uint8) * 255      # binary, ridges dark
    Image.fromarray(master).save(master_dir / "1.png")

    orientation = _ridge_orientation(image.astype(np.float32) / 255.0, PERIOD)
    orientation = ((orientation + np.pi / 2) % np.pi - np.pi / 2).astype(np.float32)
    mask = np.ones_like(image, dtype=np.uint8)
    with (supervision / "fp_1" / "1_1.npz").open("wb") as fh:
        np.savez_compressed(
            fh, mask=mask, orientation=orientation.astype(np.float16),
            period=np.full(image.shape, PERIOD, np.float16), meta=np.array("{}"),
        )

    sample = AnguliSample(
        finger_id="1", bucket="fp_1", impression=1,
        master=master_dir / "1.png", image=impression_dir / "1.png", split="train",
    )
    return [sample], supervision


def fixed_augment(**overrides) -> AugmentConfig:
    """Augmentation with every random knob pinned, so one effect can be isolated."""
    base = dict(rotation_deg=(0.0, 0.0), scale=(1.0, 1.0), translate_frac=0.0,
                p_hflip=0.0, gamma=(1.0, 1.0), severity=(0.0, 0.0))
    base.update(overrides)
    return AugmentConfig(**base)


def centre(array: np.ndarray, margin: int = 64) -> np.ndarray:
    return array[margin:-margin, margin:-margin]


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Difference between two orientation fields, wrapped to [-pi/2, pi/2)."""
    return (a - b + np.pi / 2) % np.pi - np.pi / 2


# -- the wrap helper -----------------------------------------------------------------

@pytest.mark.parametrize("angle", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_rotate_orientation_wraps_into_the_half_circle(angle):
    out = rotate_orientation(np.array([angle], np.float32), 0.0)
    assert -np.pi / 2 <= out[0] < np.pi / 2


def test_rotate_orientation_is_periodic_in_pi():
    angles = np.linspace(-1.5, 1.5, 21, dtype=np.float32)
    np.testing.assert_allclose(
        rotate_orientation(angles, 0.0), rotate_orientation(angles, np.pi), atol=1e-5
    )


# -- the convention, verified against the image itself --------------------------------

@pytest.mark.parametrize("rotation", [-15.0, -7.0, 7.0, 15.0])
def test_rotating_the_image_rotates_the_orientation_target(corpus, rotation):
    """The single easiest thing to get wrong, and it fails silently when wrong."""
    samples, supervision = corpus
    dataset = WafenDataset(samples, supervision, augment=True,
                           config=fixed_augment(rotation_deg=(rotation, rotation)))
    image, targets = dataset[0]

    produced = image[0].numpy()
    stored = targets["orientation"][0].numpy()
    # Re-estimate orientation from the image the dataset actually returned.
    measured = _ridge_orientation(produced, PERIOD)
    measured = ((measured + np.pi / 2) % np.pi - np.pi / 2).astype(np.float32)

    difference = angular_difference(centre(stored), centre(measured))
    median = float(np.median(np.abs(difference)))
    assert median < np.deg2rad(5), (
        f"orientation target disagrees with the returned image by "
        f"{np.rad2deg(median):.1f} degrees at {rotation} degrees of rotation"
    )


def test_horizontal_flip_mirrors_the_orientation_target(corpus):
    samples, supervision = corpus
    dataset = WafenDataset(samples, supervision, augment=True,
                           config=fixed_augment(p_hflip=1.0))
    image, targets = dataset[0]

    measured = _ridge_orientation(image[0].numpy(), PERIOD)
    measured = ((measured + np.pi / 2) % np.pi - np.pi / 2).astype(np.float32)
    difference = angular_difference(centre(targets["orientation"][0].numpy()),
                                    centre(measured))
    assert float(np.median(np.abs(difference))) < np.deg2rad(5)


def test_unaugmented_orientation_matches_the_cached_target(corpus):
    """Baseline: with no augmentation the target must survive untouched."""
    samples, supervision = corpus
    dataset = WafenDataset(samples, supervision, augment=False)
    image, targets = dataset[0]
    measured = _ridge_orientation(image[0].numpy(), PERIOD)
    measured = ((measured + np.pi / 2) % np.pi - np.pi / 2).astype(np.float32)
    difference = angular_difference(centre(targets["orientation"][0].numpy()),
                                    centre(measured))
    assert float(np.median(np.abs(difference))) < np.deg2rad(5)


# -- spatial alignment -----------------------------------------------------------------

def test_ridge_target_stays_aligned_with_the_image_under_rotation(corpus):
    """Alignment is separate from the orientation values and can break on its own."""
    samples, supervision = corpus
    dataset = WafenDataset(samples, supervision, augment=True,
                           config=fixed_augment(rotation_deg=(12.0, 12.0)))
    image, targets = dataset[0]
    inked = centre(image[0].numpy()) < 0.5
    ridge = centre(targets["ridge"][0].numpy()) > 0.5
    overlap = float((inked & ridge).sum()) / max(float(ridge.sum()), 1.0)
    assert overlap > 0.7, f"ridge target overlaps the dark pixels only {overlap:.2f}"


# -- shapes, types and ranges ------------------------------------------------------------

def test_item_shapes_and_dtypes(corpus):
    samples, supervision = corpus
    image, targets = WafenDataset(samples, supervision)[0]
    assert image.shape == (1, SIZE, SIZE) and image.dtype == torch.float32
    assert set(targets) == {"mask", "ridge", "orientation", "period", "period_valid",
                            "evidence", "minutiae"}
    for name, tensor in targets.items():
        assert tensor.shape == (1, SIZE, SIZE), name
        assert tensor.dtype == torch.float32, name


def test_value_ranges_are_sane(corpus):
    samples, supervision = corpus
    image, targets = WafenDataset(samples, supervision)[0]
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    for name in ("mask", "ridge", "evidence"):
        assert 0.0 <= float(targets[name].min()) <= float(targets[name].max()) <= 1.0, name
    angles = targets["orientation"]
    assert float(angles.min()) >= -np.pi / 2 - 1e-3
    assert float(angles.max()) <= np.pi / 2 + 1e-3


def test_batches_collate(corpus):
    from torch.utils.data import DataLoader

    samples, supervision = corpus
    loader = DataLoader(WafenDataset(samples * 4, supervision), batch_size=4, num_workers=0)
    image, targets = next(iter(loader))
    assert image.shape == (4, 1, SIZE, SIZE)
    assert targets["ridge"].shape == (4, 1, SIZE, SIZE)


# -- determinism ---------------------------------------------------------------------------

def test_same_seed_reproduces_the_same_item(corpus):
    samples, supervision = corpus
    a = WafenDataset(samples, supervision, seed=5)[0]
    b = WafenDataset(samples, supervision, seed=5)[0]
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1]["evidence"], b[1]["evidence"])


def test_changing_the_epoch_changes_the_degradation(corpus):
    """Fresh damage each epoch is the reason degradation is not cached."""
    samples, supervision = corpus
    dataset = WafenDataset(samples, supervision, seed=5)
    first = dataset[0][0].clone()
    dataset.set_epoch(1)
    assert not torch.equal(first, dataset[0][0])


def test_different_seeds_differ(corpus):
    samples, supervision = corpus
    assert not torch.equal(WafenDataset(samples, supervision, seed=1)[0][0],
                           WafenDataset(samples, supervision, seed=2)[0][0])


# -- failure modes ---------------------------------------------------------------------------

def test_missing_cache_raises_rather_than_yielding_zeros(corpus, tmp_path):
    """Silently returning an empty target would train the model on nothing."""
    samples, _ = corpus
    dataset = WafenDataset(samples, tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="build_supervision"):
        dataset[0]


def test_empty_sample_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        WafenDataset([], tmp_path)


def test_shape_disagreement_is_reported(corpus):
    """A cache built at a different size must fail loudly, not broadcast silently."""
    samples, supervision = corpus
    path = supervision / "fp_1" / "1_1.npz"
    with path.open("wb") as fh:
        np.savez_compressed(
            fh, mask=np.ones((64, 64), np.uint8),
            orientation=np.zeros((64, 64), np.float16),
            period=np.full((64, 64), 9.0, np.float16), meta=np.array("{}"),
        )
    with pytest.raises(ValueError, match="shapes disagree"):
        WafenDataset(samples, supervision)[0]


# -- the period validity marker -------------------------------------------------------

def test_invalid_periods_are_marked_not_clamped(corpus, tmp_path):
    """The frequency teacher emits negative periods where it could not estimate. A
    negative ridge spacing is a marker, not a target -- clamping would invent a number
    and train the network to reproduce it."""
    samples, supervision = corpus
    path = supervision / "fp_1" / "1_1.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    period = np.full_like(arrays["period"], 9.0)
    period[:64] = -7.0                      # the teacher's "no estimate" marker
    arrays["period"] = period
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)

    _, targets = WafenDataset(samples, supervision, augment=False)[0]
    valid = targets["period_valid"][0].numpy()
    returned = targets["period"][0].numpy()

    assert valid.min() == 0.0 and valid.max() == 1.0, "no invalid region was marked"
    assert (returned[valid == 0] < 0).all(), "invalid periods were altered, not marked"
    assert np.allclose(returned[valid == 1], 9.0)


def test_period_loss_ignores_invalid_pixels():
    """The marker has to reach the objective, not just the batch.

    Excluding pixels changes *which* pixels are averaged, so the loss value legitimately
    shifts. The property that matters is narrower: whatever garbage sits in the invalid
    region must not influence the result at all.
    """
    import torch
    from fpe.models.losses import WafenLoss
    from fpe.models.wafen import Wafen, WafenConfig

    net = Wafen(WafenConfig(base_channels=4, depth=3))
    output = net(torch.randn(1, 1, 32, 32))

    def loss_with(invalid_value: float) -> float:
        period = torch.full((1, 1, 32, 32), 9.0)
        period[..., :16, :] = invalid_value
        valid = torch.ones(1, 1, 32, 32)
        valid[..., :16, :] = 0.0
        return WafenLoss()(output, {
            "mask": torch.ones(1, 1, 32, 32),
            "ridge": torch.zeros(1, 1, 32, 32),
            "orientation": torch.zeros(1, 1, 32, 32),
            "evidence": torch.zeros(1, 1, 32, 32),
            "period": period, "period_valid": valid,
        })[1]["period"]

    assert loss_with(-999.0) == pytest.approx(loss_with(12345.0), abs=1e-5)


def test_period_loss_falls_back_to_the_mask_without_a_validity_target():
    """Callers that do not supply period_valid must still get a finite loss."""
    import torch
    from fpe.models.losses import WafenLoss
    from fpe.models.wafen import Wafen, WafenConfig

    net = Wafen(WafenConfig(base_channels=4, depth=3))
    output = net(torch.randn(1, 1, 32, 32))
    total, parts = WafenLoss()(output, {
        "mask": torch.ones(1, 1, 32, 32),
        "ridge": torch.zeros(1, 1, 32, 32),
        "orientation": torch.zeros(1, 1, 32, 32),
        "evidence": torch.zeros(1, 1, 32, 32),
        "period": torch.full((1, 1, 32, 32), 9.0),
    })
    assert np.isfinite(parts["period"]) and torch.isfinite(total)


def test_real_print_ridge_target_comes_from_the_cache(corpus):
    """Real prints have no master; their SNFEN-teacher ridge map is cached (fpe.data.real)."""
    samples, supervision = corpus
    path = supervision / "fp_1" / "1_1.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    teacher = np.zeros_like(arrays["mask"], dtype=np.uint8)
    teacher[:, ::2] = 255                                  # 1 = "ridge here", as SNFEN renders
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays, ridge=teacher)

    real = AnguliSample(finger_id="1", bucket="fp_1", impression=1, master=None,
                        image=samples[0].image, split="train")
    _, targets = WafenDataset([real], supervision)._load(real)
    assert np.array_equal(targets["ridge"], teacher / 255.0)


# -- minutia weight map (the minutia-aware loss's input) -----------------------------

def test_unbroken_stripes_carry_no_minutia_weight():
    """Ridges running edge to edge end only at the border, which is excluded."""
    from fpe.data.dataset import minutia_weight_map

    ridge = np.zeros((96, 96), np.float32)
    ridge[:, ::8] = 1.0
    weights = minutia_weight_map(ridge, np.ones_like(ridge))
    assert weights.sum() == 0.0


def test_a_ridge_ending_gets_a_weight_disk_at_the_ending():
    from fpe.data.dataset import minutia_weight_map

    ridge = np.zeros((96, 96), np.float32)
    ridge[:, ::8] = 1.0
    ridge[40:, 48] = 0.0                      # one ridge stops at row 40
    weights = minutia_weight_map(ridge, np.ones_like(ridge))
    assert weights[40, 48] == 1.0
    assert weights[40, 16] == 0.0             # unbroken neighbours stay unweighted
    assert weights.mean() < 0.05              # the disk is local, not a blanket


def test_a_bifurcation_gets_a_weight_disk():
    from fpe.data.dataset import minutia_weight_map

    ridge = np.zeros((96, 96), np.float32)
    ridge[:, 48] = 1.0                        # vertical ridge
    ridge[48, 48:] = 1.0                      # branch heading right: a junction at (48,48)
    weights = minutia_weight_map(ridge, np.ones_like(ridge))
    assert weights[48, 48] == 1.0


def test_minutiae_near_the_mask_border_are_excluded():
    """An ending created by the segmentation cut is framing, not a feature."""
    from fpe.data.dataset import minutia_weight_map

    ridge = np.zeros((96, 96), np.float32)
    ridge[:, ::8] = 1.0
    mask = np.ones_like(ridge)
    mask[40:] = 0.0                           # the mask cuts every ridge at row 40
    ridge[40:] = 0.0
    weights = minutia_weight_map(ridge, mask)
    assert weights.sum() == 0.0


def test_loss_charges_extra_at_minutiae_and_only_when_the_map_is_present():
    from fpe.models.losses import WafenLoss
    from fpe.models.wafen import Wafen, WafenConfig

    torch.manual_seed(0)
    model = Wafen(WafenConfig(base_channels=4, depth=2, head_channels=4))
    image = torch.rand(1, 1, 32, 32)
    base = {k: torch.rand(1, 1, 32, 32) for k in ("mask", "ridge", "period",
                                                  "period_valid", "evidence")}
    base["mask"] = torch.ones(1, 1, 32, 32)
    base["orientation"] = torch.zeros(1, 1, 32, 32)
    out = model(image)

    _, parts = WafenLoss()(out, base)
    assert "minutiae" not in parts            # absent map, absent term

    with_map = dict(base, minutiae=torch.ones(1, 1, 32, 32))
    total_hi, parts_hi = WafenLoss()(out, with_map)
    no_map_total, _ = WafenLoss()(out, base)
    assert "minutiae" in parts_hi and parts_hi["minutiae"] > 0
    assert float(total_hi) > float(no_map_total)


# -- paired views for the consistency loss -------------------------------------------

def test_paired_views_share_targets_and_differ_only_in_damage(corpus):
    samples, supervision = corpus
    image, targets = WafenDataset(samples, supervision, pairs=True)[0]
    assert image.shape[0] == 2 and image.dim() == 4
    assert not torch.equal(image[0], image[1])          # different damage
    for name in ("mask", "ridge", "orientation", "minutiae"):
        assert torch.equal(targets[name][0], targets[name][1]), name  # same finger, same frame
    assert not torch.equal(targets["evidence"][0], targets["evidence"][1])


def test_consistency_is_zero_for_identical_views_and_positive_otherwise():
    from types import SimpleNamespace

    from fpe.train import _consistency

    ridge = torch.rand(2, 1, 16, 16)
    mask = torch.ones(4, 1, 16, 16)
    same = SimpleNamespace(ridge=torch.cat([ridge[:1], ridge[:1], ridge[1:], ridge[1:]]))
    assert float(_consistency(same, mask)) == 0.0
    different = SimpleNamespace(ridge=torch.cat([ridge[:1], 1 - ridge[:1],
                                                 ridge[1:], ridge[1:]]))
    assert float(_consistency(different, mask)) > 0.1


def test_paired_batches_flatten_to_adjacent_views(corpus):
    from torch.utils.data import DataLoader

    from fpe.train import _to_device

    samples, supervision = corpus
    dataset = WafenDataset(samples * 3, supervision, pairs=True)
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    image, targets, paired = _to_device(batch, torch.device("cpu"))
    assert paired and image.shape[0] == 6
    assert torch.equal(targets["ridge"][0], targets["ridge"][1])  # views of item 0 adjacent


# -- photometric mode (arm two: domain alignment) -------------------------------------

def _with_evidence(supervision):
    path = supervision / "fp_1" / "1_1.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.setdefault("ridge", (np.random.default_rng(0).random((SIZE, SIZE)) > 0.5)
                      .astype(np.uint8) * 255)
    arrays["evidence"] = np.full((SIZE, SIZE), 0.75, np.float16)
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)


def test_photometric_mode_requires_cached_evidence(corpus):
    samples, supervision = corpus
    with pytest.raises(ValueError, match="evidence"):
        WafenDataset(samples, supervision, photometric=True)[0]


def test_photometric_pairs_keep_the_print_undamaged_and_share_evidence(corpus):
    samples, supervision = corpus
    _with_evidence(supervision)
    dataset = WafenDataset(samples, supervision, photometric=True, pairs=True,
                           augment=False)
    image, targets = dataset[0]
    assert image.shape[0] == 2
    assert not torch.equal(image[0], image[1])            # jitter differs
    assert torch.equal(targets["evidence"][0], targets["evidence"][1])
    assert float(targets["evidence"].max()) <= 1.0
    # photometric jitter is monotone-ish in intensity: ridge structure survives; the
    # wear model's evidence destruction does not happen, so evidence stays the cache's
    assert float(targets["evidence"].mean()) == pytest.approx(0.75, abs=0.01)
