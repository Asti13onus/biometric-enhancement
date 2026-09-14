"""Tests for the training loop (plan U7).

The canary test is the one that matters: a network that cannot memorise ten samples has a
wiring bug, and every later result would be built on it. Everything else here guards
against losing a long run on a machine that has already killed one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from fpe.data.anguli import AnguliSample
from fpe.data.dataset import AugmentConfig, WafenDataset
from fpe.degradation.wear import _ridge_orientation
from fpe.models.wafen import WafenConfig
from fpe.train import TrainConfig, evaluate, overfit_check, train

SIZE = 64
PERIOD = 9.0
TINY = WafenConfig(base_channels=4, depth=3)


def striped(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    theta = rng.uniform(0, np.pi)
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    base = 0.5 + 0.45 * np.sin(projection * (2 * np.pi / PERIOD))
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


@pytest.fixture
def dataset(tmp_path):
    """A small on-disk corpus, big enough to train a toy model against."""
    supervision = tmp_path / "supervision"
    samples = []
    for index in range(1, 13):
        impression_dir = tmp_path / "Impression_1" / "fp_1"
        master_dir = tmp_path / "Fingerprints" / "fp_1"
        for d in (impression_dir, master_dir, supervision / "fp_1"):
            d.mkdir(parents=True, exist_ok=True)

        image = striped(index)
        Image.fromarray(image).save(impression_dir / f"{index}.png")
        Image.fromarray((image > 128).astype(np.uint8) * 255).save(
            master_dir / f"{index}.png")

        orientation = _ridge_orientation(image.astype(np.float32) / 255.0, PERIOD)
        with (supervision / "fp_1" / f"{index}_1.npz").open("wb") as fh:
            np.savez_compressed(
                fh, mask=np.ones_like(image, np.uint8),
                orientation=orientation.astype(np.float16),
                period=np.full(image.shape, PERIOD, np.float16), meta=np.array("{}"),
            )
        samples.append(AnguliSample(
            finger_id=str(index), bucket="fp_1", impression=1,
            master=master_dir / f"{index}.png", image=impression_dir / f"{index}.png",
            split="train",
        ))
    config = AugmentConfig(patch=SIZE, severity=(0.2, 0.6))
    return WafenDataset(samples, supervision, config=config, seed=0)


def quick(**overrides) -> TrainConfig:
    base = dict(epochs=1, batch_size=2, accumulate=1, num_workers=0, amp=False,
                device="cpu", max_steps_per_epoch=2)
    base.update(overrides)
    return TrainConfig(**base)


# -- the canary -------------------------------------------------------------------

def test_the_network_can_memorise_a_handful_of_samples(dataset):
    """If this fails, the wiring is broken and no amount of data will help."""
    before = overfit_check(dataset, samples=4, steps=1, device="cpu", model_config=TINY)
    after = overfit_check(dataset, samples=4, steps=120, device="cpu", model_config=TINY)
    assert after < before, f"ridge loss did not fall: {before:.4f} -> {after:.4f}"
    assert after < 0.35, f"could not over-fit four samples (ridge loss {after:.4f})"


# -- the loop --------------------------------------------------------------------

def test_training_runs_and_writes_a_checkpoint(dataset, tmp_path):
    out = tmp_path / "run"
    model, history = train(dataset, dataset, config=quick(), model_config=TINY,
                           out_dir=out, progress=lambda _m: None)
    assert (out / "checkpoint.pt").is_file()
    assert (out / "best.pt").is_file()
    assert (out / "history.json").is_file()
    assert len(history.epochs) == 1
    assert history.best_epoch == 0


def test_history_records_every_head_separately(dataset, tmp_path):
    """A head that stops learning is invisible in a summed loss."""
    out = tmp_path / "run"
    _, history = train(dataset, dataset, config=quick(), model_config=TINY,
                       out_dir=out, progress=lambda _m: None)
    record = history.epochs[0]
    for split in ("train", "val"):
        assert set(record[split]) == {
            "ridge", "confidence", "orientation", "period", "segmentation"
        }, split
        assert all(np.isfinite(v) for v in record[split].values()), split

    saved = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert saved["epochs"][0]["train"].keys() == record["train"].keys()


def test_training_resumes_from_a_checkpoint(dataset, tmp_path):
    """This machine has killed a long run before; resuming must continue, not restart."""
    out = tmp_path / "run"
    train(dataset, dataset, config=quick(epochs=1), model_config=TINY, out_dir=out,
          progress=lambda _m: None)

    messages: list[str] = []
    _, history = train(dataset, dataset, config=quick(epochs=2), model_config=TINY,
                       out_dir=out, progress=messages.append)
    assert any("resumed" in m for m in messages), messages
    assert [e["epoch"] for e in history.epochs] == [0, 1]


def test_resume_can_be_disabled(dataset, tmp_path):
    out = tmp_path / "run"
    train(dataset, dataset, config=quick(), model_config=TINY, out_dir=out,
          progress=lambda _m: None)
    messages: list[str] = []
    train(dataset, dataset, config=quick(), model_config=TINY, out_dir=out,
          resume=False, progress=messages.append)
    assert not any("resumed" in m for m in messages)


def test_best_checkpoint_tracks_the_validation_ridge_loss(dataset, tmp_path):
    out = tmp_path / "run"
    _, history = train(dataset, dataset, config=quick(epochs=3), model_config=TINY,
                       out_dir=out, progress=lambda _m: None)
    ridges = [e["val"]["ridge"] for e in history.epochs]
    assert history.best_val_ridge == pytest.approx(min(ridges), abs=1e-9)
    assert history.best_epoch == int(np.argmin(ridges))


def test_gradient_accumulation_runs(dataset, tmp_path):
    _, history = train(dataset, dataset, config=quick(accumulate=2, max_steps_per_epoch=4),
                       model_config=TINY, out_dir=tmp_path / "run",
                       progress=lambda _m: None)
    assert np.isfinite(history.epochs[0]["train"]["ridge"])


def test_evaluate_returns_empty_for_an_empty_loader(dataset):
    from torch.utils.data import DataLoader
    from fpe.models.losses import WafenLoss
    from fpe.models.wafen import Wafen

    empty = DataLoader([], batch_size=2)
    assert evaluate(Wafen(TINY), empty, WafenLoss(), torch.device("cpu")) == {}


# -- budgets ---------------------------------------------------------------------

def test_over_budget_model_fails_before_any_training_happens(dataset, tmp_path):
    """An hour of training that cannot be reported is worse than an immediate error."""
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="budget"):
        train(dataset, dataset, config=quick(),
              model_config=WafenConfig(base_channels=128, depth=6), out_dir=out,
              progress=lambda _m: None)
    assert not (out / "checkpoint.pt").exists()


def test_training_step_stays_within_the_gpu_budget(dataset):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from fpe.models.losses import WafenLoss
    from fpe.models.wafen import Wafen

    model = Wafen().cuda()
    criterion = WafenLoss()
    torch.cuda.reset_peak_memory_stats()
    image = torch.randn(8, 1, 256, 256, device="cuda")
    targets = {k: torch.rand(8, 1, 256, 256, device="cuda")
               for k in ("mask", "ridge", "orientation", "period", "period_valid",
                         "evidence")}
    total, _ = criterion(model(image), targets)
    total.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert peak < 4.0, f"peak {peak:.2f} GB exceeds the 4 GB budget"
