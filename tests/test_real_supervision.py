"""Leakage guard and indexing for real-print distillation targets."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpe.data.real import HELD_OUT, TRAIN_DATASETS, index_real

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "docs" / "datasets" / "manifests"


@pytest.mark.parametrize("held_out", sorted(HELD_OUT))
def test_held_out_datasets_are_refused(held_out):
    """A test image as a training input is exactly what rule 4 forbids."""
    with pytest.raises(ValueError, match="held-out"):
        index_real(MANIFESTS, ROOT / "data" / "raw", ("fvc2000", held_out))


def test_training_datasets_never_include_held_out_ones():
    assert not HELD_OUT.intersection(TRAIN_DATASETS)


def test_no_finger_crosses_the_train_val_split():
    items = index_real(MANIFESTS, ROOT / "data" / "raw")
    split_of: dict[str, str] = {}
    for sample, _ in items:
        assert split_of.setdefault(sample.finger_id, sample.split) == sample.split
    assert {s.split for s, _ in items} == {"train", "val"}


def test_same_finger_number_in_different_databases_is_a_different_finger():
    """FVC reuses 101..110 in every DB; db1_b/101 and db2_b/101 are different people."""
    items = index_real(MANIFESTS, ROOT / "data" / "raw", ("fvc2000",))
    ids = {s.finger_id for s, _ in items}
    assert len(ids) == 40  # 4 DBs x 10 fingers
    assert all(s.image.suffix == ".tif" for s, _ in items)


def test_declared_dpi_is_carried_per_image():
    items = index_real(MANIFESTS, ROOT / "data" / "raw", ("fvc2002",))
    assert {dpi for _, dpi in items} == {500, 569}
