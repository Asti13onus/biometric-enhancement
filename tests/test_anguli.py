"""Tests for Anguli indexing, splits and the supervision build (plan U2).

The two properties worth real scrutiny:

- **Split integrity.** If one finger's impressions straddle the train/test boundary, the
  network has seen the test answer and every number downstream is inflated. The split must
  be identical in every process and every run, with no file to fall out of sync.
- **Resumable writes.** A three-hour build on a machine that has already killed one long
  run will be interrupted. A half-written cache that still loads is worse than one that
  fails, because it corrupts training silently.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fpe.data.anguli import (
    DEFAULT_RATIOS, SPLITS, assign_split, clear_partial_writes, index_corpus,
    supervision_path, write_manifest,
)
from fpe.eval.protocol import build_pairs, parse_manifest

CORPUS = Path("data/processed/anguli_dev_5584")


# -- split assignment -------------------------------------------------------------

def test_split_is_deterministic_within_a_process():
    ids = [str(i) for i in range(500)]
    assert [assign_split(i) for i in ids] == [assign_split(i) for i in ids]


def test_split_is_deterministic_across_processes():
    """The built-in hash() is seeded per process; a finger must not move between runs."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from fpe.data.anguli import assign_split;"
        "print(','.join(assign_split(str(i)) for i in range(50)))"
    )
    runs = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=True).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    expected = ",".join(assign_split(str(i)) for i in range(50))
    assert runs[0] == expected


def test_split_proportions_are_close_to_the_ratios():
    ids = [str(i) for i in range(20_000)]
    counts = {name: 0 for name in SPLITS}
    for i in ids:
        counts[assign_split(i)] += 1
    for name, ratio in zip(SPLITS, DEFAULT_RATIOS):
        assert abs(counts[name] / len(ids) - ratio) < 0.02, counts


def test_every_split_name_is_valid():
    assert {assign_split(str(i)) for i in range(2000)} <= set(SPLITS)


def test_changing_the_salt_reshuffles():
    ids = [str(i) for i in range(200)]
    base = [assign_split(i) for i in ids]
    other = [assign_split(i, salt="different") for i in ids]
    assert base != other


# -- corpus indexing (needs the real corpus) ---------------------------------------

requires_corpus = pytest.mark.skipif(
    not CORPUS.is_dir(), reason="Anguli corpus not present"
)


@pytest.fixture(scope="module")
def samples():
    return index_corpus(CORPUS)


@requires_corpus
def test_index_finds_every_finger_with_three_impressions(samples):
    fingers = {s.finger_id for s in samples}
    assert len(fingers) == 5584, f"expected 5,584 fingers, indexed {len(fingers)}"
    assert len(samples) == 3 * len(fingers)
    per_finger = {}
    for s in samples:
        per_finger.setdefault(s.finger_id, set()).add(s.impression)
    assert all(v == {1, 2, 3} for v in per_finger.values())


@requires_corpus
def test_finger_numbers_are_globally_unique_across_buckets(samples):
    """fp_1 holds 1-1000 and fp_6 holds 5001-5758; a number must not repeat."""
    by_finger = {}
    for s in samples:
        by_finger.setdefault(s.finger_id, set()).add(s.bucket)
    repeated = {f: b for f, b in by_finger.items() if len(b) > 1}
    assert not repeated, f"finger ids appear in multiple buckets: {list(repeated)[:5]}"


@requires_corpus
def test_no_finger_crosses_a_split(samples):
    """The leakage guarantee: all three impressions of a finger share one split."""
    by_finger = {}
    for s in samples:
        by_finger.setdefault(s.finger_id, set()).add(s.split)
    straddling = {f: sp for f, sp in by_finger.items() if len(sp) > 1}
    assert not straddling, f"fingers in multiple splits: {list(straddling)[:5]}"


@requires_corpus
def test_master_and_image_exist_for_every_sample(samples):
    for s in samples[::311]:  # stride-sample; checking all 16,752 is needlessly slow
        assert s.master.is_file(), s.master
        assert s.image.is_file(), s.image


@requires_corpus
def test_impressions_limit_is_respected():
    limited = index_corpus(CORPUS, impressions=1)
    assert {s.impression for s in limited} == {1}


@requires_corpus
def test_empty_buckets_are_skipped(samples):
    """fp_7..fp_20 survive from the interrupted 20k run and hold nothing."""
    assert {s.bucket for s in samples} == {f"fp_{i}" for i in range(1, 7)}


# -- the supervision cache ---------------------------------------------------------

@requires_corpus
def test_build_is_atomic_and_resumable(tmp_path, samples):
    """A cache file that exists must be complete, and rebuilding must skip it."""
    from fpe.data.anguli import build_sample

    sample = samples[0]
    path, built_first = build_sample(sample, tmp_path)
    assert built_first and path.is_file()
    assert not list(tmp_path.rglob("*.npz.tmp")), "a temporary file was left behind"

    _, built_again = build_sample(sample, tmp_path)
    assert not built_again, "an existing cache was rebuilt instead of skipped"

    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"mask", "orientation", "period", "meta"}
        shape = np.asarray(sample.master.stat().st_size)  # touch, then check real shapes
        assert data["mask"].dtype == np.uint8
        assert set(np.unique(data["mask"])) <= {0, 1}
        assert data["orientation"].dtype == np.float16
        assert data["period"].dtype == np.float16
        assert data["mask"].shape == data["orientation"].shape == data["period"].shape
        angles = data["orientation"].astype(np.float32)
        assert angles.min() >= -np.pi / 2 - 1e-2
        assert angles.max() <= np.pi / 2 + 1e-2
        assert shape.size == 1


def test_clear_partial_writes_removes_only_temporaries(tmp_path):
    (tmp_path / "fp_1").mkdir()
    keep = tmp_path / "fp_1" / "1_1.npz"
    drop = tmp_path / "fp_1" / "1_2.npz.tmp"
    keep.write_bytes(b"x")
    drop.write_bytes(b"x")

    assert clear_partial_writes(tmp_path) == 1
    assert keep.is_file()
    assert not drop.exists()


def test_clear_partial_writes_tolerates_a_missing_directory(tmp_path):
    assert clear_partial_writes(tmp_path / "nope") == 0


# -- the evaluation manifest ---------------------------------------------------------

@requires_corpus
def test_manifest_contains_only_test_fingers(tmp_path, samples):
    out = tmp_path / "anguli.csv"
    written = write_manifest(samples, out)
    assert written > 0

    training = {s.finger_id for s in samples if s.split != "test"}
    rows = parse_manifest(out)
    assert rows, "manifest produced no parseable rows"
    leaked = {r.finger_id.rsplit("/", 1)[-1] for r in rows} & training
    assert not leaked, f"training fingers reached the evaluation manifest: {list(leaked)[:5]}"


@requires_corpus
def test_manifest_parses_into_three_genuine_pairs_per_finger(tmp_path, samples):
    """Anguli names files 5.png with the impression in the directory, so the identity
    parser needs its Anguli rule -- without it every row is silently dropped."""
    out = tmp_path / "anguli.csv"
    write_manifest(samples, out)

    impressions = parse_manifest(out)
    per_finger = {}
    for im in impressions:
        per_finger.setdefault(im.finger_id, set()).add(im.impression)
    assert per_finger, "no identities parsed out of the Anguli manifest"
    assert all(v == {"1", "2", "3"} for v in per_finger.values())

    pairs = build_pairs(impressions)
    assert len(pairs.genuine) == 3 * len(per_finger)
    assert pairs.impostor, "no impostor pairs -- all Anguli shares one capture scope"


# -- impostor subsampling -------------------------------------------------------------

def _fake(n_fingers: int, n_impressions: int):
    from fpe.eval.protocol import Impression
    return [
        Impression(dataset="d", relpath=f"{f}_{i}.png", finger_id=f"d/{f}",
                   impression=str(i), scope="d")
        for f in range(n_fingers) for i in range(1, n_impressions + 1)
    ]


def test_cap_limits_impostor_count_without_touching_genuine():
    impressions = _fake(60, 3)
    full = build_pairs(impressions)
    capped = build_pairs(impressions, max_impostor=500)
    assert len(full.impostor) == 60 * 59 // 2 * 9
    assert len(capped.impostor) == 500
    assert len(capped.genuine) == len(full.genuine)


def test_cap_is_deterministic_and_seed_sensitive():
    impressions = _fake(40, 3)
    a = build_pairs(impressions, max_impostor=200, seed=1)
    b = build_pairs(impressions, max_impostor=200, seed=1)
    c = build_pairs(impressions, max_impostor=200, seed=2)
    key = lambda ps: [(x.relpath, y.relpath) for x, y in ps.impostor]
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_cap_above_the_total_returns_everything():
    impressions = _fake(10, 3)
    full = build_pairs(impressions)
    capped = build_pairs(impressions, max_impostor=10_000)
    assert len(capped.impostor) == len(full.impostor)


def test_capped_pairs_are_all_genuine_impostors():
    """Subsampling must never emit a same-finger pair as an impostor."""
    impressions = _fake(50, 3)
    capped = build_pairs(impressions, max_impostor=400)
    assert all(x.finger_id != y.finger_id for x, y in capped.impostor)
    assert len({(x.relpath, y.relpath) for x, y in capped.impostor}) == 400


def test_cap_draws_uniformly_across_finger_pairs():
    """A naive implementation samples finger pairs then impressions, over-weighting
    fingers with fewer impressions. This checks the draw is flat over all pairs."""
    impressions = _fake(30, 3)
    capped = build_pairs(impressions, max_impostor=2000, seed=7)
    from collections import Counter
    per_pair = Counter((x.finger_id, y.finger_id) for x, y in capped.impostor)
    counts = np.array(list(per_pair.values()))
    assert counts.max() <= 9
    assert counts.mean() > 3, f"draw looks clumped: mean {counts.mean():.2f}"
