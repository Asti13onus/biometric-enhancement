"""Manifest identity resolution (U12 prep)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_identity_columns_outrank_filename_parsing():
    """MINEX: a001_02 / a001_03 are different fingers; a001_02 / b001_02 the same."""
    from fpe.eval.protocol import build_pairs, parse_manifest

    imps = parse_manifest(ROOT / "docs/datasets/manifests/minex.csv")
    assert len(imps) == 801
    by_path = {Path(i.relpath).name: i for i in imps}
    a2, a3, b2 = by_path["a001_02.gray"], by_path["a001_03.gray"], by_path["b001_02.gray"]
    assert a2.finger_id != a3.finger_id           # same subject, different finger
    assert a2.finger_id == b2.finger_id           # same finger, different encounter
    pairs = build_pairs(imps, max_impostor=1000, seed=0)
    assert pairs.genuine, "MINEX must yield genuine pairs across encounters"
    assert all(x.finger_id == y.finger_id for x, y in pairs.genuine)
    assert all(x.finger_id != y.finger_id for x, y in pairs.impostor)


def test_filename_identity_still_rules_when_columns_are_empty():
    from fpe.eval.protocol import parse_manifest

    imps = parse_manifest(ROOT / "docs/datasets/manifests/fvc2004.csv")
    by_path = {i.relpath: i for i in imps}
    assert by_path["db1_b/101_1.tif"].finger_id == by_path["db1_b/101_2.tif"].finger_id
    assert by_path["db1_b/101_1.tif"].finger_id != by_path["db2_b/101_1.tif"].finger_id
