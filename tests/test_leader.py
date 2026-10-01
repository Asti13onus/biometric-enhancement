"""LEADER as the second extraction chain (U11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpe.metrics.nbis import read_xyt, run_mindtct
from fpe.models.leader import LeaderExtractor

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "raw" / "fvc2004" / "db1_b" / "101_2.tif"

needs_data = pytest.mark.skipif(not CLEAN.is_file(), reason="FVC2004 not present")


@needs_data
def test_leader_template_is_parseable_and_in_xyt_conventions(tmp_path):
    xyt = LeaderExtractor(tmp_path)(CLEAN)
    minutiae = read_xyt(xyt)
    assert len(minutiae) > 5
    for m in minutiae:
        assert 0 <= m.theta <= 359
        assert 0 <= m.quality <= 100
        assert m.x >= 0 and m.y >= 0


@needs_data
def test_leader_and_mindtct_agree_on_scale(tmp_path):
    """Both chains produce a plausible count on the same clean image, and both run
    end to end -- the point of a second chain is comparability, not identity."""
    from fpe.data.convert import to_nbis_input

    leader = read_xyt(LeaderExtractor(tmp_path / "l")(CLEAN))
    jpl = to_nbis_input(CLEAN, tmp_path / "m")
    mind = read_xyt(run_mindtct(jpl, tmp_path / "m" / "t"))
    assert 10 <= len(leader) <= 200
    assert 10 <= len(mind) <= 200
    # same image, same bottom-origin convention: centroids should land in the same
    # half of the image, which catches a forgotten y-flip outright
    cy_l = sum(m.y for m in leader) / len(leader)
    cy_m = sum(m.y for m in mind) / len(mind)
    assert abs(cy_l - cy_m) < 120


@needs_data
def test_leader_template_is_cached(tmp_path):
    e = LeaderExtractor(tmp_path)
    first = e(CLEAN)
    stamp = first.stat().st_mtime_ns
    assert e(CLEAN).stat().st_mtime_ns == stamp
