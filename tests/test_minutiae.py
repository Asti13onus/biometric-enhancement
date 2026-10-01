"""Cappelli-protocol minutiae correspondence (plan U9) — the spec's scenarios verbatim."""

from __future__ import annotations

import math

import pytest

from fpe.metrics.minutiae import (TAU_D, TAU_THETA, TypedMinutia, angular_difference_deg,
                                  match_minutiae, parse_min_file)


def m(x, y, theta=0.0, kind=None):
    if kind is None:
        return TypedMinutia(x, y, theta, 50, "RIG")
    return TypedMinutia(x, y, theta, 50, kind)


GRID = [m(20 * i, 20 * j, theta=(37 * (i + j)) % 360) for i in range(5) for j in range(5)]


def test_a_template_matched_against_itself_is_perfect():
    c = match_minutiae(GRID, GRID)
    assert c.precision == 1.0 and c.recall == 1.0 and c.f1 == 1.0


def test_the_14_pixel_distance_boundary():
    truth = [m(100, 100)]
    assert match_minutiae([m(113, 100)], truth).matched == 1   # 13 px: corresponds
    assert match_minutiae([m(115, 100)], truth).matched == 0   # 15 px: does not


def test_the_direction_boundary_at_pi_over_9():
    truth = [m(100, 100, theta=0.0)]
    under = math.degrees(TAU_THETA) - 0.5
    over = math.degrees(TAU_THETA) + 0.5
    assert match_minutiae([m(100, 100, theta=under)], truth).matched == 1
    assert match_minutiae([m(100, 100, theta=over)], truth).matched == 0


def test_direction_difference_wraps_the_full_circle():
    assert angular_difference_deg(5.0, 355.0) == pytest.approx(10.0)
    assert match_minutiae([m(0, 0, theta=355.0)], [m(0, 0, theta=5.0)]).matched == 1


def test_type_exact_and_type_agnostic_differ_only_on_types():
    truth = [m(0, 0, kind="RIG"), m(50, 50, kind="BIF")]
    detected = [m(0, 0, kind="BIF"), m(50, 50, kind="RIG")]  # positions right, types swapped
    assert match_minutiae(detected, truth).matched == 2
    assert match_minutiae(detected, truth, type_exact=True).matched == 0


def test_type_exact_refuses_untyped_minutiae():
    class Bare:
        x, y, theta = 0, 0, 0.0

    with pytest.raises(ValueError, match="kind"):
        match_minutiae([Bare()], [m(0, 0)], type_exact=True)


def test_assignment_is_one_to_one_and_prefers_the_closer_pair():
    truth = [m(100, 100)]
    detected = [m(101, 100), m(104, 100)]  # both admissible; only one may claim it
    c = match_minutiae(detected, truth)
    assert c.matched == 1 and c.precision == 0.5 and c.recall == 1.0
    # and the closer detection wins: adding a second truth at the far detection's spot
    c2 = match_minutiae(detected, [m(104, 100), m(101, 100)])
    assert c2.matched == 2


def test_an_empty_template_yields_zero_without_dividing_by_zero():
    c = match_minutiae([], GRID)
    assert c.precision == 0.0 and c.recall == 0.0 and c.f1 == 0.0
    c = match_minutiae(GRID, [])
    assert c.precision == 0.0 and c.recall == 0.0


def test_min_file_round_trip(tmp_path):
    text = (
        "Image (w,h) 640 480\n\n2 Minutiae Detected\n\n"
        "   0 :  213,  101 :  8 :  0.124 :BIF : APP :  3 :  220,  57;  3\n"
        "   1 :  220,   57 :  5 :  0.118 :RIG : APP :  0 :  248,  30;  1\n"
    )
    p = tmp_path / "t.min"
    p.write_text(text, encoding="ascii")
    ms = parse_min_file(p)
    assert [mm.kind for mm in ms] == ["BIF", "RIG"]
    assert ms[0].x == 213 and ms[0].y == 101
    assert ms[0].theta == pytest.approx(8 * 11.25)
    assert ms[0].quality == 12
