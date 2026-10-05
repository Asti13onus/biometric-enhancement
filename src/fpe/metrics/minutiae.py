"""Minutiae correspondence under Cappelli's protocol (plan U9).

A detected minutia corresponds to a ground-truth one when their Euclidean distance is at
most TAU_D = 14 pixels and their direction difference at most TAU_THETA = pi/9, with
greedy one-to-one assignment (closest pairs first), in the type-exact and type-agnostic
variants. Precision is the thesis's binding metric: a spurious minutia is a false
identity feature, so the figure that matters is precision against the coverage the
method was willing to reconstruct.

Ground truth here is *pseudo* ground truth — mindtct run on the clean synthetic master —
and every figure derived from it is labelled synthetic. The origin document forbids
presenting these as comparable to published SD27 numbers.

mindtct's `.xyt` template carries no minutia type, so type-exact matching reads the
`.min` file mindtct writes beside it (`parse_min_file`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["TAU_D", "TAU_THETA", "Correspondence", "match_minutiae",
           "parse_min_file", "angular_difference_deg"]

TAU_D = 14.0
"""Euclidean correspondence threshold, pixels (Cappelli 2026)."""
TAU_THETA = math.pi / 9
"""Direction threshold, radians (20 degrees)."""


@dataclass(frozen=True)
class Correspondence:
    matched: int
    n_detected: int
    n_truth: int

    @property
    def precision(self) -> float:
        """Fraction of detections that correspond to a true minutia. 0 when nothing was
        detected: an empty template asserts nothing, and asserting nothing truthfully is
        not the same as asserting everything correctly."""
        return self.matched / self.n_detected if self.n_detected else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.n_truth if self.n_truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}precision": self.precision, f"{prefix}recall": self.recall,
                f"{prefix}f1": self.f1, f"{prefix}matched": float(self.matched),
                f"{prefix}n_detected": float(self.n_detected),
                f"{prefix}n_truth": float(self.n_truth)}


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two minutia directions in degrees (full circle:
    a minutia direction has a head, unlike ridge orientation)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def match_minutiae(
    detected: Sequence, truth: Sequence, *,
    tau_d: float = TAU_D, tau_theta: float = TAU_THETA, type_exact: bool = False,
) -> Correspondence:
    """Greedy one-to-one correspondence: admissible pairs by increasing distance.

    Minutiae need `.x`, `.y` (pixels) and `.theta` (degrees); type-exact additionally
    compares `.kind` and raises if either side lacks it — silently degrading to
    type-agnostic would report the laxer number under the stricter name.
    """
    tau_theta_deg = math.degrees(tau_theta)
    pairs = []
    for i, d in enumerate(detected):
        for j, t in enumerate(truth):
            dist = math.hypot(d.x - t.x, d.y - t.y)
            if dist > tau_d:
                continue
            if angular_difference_deg(d.theta, t.theta) > tau_theta_deg:
                continue
            if type_exact:
                dk, tk = getattr(d, "kind", None), getattr(t, "kind", None)
                if dk is None or tk is None:
                    raise ValueError("type-exact matching needs `.kind` on every "
                                     "minutia; parse the .min file, not the .xyt")
                if dk != tk:
                    continue
            pairs.append((dist, i, j))

    pairs.sort(key=lambda p: p[0])
    used_d: set[int] = set()
    used_t: set[int] = set()
    matched = 0
    for _, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        matched += 1
    return Correspondence(matched, len(detected), len(truth))


@dataclass(frozen=True)
class TypedMinutia:
    x: int
    y: int
    theta: float  # degrees
    quality: int
    kind: str  # "RIG" (ridge ending) or "BIF" (bifurcation)


_MIN_LINE = re.compile(
    r"^\s*\d+\s*:\s*(?P<x>\d+),\s*(?P<y>\d+)\s*:"
    r"\s*(?P<theta>\d+)\s*:\s*(?P<q>[\d.]+)\s*:\s*(?P<kind>RIG|BIF)\b"
)


def parse_min_file(path: str | Path) -> list[TypedMinutia]:
    """Read mindtct's `.min` output, the only one of its outputs that carries type.

    mindtct's `.min` uses an 11.25-degree direction unit (converted to degrees here) and
    a **top-origin** y, unlike the bottom-origin `.xyt` (measured on one template:
    y_xyt = height - y_min exactly). Correspondence compares `.min` against `.min`, so
    no coordinate flip is needed anywhere in the U9 pipeline.
    """
    out: list[TypedMinutia] = []
    for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        m = _MIN_LINE.match(line)
        if not m:
            continue
        out.append(TypedMinutia(
            x=int(m["x"]), y=int(m["y"]),
            theta=(int(m["theta"]) * 11.25) % 360.0,
            quality=int(round(float(m["q"]) * 100)) if "." in m["q"] else int(m["q"]),
            kind=m["kind"],
        ))
    return out
