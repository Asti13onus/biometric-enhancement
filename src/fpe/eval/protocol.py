"""Comparison protocols: turning a dataset manifest into genuine and impostor pairs.

The manifests record what is on disk, not who is in the image -- identity for
most of our real sets lives only in the filename, and the conventions differ
per dataset:

    FVC "B"              db1_b/101_1.tif    finger 101, impression 1
    Neurotech CrossMatch 012_3_1.tif        subject 012, finger 3, impression 1
    Neurotech U.are.U    012_10_1.tif       subject 012, finger 10, impression 1
    FVS                  10_1.bmp           finger 10, impression 1
    MINEX                a001_02.gray       subject a001, image 02

A "finger" here is the identity unit: two images match genuinely if and only
if they are the same physical finger. For FVC that unit must be scoped by
database, because finger 101 of DB1 and finger 101 of DB2 are different
fingers captured on different sensors.

Impostor pairing defaults to `all` (every impression against every impression
of every other finger) rather than the official FVC rule (first impression
only). Our "B" subsets are small -- the official rule yields 45 impostor pairs
per database, which cannot support a stable error rate -- and we compare our
own before/after conditions rather than published FVC numbers, so the extra
statistical power is worth the documented deviation.
"""
from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

__all__ = ["Impression", "parse_manifest", "build_pairs", "PairSet"]

ImpostorMode = Literal["all", "first"]


@dataclass(frozen=True)
class Impression:
    """One image, with the identity resolved."""

    dataset: str
    relpath: str
    finger_id: str  # identity unit; genuine pairs share this
    impression: str
    scope: str = ""  # capture context, e.g. "db1_b"; impostors stay within it

    @property
    def path(self) -> Path:
        return Path(self.relpath)


@dataclass(frozen=True)
class PairSet:
    genuine: list[tuple[Impression, Impression]]
    impostor: list[tuple[Impression, Impression]]

    def __repr__(self) -> str:  # keeps experiment logs readable
        return f"PairSet(genuine={len(self.genuine)}, impostor={len(self.impostor)})"


# finger_impression:  101_1.tif, 10_1.bmp
_TWO_PART = re.compile(r"^(?P<finger>\w+?)_(?P<impr>\d+)$")
# subject_finger_impression:  012_3_1.tif
_THREE_PART = re.compile(r"^(?P<subject>\w+?)_(?P<finger>\d+)_(?P<impr>\d+)$")
# Anguli puts the impression index in the *directory*, not the filename:
#   Impression_2/fp_6/5758.png  ->  finger 5758, impression 2
_ANGULI_DIR = re.compile(r"^Impression_(?P<impr>\d+)$")


def _identity(dataset: str, relpath: str) -> tuple[str, str, str] | None:
    """(finger_id, impression, scope) for one manifest row, or None if unparseable."""
    p = Path(relpath)
    stem = p.stem

    # Anguli: the impression index lives in the top directory, and the three
    # impressions of one finger must resolve to the *same* finger id. All Anguli
    # images come from one generator, so they share a single capture scope.
    if len(p.parts) >= 3 and (m := _ANGULI_DIR.match(p.parts[0])):
        bucket = p.parts[-2]
        return f"{dataset}/{bucket}/{stem}", m["impr"], dataset

    # Scope by every directory above the file, so db1_b/101 != db2_b/101.
    scope = "/".join(p.parts[:-1])
    prefix = f"{dataset}/{scope}" if scope else dataset

    if m := _THREE_PART.match(stem):
        return f"{prefix}/{m['subject']}_{m['finger']}", m["impr"], prefix
    if m := _TWO_PART.match(stem):
        return f"{prefix}/{m['finger']}", m["impr"], prefix
    return None


def parse_manifest(
    manifest_path: str | Path, *, subset: str | None = None
) -> list[Impression]:
    """Read a manifest and resolve identity for every row it can.

    `subset` filters on a path prefix, e.g. "db1_b" for one FVC database.
    Rows whose filename does not parse are skipped; the caller sees the
    shortfall in the returned count.
    """
    path = Path(manifest_path)
    out: list[Impression] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            relpath = row["relpath"]
            if subset and not relpath.startswith(subset):
                continue
            ident = _identity(row["dataset"], relpath)
            if ident is None:
                continue
            finger_id, impr, scope = ident
            out.append(
                Impression(
                    dataset=row["dataset"],
                    relpath=relpath,
                    finger_id=finger_id,
                    impression=impr,
                    scope=scope,
                )
            )
    return out


def _by_finger(impressions: Iterable[Impression]) -> dict[str, list[Impression]]:
    groups: dict[str, list[Impression]] = {}
    for im in impressions:
        groups.setdefault(im.finger_id, []).append(im)
    for v in groups.values():
        v.sort(key=lambda i: (len(i.impression), i.impression))
    return groups


def _subsample_impostors(
    groups: dict[str, list[Impression]],
    finger_pairs: list[tuple[str, str]],
    limit: int,
    rng: np.random.Generator,
) -> list[tuple[Impression, Impression]]:
    """Draw `limit` impostor pairs uniformly without materialising all of them.

    A 606-finger test split with three impressions each yields 1.65M impostor pairs --
    enough to make a single evaluation take the better part of an hour, and the abstention
    sweep runs many evaluations. Subsampling does not bias the EER; it widens the interval
    slightly at very low false-match rates, which is a trade worth making to get the sweep
    down to minutes.

    Sampling walks a virtual index over the full cross-product, so the draw is uniform over
    every pair rather than uniform over finger pairs -- the latter would over-weight
    fingers with fewer impressions.
    """
    sizes = np.array([len(groups[a]) * len(groups[b]) for a, b in finger_pairs])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total = int(offsets[-1])
    if total <= limit:
        return [
            (x, y) for a, b in finger_pairs
            for x, y in itertools.product(groups[a], groups[b])
        ]

    picks = rng.choice(total, size=limit, replace=False)
    picks.sort()
    slots = np.searchsorted(offsets, picks, side="right") - 1

    out: list[tuple[Impression, Impression]] = []
    for flat, slot in zip(picks, slots):
        a, b = finger_pairs[int(slot)]
        within = int(flat - offsets[slot])
        left, right = groups[a], groups[b]
        out.append((left[within // len(right)], right[within % len(right)]))
    return out


def build_pairs(
    impressions: Sequence[Impression],
    *,
    impostor: ImpostorMode = "all",
    cross_scope: bool = False,
    max_impostor: int | None = None,
    seed: int = 0,
) -> PairSet:
    """All genuine pairs, plus impostor pairs per the chosen rule.

    Impostor pairs stay within a capture scope unless `cross_scope` is set.
    Comparing FVC DB1 against DB2 means comparing an optical sensor against a
    capacitive one: those impostors are trivially separable for reasons that
    have nothing to do with identity, and including them depresses the EER
    without the method having improved. Set `cross_scope=True` only for a
    deliberate cross-sensor experiment.

    `max_impostor` caps the impostor set by seeded random subsampling. Leave it
    unset for the small real datasets; set it for Anguli, where the full set runs
    to millions of pairs.
    """
    groups = _by_finger(impressions)
    scope_of = {f: members[0].scope for f, members in groups.items()}

    genuine: list[tuple[Impression, Impression]] = []
    for members in groups.values():
        genuine.extend(itertools.combinations(members, 2))

    fingers = sorted(groups)
    eligible = [
        (a, b) for a, b in itertools.combinations(fingers, 2)
        if cross_scope or scope_of[a] == scope_of[b]
    ]

    if impostor == "first":
        impostor_pairs = [(groups[a][0], groups[b][0]) for a, b in eligible]
        if max_impostor is not None and len(impostor_pairs) > max_impostor:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(impostor_pairs), max_impostor, replace=False))
            impostor_pairs = [impostor_pairs[int(i)] for i in keep]
    elif max_impostor is not None:
        impostor_pairs = _subsample_impostors(
            groups, eligible, max_impostor, np.random.default_rng(seed)
        )
    else:
        impostor_pairs = [
            (x, y) for a, b in eligible
            for x, y in itertools.product(groups[a], groups[b])
        ]

    return PairSet(genuine=genuine, impostor=impostor_pairs)
