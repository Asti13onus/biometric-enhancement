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

__all__ = ["Impression", "parse_manifest", "build_pairs", "PairSet"]

ImpostorMode = Literal["all", "first"]


@dataclass(frozen=True)
class Impression:
    """One image, with the identity resolved."""

    dataset: str
    relpath: str
    finger_id: str  # identity unit; genuine pairs share this
    impression: str

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


def _identity(dataset: str, relpath: str) -> tuple[str, str] | None:
    """(finger_id, impression) for one manifest row, or None if unparseable."""
    p = Path(relpath)
    stem = p.stem
    # Scope by every directory above the file, so db1_b/101 != db2_b/101.
    scope = "/".join(p.parts[:-1])

    if m := _THREE_PART.match(stem):
        finger = f"{m['subject']}_{m['finger']}"
        return (f"{dataset}/{scope}/{finger}" if scope else f"{dataset}/{finger}",
                m["impr"])
    if m := _TWO_PART.match(stem):
        finger = m["finger"]
        return (f"{dataset}/{scope}/{finger}" if scope else f"{dataset}/{finger}",
                m["impr"])
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
            finger_id, impr = ident
            out.append(
                Impression(
                    dataset=row["dataset"],
                    relpath=relpath,
                    finger_id=finger_id,
                    impression=impr,
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


def build_pairs(
    impressions: Sequence[Impression], *, impostor: ImpostorMode = "all"
) -> PairSet:
    """All genuine pairs, plus impostor pairs per the chosen rule."""
    groups = _by_finger(impressions)

    genuine: list[tuple[Impression, Impression]] = []
    for members in groups.values():
        genuine.extend(itertools.combinations(members, 2))

    fingers = sorted(groups)
    impostor_pairs: list[tuple[Impression, Impression]] = []
    for a, b in itertools.combinations(fingers, 2):
        if impostor == "first":
            impostor_pairs.append((groups[a][0], groups[b][0]))
        else:
            impostor_pairs.extend(itertools.product(groups[a], groups[b]))

    return PairSet(genuine=genuine, impostor=impostor_pairs)
