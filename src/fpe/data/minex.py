"""Metadata for the NIST MINEX III validation imagery.

The imagery ships as headerless raw 8-bit greyscale (`.gray`), so dimensions have to
come from somewhere else. They live in the validation driver's C++ header,
`minexiii/validation/minexiii_validation_data.h`, as a `SAMPLE_DATA` map:

    {"a001_02.gray", {329, 503, MINEX_QUALITY_FAIR,
        MINEX_FINGER_RIGHT_INDEX, MINEX_IMP_NONLIVESCAN_PLAIN}},

Besides width and height that gives us a **quality band** and a **finger position**
per image, which is exactly the finger-type stratification axis the benchmark needs
(literature review Gap 4 / thesis plan Gap D).

Filenames are `<encounter><subject>_<finger>.gray` where encounter is `a` or `b`,
so the two encounters of one finger form a genuine pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# MINEX III imagery is 500 dpi (test plan); not recorded per-image in the header.
MINEX_DPI = 500

_ENTRY = re.compile(
    r'\{\s*"(?P<name>[^"]+\.gray)"\s*,\s*\{\s*'
    r"(?P<width>\d+)\s*,\s*(?P<height>\d+)\s*,\s*"
    r"(?P<quality>MINEX_QUALITY_\w+)\s*,\s*"
    r"(?P<finger>MINEX_FINGER_\w+)\s*,\s*"
    r"(?P<impression>MINEX_IMP_\w+)\s*\}\s*\}",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MinexImage:
    name: str
    width: int
    height: int
    quality: str
    finger: str
    impression: str
    encounter: str
    subject: str

    @property
    def expected_bytes(self) -> int:
        """8-bit greyscale, no header."""
        return self.width * self.height


def _strip_prefix(token: str, prefix: str) -> str:
    return token[len(prefix) :].lower() if token.startswith(prefix) else token.lower()


def parse_metadata(header_path: str | Path) -> dict[str, MinexImage]:
    """Parse SAMPLE_DATA out of minexiii_validation_data.h, keyed by filename."""
    text = Path(header_path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, MinexImage] = {}
    for m in _ENTRY.finditer(text):
        name = m.group("name")
        stem = name[: -len(".gray")]
        encounter_subject, _, finger_num = stem.partition("_")
        out[name] = MinexImage(
            name=name,
            width=int(m.group("width")),
            height=int(m.group("height")),
            quality=_strip_prefix(m.group("quality"), "MINEX_QUALITY_"),
            finger=_strip_prefix(m.group("finger"), "MINEX_FINGER_"),
            impression=_strip_prefix(m.group("impression"), "MINEX_IMP_"),
            encounter=encounter_subject[:1],
            subject=encounter_subject[1:],
        )
    if not out:
        raise ValueError(f"no SAMPLE_DATA entries found in {header_path}")
    return out


def default_header_path(raw_root: str | Path) -> Path:
    return (
        Path(raw_root)
        / "minex"
        / "minexiii"
        / "validation"
        / "minexiii_validation_data.h"
    )
