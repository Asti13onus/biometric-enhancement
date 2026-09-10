"""Build a manifest for each downloaded dataset.

One CSV per dataset in docs/datasets/manifests/ recording, per image:
relative path, sha256, byte size, pixel dimensions and the *declared* dpi, plus any
per-image labels the dataset ships (MINEX gives quality band and finger position).

These CSVs are committed; the images are not. Every results-registry row carries the
manifest hash, so a dataset cannot change underneath a published number without the
mismatch being detectable.

Declared dpi is recorded as declared, not trusted -- see the resolution warning in
docs/datasets/DOWNLOAD_PLAN.md. SOCOFing claims 500 dpi at ~96x103 px (really ~200),
FVC2002 DB2_B is 569 dpi, L3-SF is 1200 dpi.

Usage:
    python scripts/make_manifest.py            # every dataset present under data/raw
    python scripts/make_manifest.py fvc2004    # just one
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from fpe.data import minex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "docs" / "datasets" / "manifests"

IMAGE_EXT = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".pgm"}

# Declared dpi, keyed by a prefix of "<dataset>/<relpath>"; longest match wins.
# None means the source does not state it -- recorded as unknown, never guessed.
DECLARED_DPI: dict[str, int | None] = {
    "fvc2000/db1_b": 500, "fvc2000/db2_b": 500, "fvc2000/db3_b": 500, "fvc2000/db4_b": 500,
    "fvc2002/db1_b": 500, "fvc2002/db2_b": 569, "fvc2002/db3_b": 500, "fvc2002/db4_b": 500,
    "fvc2004/db1_b": 500, "fvc2004/db2_b": 512, "fvc2004/db3_b": 512, "fvc2004/db4_b": 500,
    "neurotech_crossmatch": 500,
    "neurotech_uareu": 512,
    "fvs": None,
    "minex": minex.MINEX_DPI,
    # The published 1200 dpi applies to the pore-annotated subset (740 imgs at 512x512).
    # The main 7,400-image set is 320x240, which cannot be 1200 dpi -- that would be a
    # 0.27 x 0.20 inch finger. Left unknown until measured from ridge period.
    "l3sf/L3SF_V2/Pore ground truth": 1200,
    "l3sf/L3SF_V2/L3-SF": None,
}

FIELDS = [
    "dataset", "relpath", "sha256", "bytes", "width", "height",
    "declared_dpi", "mode", "subject", "finger", "impression", "quality",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_dpi_for(dataset: str, relpath: Path) -> int | None:
    """Longest matching prefix of '<dataset>/<relpath>' wins, so a subset can override
    its dataset's default."""
    full = f"{dataset}/{relpath.as_posix()}"
    best = max(
        (k for k in DECLARED_DPI if full == k or full.lower().startswith(k.lower() + "/")),
        key=len,
        default=None,
    )
    return DECLARED_DPI.get(best) if best else None


def rows_for_images(dataset: str, root: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT):
        rel = path.relative_to(root)
        try:
            with Image.open(path) as im:
                width, height, mode = im.width, im.height, im.mode
        except Exception as exc:  # noqa: BLE001
            print(f"  !! unreadable: {rel} ({exc})")
            width = height = None
            mode = ""
        rows.append({
            "dataset": dataset, "relpath": rel.as_posix(), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "width": width, "height": height,
            "declared_dpi": declared_dpi_for(dataset, rel), "mode": mode,
            "subject": "", "finger": "", "impression": "", "quality": "",
        })
    return rows


def rows_for_minex(dataset: str, root: Path) -> list[dict]:
    """MINEX raw .gray has no header; dimensions and labels come from the C++ header."""
    meta = minex.parse_metadata(minex.default_header_path(RAW))
    rows, missing, mismatched = [], 0, 0
    for path in sorted(root.rglob("*.gray")):
        rel = path.relative_to(root)
        info = meta.get(path.name)
        if info is None:
            missing += 1
            continue
        size = path.stat().st_size
        if size != info.expected_bytes:
            mismatched += 1
            print(f"  !! {rel}: {size} bytes but header says {info.width}x{info.height}"
                  f" = {info.expected_bytes}")
        rows.append({
            "dataset": dataset, "relpath": rel.as_posix(), "sha256": sha256_file(path),
            "bytes": size, "width": info.width, "height": info.height,
            "declared_dpi": minex.MINEX_DPI, "mode": "L(raw)",
            "subject": info.subject, "finger": info.finger,
            "impression": info.impression, "quality": info.quality,
        })
    if missing:
        print(f"  !! {missing} .gray file(s) absent from the metadata header")
    if not mismatched:
        print("  all byte sizes agree with the header dimensions")
    return rows


def manifest_hash(rows: list[dict]) -> str:
    """Hash of the per-file digests -- identifies the dataset content as a whole."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["relpath"]):
        h.update(f"{r['relpath']}:{r['sha256']}\n".encode())
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if not RAW.exists():
        print(f"nothing to do: {RAW} does not exist")
        return 1
    candidates = sorted(
        d.name for d in RAW.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    wanted = [a for a in argv if not a.startswith("-")] or candidates
    unknown = [w for w in wanted if w not in candidates]
    if unknown:
        print(f"not present under data/raw: {unknown}\navailable: {candidates}")
        return 2

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset in wanted:
        root = RAW / dataset
        print(f"\n== {dataset}")
        rows = rows_for_minex(dataset, root) if dataset == "minex" \
            else rows_for_images(dataset, root)
        if not rows:
            print("  no images found -- skipped")
            continue
        out = MANIFESTS / f"{dataset}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        digest = manifest_hash(rows)
        total_mb = sum(r["bytes"] for r in rows) / 1e6
        dims = {(r["width"], r["height"]) for r in rows}
        dpis = sorted({r["declared_dpi"] for r in rows}, key=lambda v: (v is None, v))
        print(f"  {len(rows)} images, {total_mb:.1f} MB, {len(dims)} distinct size(s), "
              f"declared dpi {dpis}")
        print(f"  manifest {out.relative_to(ROOT)}  hash {digest[:16]}")
        summary.append((dataset, len(rows), total_mb, digest[:16]))

    print(f"\n{'dataset':24} {'images':>7} {'MB':>9}  manifest hash")
    for name, n, mb, digest in summary:
        print(f"{name:24} {n:>7} {mb:>9.1f}  {digest}")
    print(f"{'TOTAL':24} {sum(s[1] for s in summary):>7} {sum(s[2] for s in summary):>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
