"""Download the Group A datasets -- direct download, no account, no agreement.

See docs/datasets/DOWNLOAD_PLAN.md for what each dataset is and why we want it.

Archives land in data/raw/_archives/ and are extracted into data/raw/<key>/.
Every archive's sha256 is recorded in data/raw/_archives/checksums.json so a
re-download that differs from what we originally got is detectable.

Usage:
    python scripts/download_group_a.py                # everything not yet present
    python scripts/download_group_a.py fvc2004 fvs    # only these keys
    python scripts/download_group_a.py --force        # re-download even if present
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVES = RAW / "_archives"
CHECKSUMS = ARCHIVES / "checksums.json"

UA = "Mozilla/5.0 (research dataset fetch; masters thesis; fingerprint enhancement)"

# key -> [(archive filename, url, subdir under data/raw/<key>/), ...]
#
# The subdir matters: the FVC "B" archives all use the same internal filenames
# (101_1.tif, 101_2.tif, ...), so extracting several into one directory silently
# overwrites all but the last. Each archive gets its own subdir.
SOURCES: dict[str, list[tuple[str, str, str]]] = {
    "fvc2000": [
        (f"fvc2000_DB{i}_B.zip", f"http://bias.csr.unibo.it/fvc2000/Downloads/DB{i}_B.zip",
         f"db{i}_b")
        for i in (1, 2, 3, 4)
    ],
    "fvc2002": [
        (f"fvc2002_DB{i}_B.zip", f"http://bias.csr.unibo.it/fvc2002/Downloads/DB{i}_B.zip",
         f"db{i}_b")
        for i in (1, 2, 3, 4)
    ],
    "fvc2004": [
        (f"fvc2004_DB{i}_B.zip", f"http://bias.csr.unibo.it/fvc2004/Downloads/DB{i}_B.zip",
         f"db{i}_b")
        for i in (1, 2, 3, 4)
    ],
    "neurotech_crossmatch": [
        (
            "CrossMatch_Sample_DB.zip",
            "https://www.neurotechnology.com/download/CrossMatch_Sample_DB.zip",
            "",
        )
    ],
    "neurotech_uareu": [
        ("UareU_sample_DB.zip", "https://www.neurotechnology.com/download/UareU_sample_DB.zip",
         "")
    ],
    "fvs": [("fingerprint_bitmaps.zip", "http://fvs.sourceforge.net/fingerprint_bitmaps.zip",
             "")],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r    {got / 1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(archive: Path, dest: Path) -> int:
    """Extract a zip, refusing entries that escape the destination or clobber an
    existing file. Returns the number of files extracted from this archive."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for m in members:
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in {archive.name}: {m.filename}")
            if target.exists():
                raise RuntimeError(
                    f"{archive.name} would overwrite {target.relative_to(dest)} -- "
                    f"give this archive its own subdir"
                )
        zf.extractall(dest)
    return len(members)


def main(argv: list[str]) -> int:
    force = "--force" in argv
    wanted = [a for a in argv if not a.startswith("-")] or list(SOURCES)
    unknown = [k for k in wanted if k not in SOURCES]
    if unknown:
        print(f"unknown keys: {unknown}\navailable: {list(SOURCES)}")
        return 2

    checksums = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    failures: list[str] = []

    for key in wanted:
        print(f"\n== {key} -> {(RAW / key).relative_to(ROOT)}")
        for name, url, subdir in SOURCES[key]:
            dest = RAW / key / subdir if subdir else RAW / key
            archive = ARCHIVES / name
            if archive.exists() and not force:
                print(f"  {name}: already downloaded")
            else:
                print(f"  {name}: {url}")
                try:
                    download(url, archive)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"    FAILED: {exc}")
                    failures.append(f"{key}/{name}: {exc}")
                    continue
            digest = sha256_file(archive)
            prior = checksums.get(name, {}).get("sha256")
            if prior and prior != digest:
                print(f"    WARNING: sha256 changed since last fetch ({prior[:12]} -> {digest[:12]})")
            checksums[name] = {
                "sha256": digest,
                "bytes": archive.stat().st_size,
                "url": url,
                "dataset": key,
            }
            try:
                n = extract(archive, dest)
                print(f"    sha256 {digest[:16]}  {archive.stat().st_size / 1e6:.1f} MB  -> {n} files")
            except Exception as exc:  # noqa: BLE001
                print(f"    EXTRACT FAILED: {exc}")
                failures.append(f"{key}/{name}: extract: {exc}")

    ARCHIVES.mkdir(parents=True, exist_ok=True)
    CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True))
    print(f"\nchecksums written to {CHECKSUMS.relative_to(ROOT)}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
