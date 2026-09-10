"""Download the Group B datasets -- free, but each needs an account.

See docs/datasets/DOWNLOAD_PLAN.md. Access state per dataset:

  socofing  Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json (or ~/.kaggle/).
            Automated once the token exists.
  chalearn  ChaLearn LAP account. The file lists are login-gated, so the archive URLs
            cannot be discovered anonymously; pass --chalearn-url once per file.
  casia     An *approved* idealtest.org account. Manual browser download behind a human
            review step; drop the archives into data/raw/_archives/ and re-run with one
            --casia-archive per file. Walkthrough: docs/datasets/casia-access.md

Usage:
    python scripts/download_group_b.py socofing
    python scripts/download_group_b.py chalearn --chalearn-url "<signed url>" ...
    python scripts/download_group_b.py casia
        --casia-archive "CASIA-FingerprintV5 (000-099).zip"
        --casia-archive "CASIA-Fingerprint-Subject-Ageing.zip"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVES = RAW / "_archives"
CHECKSUMS = ARCHIVES / "checksums.json"

UA = "Mozilla/5.0 (research dataset fetch; masters thesis; fingerprint enhancement)"
KAGGLE_SLUG = "ruizgara/socofing"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name: str, path: Path, url: str, dataset: str, **extra) -> str:
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    checksums = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    digest = sha256_file(path)
    prior = checksums.get(name, {}).get("sha256")
    if prior and prior != digest:
        print(f"    WARNING: sha256 changed since last fetch ({prior[:12]} -> {digest[:12]})")
    checksums[name] = {
        "sha256": digest, "bytes": path.stat().st_size, "url": url,
        "dataset": dataset, **extra,
    }
    CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True))
    return digest


def extract(archive: Path, dest: Path) -> int:
    """Extract a zip, refusing path escapes and refusing to clobber existing files."""
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


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            got += len(chunk)
            pct = f" ({100 * got / total:.0f}%)" if total else ""
            print(f"\r    {got / 1e6:8.1f} MB{pct}", end="", flush=True)
    print()
    tmp.replace(dest)


def do_socofing(force: bool) -> int:
    archive = ARCHIVES / "socofing.zip"
    dest = RAW / "socofing"
    if archive.exists() and not force:
        print(f"  archive already present: {archive.name}")
    else:
        token = Path(os.environ.get("USERPROFILE", Path.home())) / ".kaggle" / "kaggle.json"
        if not token.exists() and not (Path.home() / ".kaggle" / "kaggle.json").exists():
            print("  no Kaggle token found. Create one at kaggle.com -> Settings -> API")
            print(r"  -> Create New Token, then save it as %USERPROFILE%\.kaggle\kaggle.json")
            return 1
        ARCHIVES.mkdir(parents=True, exist_ok=True)
        print(f"  kaggle datasets download -d {KAGGLE_SLUG}")
        proc = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "-p", str(ARCHIVES), "--force" if force else "--unzip" if False else "-q"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr.strip()[:500]}")
            return 1
        produced = ARCHIVES / "socofing.zip"
        if not produced.exists():
            candidates = sorted(ARCHIVES.glob("*ocofing*.zip"))
            if not candidates:
                print(f"  FAILED: no archive produced. stdout: {proc.stdout[:300]}")
                return 1
            candidates[0].rename(produced)
        archive = produced

    digest = record(archive.name, archive, f"kaggle:{KAGGLE_SLUG}", "socofing",
                    licence="non-commercial research only")
    n = extract(archive, dest)
    print(f"  sha256 {digest[:16]}  {archive.stat().st_size / 1e6:.1f} MB  -> {n} files")
    return 0


def do_chalearn(urls: list[str], force: bool) -> int:
    if not urls:
        print("  ChaLearn file lists are login-gated, so the archive URLs cannot be")
        print("  discovered without an account. Sign up (free) at")
        print("     https://chalearnlap.cvc.uab.cat/register/")
        print("  then open each of these and copy the download link for every file:")
        print("     https://chalearnlap.cvc.uab.cat/dataset/32/data/55/files/   (Training)")
        print("     https://chalearnlap.cvc.uab.cat/dataset/32/data/56/files/   (Validation)")
        print("     https://chalearnlap.cvc.uab.cat/dataset/32/data/57/files/   (Test)")
        print("  and re-run with one --chalearn-url per file.")
        return 1
    dest = RAW / "chalearn"
    failed = 0
    for url in urls:
        name = Path(unquote(urlparse(url).path)).name or "chalearn_part.zip"
        archive = ARCHIVES / name
        print(f"  {name}")
        if archive.exists() and not force:
            print("    already downloaded")
        else:
            try:
                download(url, archive)
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED: {exc}")
                failed += 1
                continue
        digest = record(name, archive, url, "chalearn")
        sub = dest / archive.stem
        try:
            n = extract(archive, sub)
            print(f"    sha256 {digest[:16]}  -> {n} files in {sub.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001
            print(f"    EXTRACT FAILED: {exc}")
            failed += 1
    return 1 if failed else 0


# Expected CASIA archives, from the portal API (docs/datasets/casia-access.md).
# Dataset 7 ships as five subject-range subsets; dataset 15 as one archive.
CASIA_ARCHIVES: dict[str, str] = {
    "CASIA-FingerprintV5 (000-099).zip": "casia_v5",
    "CASIA-FingerprintV5 (100-199).zip": "casia_v5",
    "CASIA-FingerprintV5 (200-299).zip": "casia_v5",
    "CASIA-FingerprintV5 (300-399).zip": "casia_v5",
    "CASIA-FingerprintV5 (400-499).zip": "casia_v5",
    "CASIA-Fingerprint-Subject-Ageing.zip": "casia_ageing",
}


def do_casia(archive_names: list[str], force: bool) -> int:
    if not archive_names:
        print("  CASIA needs an *approved* idealtest.org account and is served through a")
        print("  JavaScript portal behind a manual review step, so it cannot be scripted.")
        print("  Full walkthrough: docs/datasets/casia-access.md")
        print("  Summary: register at https://www.idealtest.org/register with your")
        print("  institutional email (expired TLS cert -- the browser warning is theirs),")
        print("  wait for approval, then download from datasetDetail/7 and datasetDetail/15")
        print(f"  into {ARCHIVES.relative_to(ROOT)} and re-run with --casia-archive per file.")
        print("\n  Expected archives:")
        for name, dataset in CASIA_ARCHIVES.items():
            here = " [present]" if (ARCHIVES / name).exists() else ""
            print(f"    {dataset:12} {name}{here}")
        print("\n  Licence: research/education only, no publishing or redistribution.")
        print("  Manifest only -- never commit the images, and they cannot appear in the")
        print("  public benchmark release (split files referencing them only).")
        return 1

    failed = 0
    for archive_name in archive_names:
        archive = ARCHIVES / archive_name
        print(f"  {archive_name}")
        if not archive.exists():
            print(f"    not found in {ARCHIVES.relative_to(ROOT)}")
            failed += 1
            continue
        dataset = CASIA_ARCHIVES.get(archive_name)
        if dataset is None:
            dataset = "casia_v5"
            print(f"    unrecognised name; filing under {dataset}. Expected one of:")
            for name in CASIA_ARCHIVES:
                print(f"      {name}")
        digest = record(archive.name, archive, "manual:idealtest.org", dataset,
                        licence="research/education only; no publishing or redistribution")
        # Each subset zip carries distinct subject ranges, so they share one tree safely;
        # extract() still refuses if any two actually collide.
        try:
            n = extract(archive, RAW / dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"    EXTRACT FAILED: {exc}")
            failed += 1
            continue
        print(f"    sha256 {digest[:16]}  {archive.stat().st_size / 1e6:.1f} MB -> {n} files")

    missing = [n for n in CASIA_ARCHIVES if not (ARCHIVES / n).exists()]
    if missing:
        print(f"\n  still missing {len(missing)} of {len(CASIA_ARCHIVES)} expected archives:")
        for name in missing:
            print(f"    {name}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="*", default=None,
                    choices=["socofing", "chalearn", "casia"] if len(sys.argv) > 1 else None)
    ap.add_argument("--chalearn-url", action="append", default=[],
                    help="a ChaLearn archive URL copied from the logged-in file list")
    ap.add_argument("--casia-archive", action="append", default=[], metavar="FILENAME",
                    help="a manually downloaded CASIA archive in data/raw/_archives/; "
                         "repeat for each of the five V5 subsets plus the ageing set")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    wanted = args.datasets or ["socofing", "chalearn", "casia"]
    rc = 0
    for name in wanted:
        print(f"\n== {name}")
        if name == "socofing":
            rc |= do_socofing(args.force)
        elif name == "chalearn":
            rc |= do_chalearn(args.chalearn_url, args.force)
        else:
            rc |= do_casia(args.casia_archive, args.force)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
