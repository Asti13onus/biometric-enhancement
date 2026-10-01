"""The full benchmark: methods x held-out sets, severity and position strata (plan U12).

Three phases, each cell one registry row, each cell skipped when its row already exists,
so the whole thing resumes from wherever it died:

1. Methods (none, SNFEN, WAFEN-aligned) over every held-out real set. FVC2000, FVC2002
   and U.are.U are *excluded as training data* and the tables must say so; FVS's dpi is
   undeclared and assumed 500, stated in the row.
2. The severity axis (R6): CrossMatch's clean prints degraded by the wear model at
   severities 0.3 / 0.5 / 0.7, every method on each.
3. MINEX finger-position stratification (R21) from its labelled positions, re-matching
   cached templates through per-position manifests.

    python experiments/benchmark.py              # resumable, several hours
    python experiments/benchmark.py --phase 1    # one phase at a time
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fpe.eval.baseline import run_baseline  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402

CHECKPOINT = ROOT / "data" / "work" / "wafen_da" / "best.pt"
SEVERITIES = (0.3, 0.5, 0.7)
MANIFESTS = ROOT / "docs" / "datasets" / "manifests"
REAL_SETS = ("neurotech_crossmatch", "minex", "fvs")
METHODS = ("none", "SNFEN", "WAFEN")


def done_cells() -> set[tuple[str, str]]:
    cells = set()
    reg = ROOT / "results" / "registry" / "runs.jsonl"
    if reg.is_file():
        for line in reg.read_text().splitlines():
            r = json.loads(line)
            if r.get("experiment") == "benchmark":
                cells.add((r["dataset"], r["method"]))
    return cells


def enhancer_for(method: str, work: Path):
    if method == "none":
        return None
    if method == "WAFEN":
        from fpe.models.wafen_enhancer import WafenEnhancer

        return WafenEnhancer(CHECKPOINT, cache_dir=work)
    from fpe.models.pyfing_baseline import PyfingEnhancer

    return PyfingEnhancer(method, cache_dir=work)


def method_label(method: str) -> str:
    return f"WAFEN[{CHECKPOINT.parent.name}]" if method == "WAFEN" else method


def run_cell(dataset_label, manifest, data_root, work, method, notes, done, dry_run):
    label = method_label(method)
    if (dataset_label, label) in done:
        print(f"  [cached] {dataset_label} / {label}")
        return
    print(f"  running {dataset_label} / {label}")
    result = run_baseline(manifest_path=manifest, data_root=data_root, work_dir=work,
                          preprocess=enhancer_for(method, work), progress=print)
    m = result.metrics
    print(f"    EER {m['eer']:.4f} [{m['eer_ci_low']:.4f}, {m['eer_ci_high']:.4f}]  "
          f"NFIQ2 {m.get('nfiq2_mean', float('nan')):.1f}")
    if not dry_run:
        log_run(experiment="benchmark", dataset=dataset_label, method=label,
                metrics=m, config={"checkpoint": str(CHECKPOINT) if method == "WAFEN"
                                   else None},
                manifest_path=manifest, notes=notes)


def prepare_minex_png() -> tuple[Path, Path]:
    """MINEX ships headerless .gray raws PIL cannot read: convert once to PNG using the
    manifest's per-image dimensions, and emit a parallel manifest with .png relpaths."""
    from PIL import Image

    out_root = ROOT / "data" / "processed" / "minex_png"
    out_manifest = out_root / "minex_png.csv"
    src_manifest = MANIFESTS / "minex.csv"
    raw_root = ROOT / "data" / "raw" / "minex"
    rows = list(csv.DictReader(src_manifest.open(newline="", encoding="utf-8")))
    # The validation imagery ships calibration patterns alongside fingers --
    # random_rects.gray parses to subject "andom". A fingerprint row has a
    # three-digit subject; anything else is test imagery, not a finger.
    fingers = [r for r in rows if r["subject"].isdigit() and len(r["subject"]) == 3]
    if len(fingers) != len(rows):
        dropped = sorted(Path(r["relpath"]).name for r in rows if r not in fingers)
        print(f"  minex: excluding {len(dropped)} non-finger image(s): {dropped}")
    rows = fingers
    if out_manifest.is_file():
        return out_manifest, out_root
    for row in rows:
        rel = Path(row["relpath"])
        dst = out_root / rel.with_suffix(".png")
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        w, h = int(row["width"]), int(row["height"])
        raw = np.fromfile(raw_root / rel, dtype=np.uint8)
        Image.fromarray(raw.reshape(h, w)).save(dst, "PNG")
        row_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        row["sha256"], row["bytes"] = row_sha, str(dst.stat().st_size)
    with out_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row["relpath"] = str(Path(row["relpath"]).with_suffix(".png"))
            writer.writerow(row)
    return out_manifest, out_root


def prepare_degraded_crossmatch(severity: float) -> Path:
    """CrossMatch degraded by the wear model at a fixed severity, deterministic per file."""
    from PIL import Image

    from fpe.data.convert import load_greyscale
    from fpe.degradation.wear import WearConfig, WearDegradation

    out_root = ROOT / "data" / "processed" / f"crossmatch_wear_s{severity:.1f}"
    src_root = ROOT / "data" / "raw" / "neurotech_crossmatch"
    wear = WearDegradation(WearConfig())
    rows = list(csv.DictReader((MANIFESTS / "neurotech_crossmatch.csv")
                               .open(newline="", encoding="utf-8")))
    todo = [r for r in rows if not (out_root / r["relpath"]).is_file()]
    for n, row in enumerate(todo, 1):
        rel = row["relpath"]
        img = load_greyscale(src_root / rel).astype(np.float32) / 255.0
        seed = int.from_bytes(hashlib.blake2b(f"{severity}:{rel}".encode(),
                                              digest_size=4).digest(), "big")
        out = wear(img, seed=seed, severity=severity).image
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(np.clip(out, 0, 1) * 255).astype(np.uint8)).save(
            dst, "PNG" if dst.suffix == ".png" else None)
        if n % 100 == 0:
            print(f"    degrading s{severity}: {n}/{len(todo)}")
    return out_root


def minex_positions(manifest: Path) -> dict[str, list[dict]]:
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_pos: dict[str, list[dict]] = {}
    for row in rows:
        by_pos.setdefault(row["finger"], []).append(row)
    return by_pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    done = done_cells()
    phases = [args.phase] if args.phase else [1, 2, 3]

    minex_manifest, minex_root = prepare_minex_png()

    if 1 in phases:
        print("phase 1: methods x held-out real sets")
        for name in REAL_SETS:
            manifest = minex_manifest if name == "minex" else MANIFESTS / f"{name}.csv"
            data_root = minex_root if name == "minex" else ROOT / "data" / "raw" / name
            for method in METHODS:
                work = ROOT / "data" / "work" / f"bm_{name}_{method}"
                notes = ("FVS declares no dpi; 500 assumed." if name == "fvs" else
                         "MINEX .gray converted to PNG; identities from manifest "
                         "columns." if name == "minex" else None)
                run_cell(name, manifest, data_root, work, method, notes, done,
                         args.dry_run)

    if 2 in phases:
        print("phase 2: wear-severity axis on CrossMatch")
        for severity in SEVERITIES:
            data_root = prepare_degraded_crossmatch(severity)
            for method in METHODS:
                label = f"neurotech_crossmatch (wear {severity:.1f})"
                work = ROOT / "data" / "work" / f"bm_cm_s{severity:.1f}_{method}"
                run_cell(label, MANIFESTS / "neurotech_crossmatch.csv", data_root,
                         work, method,
                         f"Clean CrossMatch degraded by the wear model at severity "
                         f"{severity}; deterministic per-file seeds.", done,
                         args.dry_run)

    if 3 in phases:
        print("phase 3: MINEX finger-position strata (cached templates)")
        strata_dir = ROOT / "data" / "work" / "bm_minex_strata"
        strata_dir.mkdir(parents=True, exist_ok=True)
        rows_by_pos = minex_positions(minex_manifest)
        for position, rows in sorted(rows_by_pos.items()):
            sub = strata_dir / f"minex_{position}.csv"
            with sub.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            for method in METHODS:
                work = ROOT / "data" / "work" / f"bm_minex_{method}"  # shared caches
                run_cell(f"minex/{position}", sub, minex_root, work, method,
                         "Position stratum; templates shared with the full-set run.",
                         done, args.dry_run)
    print("benchmark complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
