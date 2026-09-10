"""Generate paired clean/degraded fingerprints with Anguli.

Anguli is the generator that produced the ChaLearn LAP Track 3 set, at the same
275x400 geometry, so this is our route to an equivalent paired corpus if ChaLearn
stays inaccessible -- and the substrate the wear-degradation model is applied to
either way. See docs/decisions/0003-chalearn-contingency.md.

Two operational traps this wrapper exists to handle:

1. Anguli resolves its `Filterbank/` and `Densitymaps/` directories relative to the
   *current working directory*, not the executable, and fails with
   "Error: Not able to load Filter Bank" if run from anywhere else.
2. Its default image type is **jpg**. Lossy compression on ridge structure would
   contaminate every downstream measurement, so png is forced here.

It also will not create its own output directory ("WARNING: Cannon create directory"),
so we create it first.

Output layout, paired by construction:
    <out>/Fingerprints/fp_N/M.png    clean master  (ground truth)
    <out>/Impression_1/fp_N/M.png    degraded impression 1
    <out>/Impression_2/fp_N/M.png    degraded impression 2   ...
    <out>/Meta Info/fp_N/M.txt       pattern class + singular points
    <out>/Anguli.conf                the exact config used, including the seed

Usage:
    python scripts/generate_anguli.py --num 200 --impressions 3 --out data/processed/anguli_dev
    python scripts/generate_anguli.py --num 5 --preset chalearn-like --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANGULI_DIR = ROOT / "data" / "external" / "anguli" / "Anguli-MSVC"
ANGULI_EXE = ANGULI_DIR / "Anguli.exe"

# Degradation presets. "clean" leaves the impression model as the only difference
# between impressions; "chalearn-like" approximates the artefact mix the ChaLearn
# set was built with (blur/noise/scratches/rotation/translation).
#
# NOTE: this is a *latent-print* degradation model. It is the baseline arm the thesis
# argues against, not the wear model -- see STRATEGY.md section 1.
PRESETS: dict[str, dict[str, str]] = {
    "clean": {},
    "chalearn-like": {"noise": "2 6", "scratch": "3 12", "rot": "15", "trans": "10"},
    "severe": {"noise": "5 8", "scratch": "8 20", "rot": "25", "trans": "15"},
}


def build_command(args: argparse.Namespace, out: Path) -> list[str]:
    cmd = [
        str(ANGULI_EXE),
        "-num", str(args.num),
        "-ni", str(args.impressions),
        "-seed", str(args.seed),
        "-numT", str(args.threads),
        "-itype", args.itype,
        "-outdir", str(out),
    ]
    if args.meta:
        cmd.append("-meta")
    if args.cdist:
        cmd += ["-cdist", args.cdist]
    for flag, value in PRESETS[args.preset].items():
        cmd += [f"-{flag}", value]
    for override in args.set or []:
        flag, _, value = override.partition("=")
        cmd += [f"-{flag}", value]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, required=True, help="number of distinct fingers")
    ap.add_argument("--impressions", type=int, default=2, help="impressions per finger")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="chalearn-like")
    ap.add_argument("--itype", default="png",
                    help="image type; keep png -- jpg is lossy and corrupts ridge detail")
    ap.add_argument("--cdist", help="pattern class distribution: natural, arch, tarch, "
                                    "right, left, dloop, ...")
    ap.add_argument("--meta", action="store_true", default=True)
    ap.add_argument("--set", action="append", metavar="FLAG=VALUE",
                    help="pass an extra Anguli flag, e.g. --set noise='3 7'")
    args = ap.parse_args()

    if not ANGULI_EXE.exists():
        print(f"Anguli not found at {ANGULI_EXE}")
        print("Fetch it from https://dsl.cds.iisc.ac.in/projects/Anguli/ (expired TLS cert;")
        print("use curl -k) and extract into data/external/anguli/")
        return 1
    if args.itype.lower() in {"jpg", "jpeg"}:
        print("refusing jpg: lossy compression corrupts ridge structure. Use --itype png.")
        return 2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)  # Anguli will not create it itself

    cmd = build_command(args, out)
    print(f"$ {' '.join(cmd)}\n  (cwd={ANGULI_DIR})")
    # cwd must be the Anguli install dir or the filter bank fails to load
    proc = subprocess.run(cmd, cwd=ANGULI_DIR, text=True)
    if proc.returncode != 0:
        print(f"Anguli exited {proc.returncode}")
        return proc.returncode

    clean = sorted((out / "Fingerprints").rglob(f"*.{args.itype}"))
    impressions = {
        d.name: sorted(d.rglob(f"*.{args.itype}"))
        for d in sorted(out.glob("Impression_*")) if d.is_dir()
    }
    print(f"\n{len(clean)} clean masters")
    for name, files in impressions.items():
        print(f"{len(files):6} in {name}")

    (out / "generation.json").write_text(json.dumps({
        "command": cmd, "cwd": str(ANGULI_DIR), "preset": args.preset,
        "preset_flags": PRESETS[args.preset], "seed": args.seed,
        "num_fingers": args.num, "impressions_per_finger": args.impressions,
        "clean_images": len(clean),
        "degraded_images": {k: len(v) for k, v in impressions.items()},
        "note": "latent-style degradation (baseline arm), not the wear model",
    }, indent=2))
    print(f"\nwrote {(out / 'generation.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
