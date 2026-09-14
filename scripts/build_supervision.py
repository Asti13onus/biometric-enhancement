"""Build the cached training supervision for the Anguli corpus (plan U2).

Runs Cappelli's pretrained estimators over every **clean** impression and caches the
segmentation, orientation and frequency targets. Roughly 16,750 impressions at ~0.6 s each,
so about three hours -- which is why the whole thing is built to survive interruption.

    python scripts/build_supervision.py --limit 200      # validate on a subset first
    python scripts/build_supervision.py                  # the full pass
    python scripts/build_supervision.py --manifest-only  # just the evaluation manifest

**Resumability.** Every cache file is written to a temporary name and then renamed into
place, and a rename cannot half-finish. So a file that exists is complete, and resuming is
simply "skip what is already there". Kill this script at any point -- Ctrl-C, a crash, a
reboot -- and re-running it picks up where it stopped without redoing finished work and
without leaving a truncated array that would load successfully and quietly corrupt
training. Stale temporary files from a hard kill are cleared at startup.

This machine has already lost one long run to memory exhaustion, so the defaults are
conservative: one image at a time, nothing bulk-loaded, progress reported as it goes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Keras reads its backend at import time, and pyfing imports keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.data.anguli import (  # noqa: E402
    SPLITS, build_sample, clear_partial_writes, index_corpus, supervision_path,
    write_manifest,
)

DEFAULT_CORPUS = ROOT / "data" / "processed" / "anguli_dev_5584"
DEFAULT_OUT = ROOT / "data" / "processed" / "supervision"
DEFAULT_MANIFEST = ROOT / "docs" / "datasets" / "manifests" / "anguli_dev_5584.csv"


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--impressions", type=int, default=None,
                    help="use only the first N impressions per finger (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="build at most N samples this run, then stop cleanly")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild samples that already have a cache")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the evaluation manifest and exit")
    ap.add_argument("--report-every", type=int, default=100)
    args = ap.parse_args()

    if not args.corpus.is_dir():
        ap.error(f"corpus not found: {args.corpus}")

    print(f"indexing {args.corpus.relative_to(ROOT)} ...")
    samples = index_corpus(args.corpus, impressions=args.impressions)
    fingers = {s.finger_id for s in samples}
    counts = {name: len({s.finger_id for s in samples if s.split == name}) for name in SPLITS}
    print(f"  {len(samples):,} impressions from {len(fingers):,} fingers")
    print("  fingers per split: " + "  ".join(f"{k} {v:,}" for k, v in counts.items()))

    written = write_manifest(samples, args.manifest)
    print(f"  manifest: {written:,} test-split images -> "
          f"{args.manifest.relative_to(ROOT)}")
    if args.manifest_only:
        return 0

    stale = clear_partial_writes(args.out)
    if stale:
        print(f"  cleared {stale} partial file(s) from an interrupted run")

    todo = [s for s in samples
            if args.overwrite or not supervision_path(s, args.out).is_file()]
    done_already = len(samples) - len(todo)
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        print(f"\nnothing to do: all {len(samples):,} samples are already built")
        return 0

    print(f"\n{done_already:,} already built, {len(todo):,} to build this run")
    print(f"writing to {args.out.relative_to(ROOT)}  (safe to interrupt; re-run to resume)\n")

    started = time.perf_counter()
    built = 0
    try:
        for n, sample in enumerate(todo, start=1):
            _, was_built = build_sample(sample, args.out, overwrite=args.overwrite)
            built += int(was_built)
            if n % args.report_every == 0 or n == len(todo):
                elapsed = time.perf_counter() - started
                rate = elapsed / n
                remaining = rate * (len(todo) - n)
                print(f"  {n:,}/{len(todo):,}  {rate * 1000:.0f} ms/img  "
                      f"elapsed {human(elapsed)}  eta {human(remaining)}")
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        print(f"\ninterrupted after {built:,} samples in {human(elapsed)}.")
        print("progress is saved -- re-run the same command to resume.")
        return 130

    elapsed = time.perf_counter() - started
    total_done = done_already + built
    print(f"\nbuilt {built:,} samples in {human(elapsed)} "
          f"({total_done:,}/{len(samples):,} complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
