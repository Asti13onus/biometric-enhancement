"""Label real, non-test prints with the pyfing teachers for distillation.

Runs SUFS, SNFOE, SNFFE and SNFEN over FVC2000, FVC2002 and Neurotechnology U.are.U
(1,160 prints) and caches mask, orientation, period and ridge targets. FVC2004 and
Neurotechnology CrossMatch are held out and refused -- see `fpe.data.real`.

    python scripts/build_real_supervision.py --limit 5    # check on a handful first
    python scripts/build_real_supervision.py              # the full pass (~40 min at 2.1 s/img)

Resumable exactly like `build_supervision.py`: each cache file is written atomically, so
killing this at any point and re-running it continues where it stopped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.data.anguli import clear_partial_writes, supervision_path  # noqa: E402
from fpe.data.real import TRAIN_DATASETS, build_real_sample, index_real  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "supervision_real"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    items = index_real(ROOT / "docs" / "datasets" / "manifests", ROOT / "data" / "raw",
                       tuple(args.datasets))
    splits = {s: sum(1 for x, _ in items if x.split == s) for s in ("train", "val")}
    print(f"{len(items):,} real prints from {', '.join(args.datasets)}  "
          f"(train {splits['train']:,}, val {splits['val']:,})")
    missing = [x for x, _ in items if not x.image.is_file()]
    if missing:
        print(f"{len(missing)} image(s) missing, e.g. {missing[0].image}")
        return 1

    stale = clear_partial_writes(args.out)
    if stale:
        print(f"cleared {stale} partial file(s) from an interrupted run")
    todo = [(x, dpi) for x, dpi in items if not supervision_path(x, args.out).is_file()]
    print(f"{len(items) - len(todo):,} already built, {len(todo):,} to build")
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        return 0

    started = time.perf_counter()
    failed = []
    try:
        for n, (sample, dpi) in enumerate(todo, start=1):
            try:
                build_real_sample(sample, args.out, dpi=dpi)
            except (ValueError, OSError) as exc:
                # One unreadable or pathological print must not stop the other thousand;
                # it simply gets no cache and is left out of training. Counted, not hidden.
                failed.append(sample.image)
                print(f"  skipped {sample.image.name}: {exc}", flush=True)
            if n % args.report_every == 0 or n == len(todo):
                rate = (time.perf_counter() - started) / n
                print(f"  {n:,}/{len(todo):,}  {rate:.2f} s/img  "
                      f"eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; progress is saved -- re-run to resume")
        return 130
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min -> {args.out}  "
          f"({len(failed)} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
