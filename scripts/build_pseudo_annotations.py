"""Pseudo-annotate real, non-test prints with WAFEN itself (arm two: domain alignment).

The Joshi line (CDC-GAN 2022; pseudo-annotations 2023) aligns a synthetically-trained
model to the real domain without ground truth: the model's own outputs on real prints
become training targets. This script runs a frozen checkpoint over FVC2000, FVC2002 and
U.are.U and caches its five outputs in the supervision layout `WafenDataset` reads --
including `evidence` (the model's confidence), which photometric-mode training uses as
the confidence target since no wear model runs to provide one.

    python scripts/build_pseudo_annotations.py                    # ~5 min on the GPU
    python scripts/build_pseudo_annotations.py --checkpoint ...   # a different annotator

Distinct from scripts/build_real_supervision.py, whose annotator is pyfing (external
teachers -- that is distillation, arm one's addendum). Here the annotator is the model
being aligned, which is what makes it *unsupervised* domain alignment. FVC2004 and
Neurotechnology CrossMatch remain held out (fpe.data.real refuses them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fpe.data.anguli import clear_partial_writes, supervision_path  # noqa: E402
from fpe.data.convert import load_greyscale  # noqa: E402
from fpe.data.real import TRAIN_DATASETS, index_real  # noqa: E402
from fpe.models.wafen import Wafen, WafenConfig, decode_orientation  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "supervision_pseudo"
DEFAULT_CHECKPOINT = ROOT / "data" / "work" / "wafen" / "best.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="the annotator; the arm-two default is the synthetic-only model")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Wafen(WafenConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    stride = 2 ** (model.config.depth - 1)
    annotator = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }

    items = index_real(ROOT / "docs" / "datasets" / "manifests", ROOT / "data" / "raw",
                       tuple(args.datasets))
    stale = clear_partial_writes(args.out)
    if stale:
        print(f"cleared {stale} partial file(s)")
    todo = [(s, dpi) for s, dpi in items if not supervision_path(s, args.out).is_file()]
    print(f"{len(items):,} prints, {len(items) - len(todo):,} already annotated, "
          f"{len(todo):,} to do on {device}")
    if args.limit is not None:
        todo = todo[:args.limit]

    started = time.perf_counter()
    for n, (sample, dpi) in enumerate(todo, start=1):
        img = load_greyscale(sample.image).astype(np.float32) / 255.0
        h, w = img.shape
        padded = np.pad(img, ((0, -h % stride), (0, -w % stride)), constant_values=1.0)
        with torch.no_grad():
            out = model(torch.from_numpy(padded)[None, None].to(device))
        mask = (out.segmentation[0, 0, :h, :w] >= 0.5).cpu().numpy().astype(np.uint8)
        orientation = decode_orientation(out.orientation)[0, 0, :h, :w].cpu().numpy()
        ridge = np.round(out.ridge[0, 0, :h, :w].cpu().numpy() * 255).astype(np.uint8)
        period = out.period[0, 0, :h, :w].cpu().numpy()
        evidence = out.confidence[0, 0, :h, :w].cpu().numpy()

        path = supervision_path(sample, args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({
            "finger_id": sample.finger_id, "dataset": sample.bucket,
            "impression": sample.impression, "split": sample.split, "dpi": dpi,
            "teachers": {"all": "self (pseudo-annotation)", **annotator},
            "image": sample.image.name,
        })
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:  # handle, not path: savez would append ".npz"
            np.savez_compressed(
                fh, mask=mask, ridge=np.where(mask > 0, ridge, 0).astype(np.uint8),
                orientation=orientation.astype(np.float16),
                period=period.astype(np.float16),
                evidence=evidence.astype(np.float16), meta=np.array(meta),
            )
        os.replace(tmp, path)
        if n % 100 == 0 or n == len(todo):
            rate = (time.perf_counter() - started) / n
            print(f"  {n:,}/{len(todo):,}  {rate * 1000:.0f} ms/img  "
                  f"eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
