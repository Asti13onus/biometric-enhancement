"""Train WAFEN on the wear-degraded Anguli corpus (plan U7).

    python experiments/train_wafen.py --canary          # wiring check, ~1 minute
    python experiments/train_wafen.py --epochs 25

The canary runs first by default and the real run refuses to start if it fails. A network
that cannot memorise ten images has a bug, and discovering that an hour into training --
or worse, from a bad result three days later -- is entirely avoidable.

Every run writes a registry row carrying the config hash and git SHA, so a training run is
as traceable as an evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpe.data.anguli import index_corpus, supervision_path  # noqa: E402
from fpe.data.dataset import AugmentConfig, WafenDataset  # noqa: E402
from fpe.degradation.wear import WearConfig  # noqa: E402
from fpe.eval.registry import log_run  # noqa: E402
from fpe.models.wafen import Wafen, WafenConfig  # noqa: E402
from fpe.train import TrainConfig, overfit_check, resolve_device, train  # noqa: E402

CORPUS = ROOT / "data" / "processed" / "anguli_dev_5584"
SUPERVISION = ROOT / "data" / "processed" / "supervision"
CANARY_THRESHOLD = 0.35


class MixedDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset that forwards `set_epoch`, so every part sees fresh damage each epoch."""

    def set_epoch(self, epoch: int) -> None:
        for part in self.datasets:
            part.set_epoch(epoch)

    @property
    def pairs(self) -> bool:
        return any(getattr(part, "pairs", False) for part in self.datasets)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--supervision", type=Path, default=SUPERVISION)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "work" / "wafen")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = load in the main process. Each Windows worker re-imports "
                         "torch and its CUDA DLLs; two of them exhausted the pagefile "
                         "(WinError 1455) and hung the run")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 mixed precision. Off by default: on the GTX 1650 a Conv2d "
                         "with inputs near 5 returns NaN under autocast (fp32 is exact), "
                         "and the card has no tensor cores, so fp16 buys little")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads for torch and OpenCV; PROJECT_RULES.md caps this at 4")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None,
                    help="evenly spaced validation subset, to keep per-epoch eval short")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canary", action="store_true", help="run the canary and exit")
    ap.add_argument("--skip-canary", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="do not write the registry")
    ap.add_argument("--real-supervision", type=Path, default=None,
                    help="also train on real non-test prints labelled by "
                         "scripts/build_real_supervision.py; validation then uses the "
                         "real val split, since real prints are what is being fixed")
    ap.add_argument("--real-mode", choices=["wear", "photometric"], default="wear",
                    help="how real prints are presented: 'wear' damages them further with "
                         "our wear model (arm one's distillation); 'photometric' jitters "
                         "contrast/clarity only (arm two, CDC-style domain alignment) and "
                         "needs a pseudo-annotation cache carrying `evidence`")
    ap.add_argument("--real-repeat", type=int, default=4,
                    help="oversample the ~1k real prints against ~10k Anguli")
    ap.add_argument("--pairs", action="store_true",
                    help="two independently-damaged, pixel-aligned views per item plus a "
                         "consistency loss between their reconstructions -- each item "
                         "then costs two images of VRAM, so halve --batch-size")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from these weights (fresh optimiser), e.g. best.pt")
    args = ap.parse_args()

    samples = [s for s in index_corpus(args.corpus)
               if supervision_path(s, args.supervision).is_file()]
    if not samples:
        print(f"no cached supervision under {args.supervision}; "
              f"run scripts/build_supervision.py first")
        return 1
    by_split = {name: [s for s in samples if s.split == name]
                for name in ("train", "val", "test")}
    if not by_split["train"] or not by_split["val"]:
        print(f"need both train and val samples; have "
              f"{len(by_split['train'])} / {len(by_split['val'])}")
        return 1
    if args.limit_train:
        by_split["train"] = by_split["train"][:args.limit_train]
    if args.limit_val:
        val = by_split["val"]
        by_split["val"] = val[::max(1, len(val) // args.limit_val)][:args.limit_val]
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)

    real = {"train": [], "val": []}
    if args.real_supervision:
        from fpe.data.real import index_real

        for sample, _ in index_real(ROOT / "docs" / "datasets" / "manifests",
                                    ROOT / "data" / "raw"):
            if supervision_path(sample, args.real_supervision).is_file():
                real[sample.split].append(sample)
        if not real["train"] or not real["val"]:
            print(f"no real supervision under {args.real_supervision}; "
                  f"run scripts/build_real_supervision.py first")
            return 1
        print(f"real prints: {len(real['train']):,} train (x{args.real_repeat}), "
              f"{len(real['val']):,} val")

    augment = AugmentConfig(patch=args.patch)
    datasets = {
        "train": WafenDataset(by_split["train"], args.supervision, config=augment,
                              wear=WearConfig(), augment=True, seed=args.seed,
                              pairs=args.pairs),
        # Validation is deterministic: no augmentation, fixed mid severity, so the number
        # moves only when the model does.
        "val": WafenDataset(by_split["val"], args.supervision, config=augment,
                            wear=WearConfig(), augment=False, seed=args.seed,
                            pairs=args.pairs),
    }
    if real["train"]:
        photometric = args.real_mode == "photometric"
        real_train = WafenDataset(real["train"] * args.real_repeat, args.real_supervision,
                                  config=augment, wear=WearConfig(), augment=True,
                                  seed=args.seed, pairs=args.pairs,
                                  photometric=photometric)
        datasets["train"] = MixedDataset([datasets["train"], real_train])
        datasets["val"] = WafenDataset(real["val"], args.real_supervision, config=augment,
                                       wear=WearConfig(), augment=False, seed=args.seed,
                                       pairs=args.pairs, photometric=photometric)
    model_config = WafenConfig()
    device = resolve_device(args.device)
    print(f"device {device}  |  {Wafen(model_config).parameter_count():,} parameters")
    print(f"cached: {len(by_split['train']):,} train, {len(by_split['val']):,} val, "
          f"{len(by_split['test']):,} test")

    if not args.skip_canary:
        print("\ncanary: over-fitting ten samples ...")
        ridge = overfit_check(datasets["train"], samples=10, steps=200,
                              device=args.device)
        verdict = "pass" if ridge < CANARY_THRESHOLD else "FAIL"
        print(f"  ridge loss {ridge:.4f}  ->  {verdict}")
        if ridge >= CANARY_THRESHOLD:
            print("\nThe network cannot memorise ten samples. That is a wiring bug, not a "
                  "data problem -- fix it before spending hours on a full run.")
            return 2
        if args.canary:
            return 0

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, accumulate=args.accumulate,
        learning_rate=args.learning_rate, num_workers=args.workers, seed=args.seed,
        amp=args.amp,
        max_steps_per_epoch=args.max_steps_per_epoch, device=args.device,
    )
    checkpoint = args.out_dir / "checkpoint.pt"
    if not args.no_resume and checkpoint.is_file():
        done = torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"]
        if done >= config.epochs - 1:
            # Re-running a finished run must not append a second registry row for it.
            print(f"already trained {done + 1}/{config.epochs} epochs -> {args.out_dir}")
            return 0
    print(f"\ntraining {config.epochs} epochs, effective batch "
          f"{config.batch_size * config.accumulate}\n")
    model, history = train(
        datasets["train"], datasets["val"], config=config, model_config=model_config,
        out_dir=args.out_dir, resume=not args.no_resume, init_weights=args.init_from,
    )

    best = history.epochs[history.best_epoch] if history.epochs else {}
    print(f"\nbest epoch {history.best_epoch}: validation ridge "
          f"{history.best_val_ridge:.4f}")
    if args.dry_run:
        print("--dry-run: registry not written")
        return 0

    log_run(
        experiment="wafen-train",
        dataset=f"{args.corpus.name} (wear-degraded)"
        + (" + real non-test prints" if real["train"] else ""),
        method="wafen",
        metrics={
            "best_val_ridge": history.best_val_ridge,
            "best_epoch": history.best_epoch,
            "parameters": Wafen(model_config).parameter_count(),
            **{f"val_{k}": v for k, v in best.get("val", {}).items()},
            **{f"train_{k}": v for k, v in best.get("train", {}).items()},
            "n_train": len(by_split["train"]), "n_val": len(by_split["val"]),
            "n_real_train": len(real["train"]), "n_real_val": len(real["val"]),
        },
        config={**config.to_dict(), "model": model_config.to_dict(),
                "patch": args.patch,
                "real_repeat": args.real_repeat if real["train"] else 0,
                "pairs": args.pairs, "real_mode": args.real_mode,
                "init_from": str(args.init_from) if args.init_from else None},
        notes=("Anguli + real FVC2000/FVC2002/U.are.U prints with pyfing-teacher targets; "
               "FVC2004 and CrossMatch held out. Val is the real val split."
               if real["train"] else
               "Trained on synthetic Anguli only; no real image appears in training."),
    )
    print("registry row appended -> results/registry/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
