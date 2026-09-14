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
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low; 7.8 GB of RAM binds before cores do")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canary", action="store_true", help="run the canary and exit")
    ap.add_argument("--skip-canary", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="do not write the registry")
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

    augment = AugmentConfig(patch=args.patch)
    datasets = {
        "train": WafenDataset(by_split["train"], args.supervision, config=augment,
                              wear=WearConfig(), augment=True, seed=args.seed),
        # Validation is deterministic: no augmentation, fixed mid severity, so the number
        # moves only when the model does.
        "val": WafenDataset(by_split["val"], args.supervision, config=augment,
                            wear=WearConfig(), augment=False, seed=args.seed),
    }
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
        max_steps_per_epoch=args.max_steps_per_epoch, device=args.device,
    )
    print(f"\ntraining {config.epochs} epochs, effective batch "
          f"{config.batch_size * config.accumulate}\n")
    model, history = train(
        datasets["train"], datasets["val"], config=config, model_config=model_config,
        out_dir=args.out_dir, resume=not args.no_resume,
    )

    best = history.epochs[history.best_epoch] if history.epochs else {}
    print(f"\nbest epoch {history.best_epoch}: validation ridge "
          f"{history.best_val_ridge:.4f}")
    if args.dry_run:
        print("--dry-run: registry not written")
        return 0

    log_run(
        experiment="wafen-train",
        dataset=f"{args.corpus.name} (wear-degraded)",
        method="wafen",
        metrics={
            "best_val_ridge": history.best_val_ridge,
            "best_epoch": history.best_epoch,
            "parameters": Wafen(model_config).parameter_count(),
            **{f"val_{k}": v for k, v in best.get("val", {}).items()},
            **{f"train_{k}": v for k, v in best.get("train", {}).items()},
            "n_train": len(by_split["train"]), "n_val": len(by_split["val"]),
        },
        config={**config.to_dict(), "model": model_config.to_dict(),
                "patch": args.patch},
        notes="Trained on synthetic Anguli only; no real image appears in training.",
    )
    print("registry row appended -> results/registry/runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
