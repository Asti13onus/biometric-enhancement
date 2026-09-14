"""Training loop for WAFEN (plan U7).

Deliberately economical, following the precedent this thesis builds on: the state of the
art trained for 25 minutes on 360 images. Long training is not where the returns are here,
and a 4 GB GPU makes it expensive to pretend otherwise.

Three things this does that a minimal loop would not, each because of a specific way the
work can fail quietly:

- **Per-head losses are logged separately.** A head that silently stops learning is the
  most likely way a five-headed design fails, and a summed loss hides it completely. The
  history carries every component for every epoch.
- **The over-fit canary runs first.** `overfit_check` trains on a handful of samples until
  the loss collapses. If the network cannot memorise ten images, no quantity of data will
  help, and finding that out takes a minute instead of an hour.
- **Checkpoints carry the optimiser and scheduler state**, so a run killed on this machine
  -- which has form -- resumes rather than restarts.

Device handling is deliberate rather than automatic: mixed precision and gradient
accumulation exist to fit a 4 GB card, and both are no-ops on CPU so the same code path is
exercised by the tests.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fpe.models.losses import LossWeights, WafenLoss
from fpe.models.wafen import Wafen, WafenConfig

__all__ = ["TrainConfig", "train", "overfit_check", "evaluate"]


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 8
    accumulate: int = 2
    """Effective batch is batch_size * accumulate; raises it without raising memory."""
    learning_rate: float = 3e-4
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-4
    amp: bool = True
    """Mixed precision. Ignored on CPU, where autocast to fp16 is not a saving."""
    num_workers: int = 2
    """Never raise this on a 7.8 GB machine; RAM binds before cores do."""
    seed: int = 0
    max_steps_per_epoch: int | None = None
    device: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class History:
    epochs: list[dict] = field(default_factory=list)
    best_val_ridge: float = math.inf
    best_epoch: int = -1

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "best_val_ridge": self.best_val_ridge,
                "best_epoch": self.best_epoch}


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _schedule(step: int, total: int, warmup: int) -> float:
    """Linear warm-up into cosine decay, as a multiplier on the base learning rate."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _to_device(batch, device):
    image, targets = batch
    return image.to(device), {k: v.to(device) for k, v in targets.items()}


@torch.no_grad()
def evaluate(model: Wafen, loader: DataLoader, criterion: WafenLoss,
             device: torch.device) -> dict[str, float]:
    """Mean loss per head over a loader. Returns an empty dict for an empty loader."""
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        image, targets = _to_device(batch, device)
        _, parts = criterion(model(image), targets)
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    model.train()
    return {k: v / batches for k, v in totals.items()} if batches else {}


def overfit_check(
    dataset, *, samples: int = 10, steps: int = 150, device: str | None = None,
    model_config: WafenConfig | None = None, learning_rate: float = 1e-3,
) -> float:
    """Train on a handful of items until the loss collapses; return the final ridge loss.

    The cheapest possible check that the wiring is right. A network that cannot memorise
    ten images has a bug, not a data problem, and this finds it in a minute.
    """
    torch.manual_seed(0)
    device = resolve_device(device)
    model = Wafen(model_config or WafenConfig()).to(device).train()
    criterion = WafenLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    batch = [dataset[i] for i in range(min(samples, len(dataset)))]
    images = torch.stack([b[0] for b in batch]).to(device)
    targets = {k: torch.stack([b[1][k] for b in batch]).to(device) for k in batch[0][1]}

    ridge = math.inf
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        total, parts = criterion(model(images), targets)
        total.backward()
        optimiser.step()
        ridge = parts["ridge"]
    return ridge


def train(
    train_dataset, val_dataset, *, config: TrainConfig | None = None,
    model_config: WafenConfig | None = None, weights: LossWeights | None = None,
    out_dir: str | Path = "data/work/wafen", resume: bool = True,
    progress=print,
) -> tuple[Wafen, History]:
    """Train WAFEN. Returns the model with the best validation ridge loss loaded."""
    config = config or TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    best_path = out / "best.pt"

    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model = Wafen(model_config or WafenConfig()).to(device)
    criterion = WafenLoss(weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, drop_last=True,
                            persistent_workers=config.num_workers > 0),
        "val": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers),
    }
    steps_per_epoch = len(loaders["train"])
    if config.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * config.epochs // max(config.accumulate, 1))
    warmup = int(total_steps * config.warmup_fraction)

    history = History()
    start_epoch = 0
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        history = History(**state["history"])
        start_epoch = state["epoch"] + 1
        progress(f"resumed from epoch {state['epoch']} ({checkpoint_path})")

    step = start_epoch * steps_per_epoch // max(config.accumulate, 1)
    for epoch in range(start_epoch, config.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        started = time.perf_counter()
        running: dict[str, float] = {}
        seen = 0

        optimiser.zero_grad(set_to_none=True)
        for index, batch in enumerate(loaders["train"]):
            if index >= steps_per_epoch:
                break
            image, targets = _to_device(batch, device)
            with torch.autocast("cuda", enabled=use_amp):
                total, parts = criterion(model(image), targets)
            scaler.scale(total / config.accumulate).backward()

            if (index + 1) % config.accumulate == 0:
                for group in optimiser.param_groups:
                    group["lr"] = config.learning_rate * _schedule(step, total_steps, warmup)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                step += 1

            for name, value in parts.items():
                running[name] = running.get(name, 0.0) + value
            seen += 1

        train_losses = {k: v / max(seen, 1) for k, v in running.items()}
        val_losses = evaluate(model, loaders["val"], criterion, device)
        record = {
            "epoch": epoch, "seconds": time.perf_counter() - started,
            "train": train_losses, "val": val_losses,
            "lr": optimiser.param_groups[0]["lr"],
        }
        history.epochs.append(record)
        progress(
            f"  epoch {epoch:3d}  ridge {train_losses.get('ridge', float('nan')):.4f}"
            f" / {val_losses.get('ridge', float('nan')):.4f}"
            f"  conf {train_losses.get('confidence', float('nan')):.4f}"
            f"  ori {train_losses.get('orientation', float('nan')):.4f}"
            f"  ({record['seconds']:.0f}s)"
        )

        val_ridge = val_losses.get("ridge", math.inf)
        if val_ridge < history.best_val_ridge:
            history.best_val_ridge = val_ridge
            history.best_epoch = epoch
            torch.save({"model": model.state_dict(),
                        "config": (model_config or WafenConfig()).to_dict()}, best_path)

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
            "history": history.to_dict(),
        }, checkpoint_path)
        (out / "history.json").write_text(json.dumps(history.to_dict(), indent=2),
                                          encoding="utf-8")

    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model"])
    return model, history
