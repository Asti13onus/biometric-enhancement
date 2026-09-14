"""Training dataset: cached targets in, degraded patches out (plan U4).

One item is one impression. Loading is lazy and per item -- this machine has 7.8 GB of RAM
and the corpus is ~5 GB of cached targets, so nothing is bulk-loaded and dataloader workers
stay at 2.

Degradation is applied **on the fly** rather than cached. The wear model is cheap, and
regenerating it each epoch means the network sees fresh damage on every pass instead of
memorising one fixed corruption per finger. Severity is resampled per item too, so a single
training run covers the whole severity range rather than one point on it.

## The thing most likely to be silently wrong

Geometric augmentation has to transform the *orientation values*, not merely move the
pixels holding them. Rotating an image by phi rotates every ridge by phi, so the stored
angles must have phi added; a horizontal flip mirrors every ridge, so the angles must be
negated. Resampling the orientation map without doing that produces targets that look
perfectly plausible -- correct shape, correct range, spatially aligned -- and are wrong
everywhere. The model would train happily and never learn orientation.

That is why the augmentation here is tested end-to-end against a structure tensor
recomputed on the rotated image, rather than against the bookkeeping's own arithmetic.

Order of operations: geometry first on the clean impression, then degradation, then
photometric jitter, then the crop. Degrading before cropping matters because contact
pressure falls off across the *whole finger* -- cropping first would give every patch its
own little contact centre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from fpe.data.anguli import AnguliSample, supervision_path
from fpe.data.convert import load_greyscale
from fpe.degradation.wear import WearConfig, WearDegradation

__all__ = ["AugmentConfig", "WafenDataset", "rotate_orientation"]

TARGET_KEYS = ("mask", "ridge", "orientation", "period", "period_valid", "evidence")

PERIOD_RANGE = (3.0, 25.0)
"""Plausible ridge periods in pixels at 500 dpi.

The frequency teacher emits values outside this -- negative periods appear in regions it
could not estimate, and a negative ridge spacing is not a hard target, it is a marker.
Those pixels are excluded from the period loss rather than clamped, because clamping would
invent a number and train the network to reproduce it.
"""


@dataclass(frozen=True)
class AugmentConfig:
    """Fingerprint-specific augmentation, following Cappelli's published recipe."""

    patch: int = 256
    rotation_deg: tuple[float, float] = (-20.0, 20.0)
    scale: tuple[float, float] = (0.85, 1.15)
    translate_frac: float = 0.05
    p_hflip: float = 0.5
    gamma: tuple[float, float] = (0.7, 1.4)
    severity: tuple[float, float] = (0.0, 1.0)
    foreground_tries: int = 24
    """Attempts to land a patch on the finger before accepting whatever came up."""


def rotate_orientation(angles: np.ndarray, radians: float) -> np.ndarray:
    """Add a rotation to an orientation field, wrapped into [-pi/2, pi/2).

    Ridge orientation is defined modulo pi, so this wraps at pi and not 2pi.
    """
    shifted = angles + radians
    return ((shifted + np.pi / 2) % np.pi - np.pi / 2).astype(np.float32)


class WafenDataset(Dataset):
    """Yields (image, targets) with degradation and augmentation applied per item."""

    def __init__(
        self,
        samples: list[AnguliSample],
        supervision_root: str | Path,
        *,
        config: AugmentConfig | None = None,
        wear: WearConfig | None = None,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("no samples")
        self.samples = samples
        self.root = Path(supervision_root)
        self.config = config or AugmentConfig()
        self.degrade = WearDegradation(wear or WearConfig())
        self.augment = augment
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the per-item seeds so a new epoch sees fresh damage."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, index))

    def _load(self, sample: AnguliSample) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        path = supervision_path(sample, self.root)
        if not path.is_file():
            raise FileNotFoundError(
                f"no cached supervision for {sample.key} at {path}; "
                f"run scripts/build_supervision.py"
            )
        with np.load(path, allow_pickle=False) as data:
            mask = data["mask"].astype(np.float32)
            orientation = data["orientation"].astype(np.float32)
            period = data["period"].astype(np.float32)
        image = load_greyscale(sample.image).astype(np.float32) / 255.0
        ridge = (load_greyscale(sample.master).astype(np.float32) / 255.0)
        # Anguli masters are white background with black ridges; the head predicts ridge
        # presence, so invert to make 1 mean "ridge here".
        ridge = 1.0 - ridge
        if not (image.shape == mask.shape == orientation.shape == ridge.shape):
            raise ValueError(
                f"{sample.key}: shapes disagree -- image {image.shape}, mask "
                f"{mask.shape}, orientation {orientation.shape}, ridge {ridge.shape}"
            )
        low, high = PERIOD_RANGE
        period_valid = ((period >= low) & (period <= high)).astype(np.float32)
        return image, {"mask": mask, "ridge": ridge, "orientation": orientation,
                       "period": period, "period_valid": period_valid}

    def _geometric(self, image, targets, rng):
        """Rotate, scale, translate and flip -- image and every target together."""
        cfg = self.config
        h, w = image.shape
        angle = float(rng.uniform(*cfg.rotation_deg))
        scale = float(rng.uniform(*cfg.scale))
        tx = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac)) * w
        ty = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac)) * h

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        def warp(array, border, interpolation):
            return cv2.warpAffine(array, matrix, (w, h), flags=interpolation,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border)

        # Background reads as "no ridge", not as ink.
        image = warp(image, 1.0, cv2.INTER_LINEAR)
        out = {
            "mask": warp(targets["mask"], 0.0, cv2.INTER_NEAREST),
            "ridge": warp(targets["ridge"], 0.0, cv2.INTER_NEAREST),
            "period": warp(targets["period"], 0.0, cv2.INTER_LINEAR),
            "period_valid": warp(targets["period_valid"], 0.0, cv2.INTER_NEAREST),
        }
        # cv2 rotates counter-clockwise on screen while image y runs downward, so a
        # positive angle turns ridges by -angle in the usual orientation convention.
        out["orientation"] = rotate_orientation(
            warp(targets["orientation"], 0.0, cv2.INTER_NEAREST), -np.deg2rad(angle)
        )

        if rng.random() < cfg.p_hflip:
            image = np.ascontiguousarray(image[:, ::-1])
            for key in out:
                out[key] = np.ascontiguousarray(out[key][:, ::-1])
            out["orientation"] = rotate_orientation(-out["orientation"], 0.0)
        return image, out

    def _crop(self, image, targets, evidence, rng):
        """A patch that actually contains finger, where one can be found."""
        size = self.config.patch
        h, w = image.shape
        if h < size or w < size:
            pad_y, pad_x = max(0, size - h), max(0, size - w)
            pad = ((0, pad_y), (0, pad_x))
            image = np.pad(image, pad, constant_values=1.0)
            evidence = np.pad(evidence, pad, constant_values=1.0)
            targets = {k: np.pad(v, pad, constant_values=0.0) for k, v in targets.items()}
            h, w = image.shape

        best = None
        for _ in range(self.config.foreground_tries):
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            coverage = float(targets["mask"][y:y + size, x:x + size].mean())
            if best is None or coverage > best[0]:
                best = (coverage, y, x)
            if coverage > 0.5:
                break
        _, y, x = best
        window = (slice(y, y + size), slice(x, x + size))
        return (image[window], {k: v[window] for k, v in targets.items()},
                evidence[window])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[index]
        rng = self._rng(index)
        image, targets = self._load(sample)

        if self.augment:
            image, targets = self._geometric(image, targets, rng)

        severity = float(rng.uniform(*self.config.severity)) if self.augment else 0.5
        # Reuse the teacher's orientation rather than letting the wear model re-derive a
        # rough one: better grounded, and it removes a structure tensor per item.
        result = self.degrade(image, seed=int(rng.integers(0, 2**31 - 1)),
                              severity=severity, mask=targets["mask"],
                              orientation=targets["orientation"])
        degraded, evidence = result.image, result.evidence

        if self.augment:
            gamma = float(rng.uniform(*self.config.gamma))
            degraded = np.clip(degraded, 1e-6, 1.0) ** gamma

        degraded, targets, evidence = self._crop(degraded, targets, evidence, rng)
        targets["evidence"] = evidence

        tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0)
                   for k, v in targets.items()}
        return torch.from_numpy(np.ascontiguousarray(degraded)).float().unsqueeze(0), tensors
