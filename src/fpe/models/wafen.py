"""WAFEN -- the wear-aware network (plan U5).

One encoder-decoder that estimates, in a single forward pass, everything the current state
of the art computes with four separate networks: segmentation, ridge orientation, ridge
frequency, an enhanced ridge map, and -- the head nobody else can train -- a per-pixel
confidence.

**The architecture is deliberately unremarkable.** The literature this thesis builds on is
explicit that architectural novelty is where returns have gone: a GAN trained on 130,000
images loses to a five-level encoder-decoder trained on 360, and the gains that did arrive
came from better orientation and frequency estimation rather than bigger enhancers. So the
shape here is SNFEN's shape. What is new is *what it estimates* and *what supervises it*.

Three design points worth stating, because each is a place where a reasonable person would
have done something else:

- **One input channel.** SNFEN takes five -- image, mask, two orientation channels and
  frequency -- because three upstream networks produce them first. WAFEN takes the image
  alone and produces those estimates itself, which is what collapses four forward passes
  into one. Measured on this machine, the four-network chain costs ~1.05 s per image on
  CPU against a 500 ms deployment budget.
- **Orientation is two channels, not one.** Ridge orientation wraps at pi and has no
  head or tail, so a network predicting a raw angle is punished enormously for answering
  179 degrees when the truth is 1. The head predicts (cos 2t, sin 2t) -- a point on a
  circle -- where those two answers are neighbours and the penalty matches the error.
- **The parameter budget is enforced in code.** Edge deployability is a stated thesis
  goal, so a configuration that exceeds it is out of scope by definition and the model
  refuses to build rather than quietly becoming a result that cannot be claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

__all__ = ["WafenConfig", "Wafen", "WafenOutput", "decode_orientation",
           "estimate_parameters"]

MAX_PARAMETERS = 10_000_000
"""Hard budget from the thesis constraints. Exceeding it is not a result we can report."""


@dataclass(frozen=True)
class WafenConfig:
    """Shape of the network. Defaults land near 5.5M parameters, close to SNFEN's 4.9M."""

    in_channels: int = 1
    base_channels: int = 16
    depth: int = 5
    """Encoder levels. Four poolings, so the bottleneck sees 1/16 resolution."""
    head_channels: int = 16
    kernel_size: int = 5
    max_parameters: int = MAX_PARAMETERS

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WafenOutput:
    """The five heads. Every tensor is at input resolution.

    `segmentation`, `ridge` and `confidence` are probabilities in [0,1]. `orientation` is
    the double-angle pair; use `decode_orientation` for radians. `period` is in pixels.
    """

    segmentation: torch.Tensor   # (B, 1, H, W)
    orientation: torch.Tensor    # (B, 2, H, W) -- cos 2t, sin 2t
    period: torch.Tensor         # (B, 1, H, W)
    ridge: torch.Tensor          # (B, 1, H, W)
    confidence: torch.Tensor     # (B, 1, H, W)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "segmentation": self.segmentation, "orientation": self.orientation,
            "period": self.period, "ridge": self.ridge, "confidence": self.confidence,
        }


def decode_orientation(orientation: torch.Tensor) -> torch.Tensor:
    """(cos 2t, sin 2t) -> ridge angle in radians, in [-pi/2, pi/2)."""
    if orientation.shape[1] != 2:
        raise ValueError(f"expected 2 orientation channels, got {orientation.shape[1]}")
    return 0.5 * torch.atan2(orientation[:, 1:2], orientation[:, 0:1])


def estimate_parameters(cfg: WafenConfig) -> int:
    """Parameter count from the config alone, without building anything.

    Convolution weights dominate -- batch-norm terms are a rounding error at these widths
    -- so this is a slight under-estimate, which is why the exact count is still checked
    after construction.
    """
    k = cfg.kernel_size ** 2
    widths = [cfg.base_channels * 2 ** i for i in range(cfg.depth)]
    total = 0
    previous = cfg.in_channels
    for width in widths:                       # encoder: two convolutions per level
        total += k * previous * width + k * width * width
        previous = width
    for level in reversed(range(cfg.depth - 1)):   # decoder, with skip concatenation
        fan_in = widths[level + 1] + widths[level]
        total += k * fan_in * widths[level] + k * widths[level] * widths[level]
    total += k * widths[0] * cfg.head_channels + k * cfg.head_channels ** 2
    total += 6 * cfg.head_channels             # five heads, one of them two channels
    return total


def _block(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Wafen(nn.Module):
    """Shared encoder-decoder with five heads."""

    def __init__(self, config: WafenConfig | None = None) -> None:
        super().__init__()
        self.config = config or WafenConfig()
        cfg = self.config
        if cfg.depth < 2:
            raise ValueError(f"depth must be at least 2, got {cfg.depth}")

        widths = [cfg.base_channels * 2 ** i for i in range(cfg.depth)]

        # Check the budget *before* allocating. A wildly over-sized configuration would
        # otherwise exhaust memory during construction and surface as an allocation
        # failure rather than as the scope decision it actually is.
        estimate = estimate_parameters(cfg)
        if estimate > cfg.max_parameters:
            raise ValueError(
                f"configuration needs about {estimate:,} parameters, over the "
                f"{cfg.max_parameters:,} budget. Edge deployability is a stated thesis "
                f"goal, so this is out of scope rather than merely large -- reduce "
                f"base_channels or depth."
            )

        self.encoders = nn.ModuleList()
        previous = cfg.in_channels
        for width in widths:
            self.encoders.append(_block(previous, width, cfg.kernel_size))
            previous = width
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder mirrors the encoder; each level concatenates its skip connection, which
        # is what returns the fine detail that pooling discarded.
        self.decoders = nn.ModuleList()
        for level in reversed(range(cfg.depth - 1)):
            self.decoders.append(
                _block(widths[level + 1] + widths[level], widths[level], cfg.kernel_size)
            )

        self.head = _block(widths[0], cfg.head_channels, cfg.kernel_size)
        self.segmentation = nn.Conv2d(cfg.head_channels, 1, 1)
        self.orientation = nn.Conv2d(cfg.head_channels, 2, 1)
        self.period = nn.Conv2d(cfg.head_channels, 1, 1)
        self.ridge = nn.Conv2d(cfg.head_channels, 1, 1)
        self.confidence = nn.Conv2d(cfg.head_channels, 1, 1)

        total = self.parameter_count()
        if total > cfg.max_parameters:  # exact count, in case the estimate was optimistic
            raise ValueError(
                f"{total:,} parameters exceeds the {cfg.max_parameters:,} budget."
            )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> WafenOutput:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        stride = 2 ** (self.config.depth - 1)
        if x.shape[-1] % stride or x.shape[-2] % stride:
            raise ValueError(
                f"input {tuple(x.shape[-2:])} must be divisible by {stride} "
                f"for depth {self.config.depth}; pad or crop before the forward pass"
            )

        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat([x, skip], dim=1))

        features = self.head(x)
        orientation = self.orientation(features)
        # Project onto the unit circle: only the direction carries meaning, and leaving
        # the magnitude free lets the head trade angular accuracy for vector length.
        orientation = orientation / orientation.norm(dim=1, keepdim=True).clamp_min(1e-6)

        return WafenOutput(
            segmentation=torch.sigmoid(self.segmentation(features)),
            orientation=orientation,
            period=self.period(features),
            ridge=torch.sigmoid(self.ridge(features)),
            confidence=torch.sigmoid(self.confidence(features)),
        )
