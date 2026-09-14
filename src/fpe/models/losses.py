"""Per-head losses for WAFEN (plan U6).

Five heads, five objectives. Two of them encode arguments the thesis actually makes, and
are worth reading rather than skimming:

**The ridge loss is asymmetric on purpose.** It is the Tversky index with alpha = 0.7, so
a false ridge costs about 2.3 times a missed one. This is not a tuning choice. A spurious
ridge produces a spurious *minutia*, and a spurious minutia is a false identity feature --
in a civil identity system that is actively dangerous in a way a missing feature is not.
Precision is the binding constraint in this literature, and the loss is where that belief
has to be spent or it is only a claim.

**The orientation loss respects the wrap.** Ridge orientation has no head or tail, so
theta and theta + pi are the same ridge. The loss is one minus the cosine of the doubled
angular difference, which is zero when the prediction is right *or* right-plus-pi, and
maximal at pi/2 -- genuinely perpendicular. A plain regression on raw angles would charge
178 units for answering 179 degrees when the truth is 1.

Every loss is restricted to the foreground. Outside the mask there is no ridge to get right
or wrong, and averaging over background would let a model score well by predicting nothing
almost everywhere.

Head weighting is deliberately left as fixed scalars rather than learned uncertainty
weighting. It is the simpler falsifiable choice; reach for something cleverer only if a
head visibly fails to learn, which per-head logging in training will show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

__all__ = ["tversky_loss", "orientation_loss", "masked_l1", "segmentation_loss",
           "LossWeights", "WafenLoss"]

EPS = 1e-6


def _prepare_mask(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(like)
    if mask.shape != like.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(like.shape)}")
    return mask.to(like.dtype)


def tversky_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Asymmetric overlap loss on the ridge map. 0 is perfect, 1 is fully inverted.

    `alpha` weights false ridges; `1 - alpha` weights missed ridges. At the default, a
    false ridge costs alpha / (1 - alpha) = 2.33x a missed one.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    weights = _prepare_mask(mask, prediction)
    p = prediction * weights
    t = target * weights

    true_ridge = (p * t).sum()
    false_ridge = (p * (1 - t) * weights).sum()      # predicted ridge, no ridge there
    missed_ridge = ((1 - p) * t * weights).sum()     # ridge there, not predicted
    index = (true_ridge + EPS) / (
        true_ridge + alpha * false_ridge + (1 - alpha) * missed_ridge + EPS
    )
    return 1.0 - index


def orientation_loss(
    prediction: torch.Tensor, target_angle: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Angular loss on the double-angle representation.

    `prediction` is (B, 2, H, W) holding (cos 2t, sin 2t); `target_angle` is (B, 1, H, W)
    in radians. Zero when the prediction matches, or matches plus pi.
    """
    if prediction.shape[1] != 2:
        raise ValueError(f"expected 2 channels, got {prediction.shape[1]}")
    weights = _prepare_mask(mask, target_angle)
    cos_t = torch.cos(2 * target_angle)
    sin_t = torch.sin(2 * target_angle)
    agreement = prediction[:, 0:1] * cos_t + prediction[:, 1:2] * sin_t
    return ((1.0 - agreement) * weights).sum() / weights.sum().clamp_min(EPS)


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean absolute error inside the mask. Used for frequency and confidence."""
    weights = _prepare_mask(mask, prediction)
    return ((prediction - target).abs() * weights).sum() / weights.sum().clamp_min(EPS)


def segmentation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over the whole image -- the mask is what this head predicts."""
    p = prediction.clamp(EPS, 1.0 - EPS)
    return -(target * p.log() + (1 - target) * (1 - p).log()).mean()


@dataclass
class LossWeights:
    """Fixed weights, chosen to bring each term to a comparable scale.

    Ridge carries the most weight because it is the head the thesis's claims rest on;
    confidence carries real weight because it is the head nothing else can train.
    """

    ridge: float = 1.0
    confidence: float = 1.0
    orientation: float = 0.5
    period: float = 0.1
    """Period is in pixels, roughly an order of magnitude larger than the other terms."""
    segmentation: float = 0.5


class WafenLoss(nn.Module):
    """The combined objective, returning the total **and** every component.

    Per-head values are returned rather than only the sum because a head silently failing
    to learn is the most likely way the multi-task design fails, and an aggregate hides it
    completely.
    """

    def __init__(self, weights: LossWeights | None = None, alpha: float = 0.7) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.alpha = alpha

    def forward(self, output, targets: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = targets["mask"]
        parts = {
            "ridge": tversky_loss(output.ridge, targets["ridge"], mask, self.alpha),
            "confidence": masked_l1(output.confidence, targets["evidence"], mask),
            "orientation": orientation_loss(output.orientation, targets["orientation"],
                                            mask),
            "period": masked_l1(output.period, targets["period"], mask),
            "segmentation": segmentation_loss(output.segmentation, mask),
        }
        total = sum(getattr(self.weights, name) * value for name, value in parts.items())
        return total, {name: float(value.detach()) for name, value in parts.items()}
