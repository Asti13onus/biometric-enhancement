"""Latent-style degradation -- the baseline arm the thesis argues against.

This implements the artefact list ChaLearn published for its Track 3 degraded images:
blur, brightness, contrast, elastic transformation, occlusion, scratches, resolution
reduction, rotation, and compositing onto background textures.

Why this exists in our own code rather than as a download: ChaLearn registration is
broken (ADR 0003), and Anguli's own flags cover only about four of the nine artefacts and
cannot composite backgrounds at all. Keeping every artefact in one auditable place also
matters more than usual for this thesis, whose central claim is *about* degradation models
-- "what exactly was applied" has to be inspectable.

**This models a crime-scene latent print, not a worn finger.** Ridge signal is present but
occluded. That is the point: it is the control condition against which the wear-degradation
model (Contribution 1, `fpe.degradation.wear`) is measured. Do not extend this toward wear
physiology -- add to the wear model instead.

Fidelity caveat to state in the thesis: ChaLearn published the artefact *types* but not
their parameter distributions, so severity ranges here are our own, chosen to span visually
comparable degradation. Exact numerical comparability with published ChaLearn results is
not claimed (ADR 0003).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .backgrounds import BackgroundBank


@dataclass(frozen=True)
class LatentConfig:
    """Severity ranges for each artefact. Each is sampled per image.

    `p_*` values are the probability the artefact is applied at all, so a degraded image
    receives a random subset rather than all nine every time.
    """

    # geometry
    rotation_deg: tuple[float, float] = (-15.0, 15.0)
    translation_frac: tuple[float, float] = (0.0, 0.04)
    p_elastic: float = 0.5
    elastic_alpha: tuple[float, float] = (8.0, 24.0)   # displacement magnitude, px
    elastic_sigma: tuple[float, float] = (6.0, 12.0)   # smoothness of the field

    # photometric
    p_brightness: float = 0.7
    brightness_delta: tuple[float, float] = (-0.20, 0.20)
    p_contrast: float = 0.7
    contrast_gain: tuple[float, float] = (0.55, 1.35)

    # acquisition
    p_blur: float = 0.7
    blur_sigma: tuple[float, float] = (0.6, 2.2)
    p_resolution: float = 0.4
    resolution_scale: tuple[float, float] = (0.35, 0.8)  # downsample then back up
    p_noise: float = 0.6
    noise_sigma: tuple[float, float] = (0.01, 0.06)

    # structural damage
    p_scratches: float = 0.7
    n_scratches: tuple[int, int] = (2, 14)
    scratch_width: tuple[int, int] = (1, 3)
    p_occlusion: float = 0.5
    n_occlusions: tuple[int, int] = (1, 4)
    occlusion_frac: tuple[float, float] = (0.03, 0.16)  # of image area, per blob

    # background compositing
    p_background: float = 0.9
    background_weight: tuple[float, float] = (0.25, 0.7)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DegradationRecord:
    """What was actually applied to one image, so any sample can be explained later."""

    applied: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def note(self, name: str, **params) -> None:
        self.applied.append(name)
        self.params[name] = params


def _u(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def _i(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _elastic_field(shape, alpha, sigma, rng):
    h, w = shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    norm = max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    dx, dy = dx / norm * alpha, dy / norm * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return (xx + dx).astype(np.float32), (yy + dy).astype(np.float32)


class LatentDegradation:
    """Applies the ChaLearn artefact list. Seeded, and records what it did."""

    def __init__(self, config: LatentConfig | None = None,
                 backgrounds: BackgroundBank | None = None):
        self.config = config or LatentConfig()
        self.backgrounds = backgrounds

    def __call__(self, image: np.ndarray, seed: int) -> tuple[np.ndarray, DegradationRecord]:
        """`image`: greyscale, any dtype. Returns (degraded float32 [0,1], record)."""
        cfg = self.config
        rng = np.random.default_rng(seed)
        rec = DegradationRecord()

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = np.clip(img, 0.0, 1.0)
        h, w = img.shape[:2]

        # -- geometry: rotation + translation, in one affine warp -------------------
        angle = _u(rng, cfg.rotation_deg)
        tx = _u(rng, cfg.translation_frac) * w * rng.choice([-1, 1])
        ty = _u(rng, cfg.translation_frac) * h * rng.choice([-1, 1])
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        # border = white: unrecorded area reads as "no ridge", not as black ink
        img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        rec.note("rotation_translation", angle_deg=angle, tx=float(tx), ty=float(ty))

        if rng.random() < cfg.p_elastic:
            alpha, sigma = _u(rng, cfg.elastic_alpha), _u(rng, cfg.elastic_sigma)
            map_x, map_y = _elastic_field((h, w), alpha, sigma, rng)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
            rec.note("elastic", alpha=alpha, sigma=sigma)

        # -- structural damage, before photometrics so damage gets blurred too ------
        if rng.random() < cfg.p_scratches:
            n = _i(rng, cfg.n_scratches)
            overlay = np.zeros((h, w), np.float32)
            for _ in range(n):
                p0 = (int(rng.integers(w)), int(rng.integers(h)))
                length = rng.uniform(0.1, 0.6) * max(h, w)
                theta = rng.uniform(0, 2 * np.pi)
                p1 = (int(p0[0] + length * np.cos(theta)),
                      int(p0[1] + length * np.sin(theta)))
                cv2.line(overlay, p0, p1, 1.0, _i(rng, cfg.scratch_width), cv2.LINE_AA)
            overlay = cv2.GaussianBlur(overlay, (0, 0), 0.6)
            # scratches are bright: the lifting medium removes ink
            img = np.clip(img + overlay * rng.uniform(0.4, 0.9), 0.0, 1.0)
            rec.note("scratches", count=n)

        if rng.random() < cfg.p_occlusion:
            n = _i(rng, cfg.n_occlusions)
            mask = np.zeros((h, w), np.float32)
            for _ in range(n):
                area = _u(rng, cfg.occlusion_frac) * h * w
                radius = max(3, int(np.sqrt(area / np.pi)))
                centre = (int(rng.integers(w)), int(rng.integers(h)))
                axes = (radius, max(3, int(radius * rng.uniform(0.5, 1.5))))
                cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)),
                            0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, radius * 0.15))
            # occluders both hide and smear -- blend toward a flat local level
            level = float(rng.uniform(0.35, 0.95))
            img = img * (1 - mask) + level * mask
            rec.note("occlusion", count=n)

        # -- background compositing ------------------------------------------------
        if self.backgrounds is not None and rng.random() < cfg.p_background:
            bg = self.backgrounds.sample(rng, (h, w))
            weight = _u(rng, cfg.background_weight)
            # Multiplicative ("darken") compositing: the texture attenuates the print
            # the way a surface shows through a lifted latent, rather than averaging
            # into it, which would wash ridges out uniformly.
            img = np.clip(img * (1 - weight) + img * bg * weight, 0.0, 1.0)
            rec.note("background", weight=weight, split=self.backgrounds.split)

        # -- photometric -----------------------------------------------------------
        if rng.random() < cfg.p_contrast:
            gain = _u(rng, cfg.contrast_gain)
            img = np.clip((img - 0.5) * gain + 0.5, 0.0, 1.0)
            rec.note("contrast", gain=gain)

        if rng.random() < cfg.p_brightness:
            delta = _u(rng, cfg.brightness_delta)
            img = np.clip(img + delta, 0.0, 1.0)
            rec.note("brightness", delta=delta)

        # -- acquisition -----------------------------------------------------------
        if rng.random() < cfg.p_blur:
            sigma = _u(rng, cfg.blur_sigma)
            img = cv2.GaussianBlur(img, (0, 0), sigma)
            rec.note("blur", sigma=sigma)

        if rng.random() < cfg.p_resolution:
            scale = _u(rng, cfg.resolution_scale)
            small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            rec.note("resolution", scale=scale)

        if rng.random() < cfg.p_noise:
            sigma = _u(rng, cfg.noise_sigma)
            img = np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0.0, 1.0)
            rec.note("noise", sigma=sigma)

        return img, rec
