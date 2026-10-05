"""Wear degradation -- the thesis's own degradation model (Contribution 1).

Occupational and age-related wear is physically different from crime-scene latent
degradation, which is what essentially every supervised enhancement method trains on.
A latent print has intact ridge signal that is *occluded*; a worn finger has ridge signal
that is *attenuated or absent*. This module models the second, from four mechanisms
reported in the wear and ageing literature:

1. **Ridge amplitude attenuation.** Abrasion and keratin loss flatten the ridge profile,
   so the ridge-to-valley contrast compresses toward the local mean while ridge *position*
   survives. This is the mechanism that makes worn prints hard: the structure is still
   there, just faint.
2. **Flexion creases.** Persistent skin folds cut across the ridge flow and read as
   permanent linear valleys. Unlike a scratch on a latent, a crease removes ridge
   information rather than adding a bright artefact over it.
3. **Dryness-induced fragmentation.** Dry skin transfers intermittently, breaking each
   ridge into dashes. The breaks are correlated **along** the ridge, not isotropic --
   modelling them with round noise blobs is the common shortcut and it is wrong.
4. **Pressure-dependent partial contact.** Worn, flattened fingertips make poor contact
   at the periphery, so signal falls off away from the core.

The contrast with `fpe.degradation.latent` is the thesis's central experiment, so the two
modules are deliberately parallel in shape: same config-dataclass style, same seeding
contract, same per-image record of what was applied.

**The evidence map is what makes this module more than a degradation model.** Because we
synthesise the damage, we know per pixel how much ridge evidence survived, and that array
is the supervision target for the network's confidence head. No method that inherits its
degradation from someone else can produce it. It is computed analytically from the effects
applied, never by comparing the before and after images -- an image difference would
conflate "evidence destroyed" with "pixel changed", and attenuation changes many pixels
while destroying little evidence.

Severity is a single scalar in [0, 1] scaling all four mechanisms, so a controlled severity
sweep is one parameter. Two properties hold by construction and are tested:

- severity 0 is exactly the identity, with an all-ones evidence map;
- for a fixed seed, damage is **monotone** in severity -- raising severity never restores
  evidence. This is why all randomness is sampled at unit scale up front and severity only
  scales magnitudes and counts afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

__all__ = ["WearConfig", "WearRecord", "WearResult", "WearDegradation"]


@dataclass(frozen=True)
class WearConfig:
    """Parameters for the four wear mechanisms.

    Values are at **severity 1.0**; severity scales them down linearly. Ranges that are
    tuples are sampled per image and are independent of severity, so that the per-seed
    monotonicity property holds.
    """

    # -- ridge amplitude attenuation ------------------------------------------------
    enable_attenuation: bool = True
    attenuation_floor: float = 0.12
    """Fraction of ridge contrast surviving in the worst-affected areas at severity 1."""
    attenuation_scale_px: float = 48.0
    """Spatial scale of the attenuation field; wear is patchy, not uniform."""

    # -- flexion creases --------------------------------------------------------------
    enable_creases: bool = True
    max_creases: int = 5
    crease_width_px: tuple[float, float] = (2.0, 6.0)
    crease_softness_px: float = 1.2

    # -- dryness fragmentation ---------------------------------------------------------
    enable_fragmentation: bool = True
    fragmentation_max: float = 0.40
    """Fraction of the image broken at severity 1."""
    fragment_length_px: float = 15.0
    """Correlation length **along** the ridge -- how long each surviving dash is."""
    fragment_width_px: float = 2.5
    """Correlation length across the ridge."""
    n_orientation_bins: int = 8
    """Oriented filter bank size for anisotropic break generation."""
    fragment_noise_scale: float = 0.4
    """Resolution at which the oriented break field is generated, then upsampled.

    The field is smooth random noise that gets thresholded, so generating it at reduced
    scale preserves its correlation structure in image units while cutting the cost
    quadratically in both image area and kernel size. At full resolution this was 96 ms of
    a 163 ms call -- a 91x91 kernel applied once per orientation -- which starved the GPU
    during training. Set to 1.0 to disable.
    """

    # -- pressure-dependent partial contact ---------------------------------------------
    enable_contact: bool = True
    contact_falloff: float = 0.85
    """Contact lost at the periphery at severity 1."""
    contact_centre_jitter: float = 0.12
    """Core displacement from image centre, as a fraction of image size."""
    contact_extent: tuple[float, float] = (0.55, 0.85)
    """Radius of good contact, as a fraction of the half-diagonal."""

    # -- shared -------------------------------------------------------------------------
    ridge_period_px: float = 9.0
    """Nominal ridge period at 500 dpi; sets the local-statistics window."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WearRecord:
    """What was applied to one image, so any sample can be explained afterwards."""

    severity: float = 0.0
    applied: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def note(self, name: str, **params) -> None:
        self.applied.append(name)
        self.params[name] = params

    def to_dict(self) -> dict:
        return {"severity": self.severity, "applied": list(self.applied),
                "params": dict(self.params)}


@dataclass(frozen=True)
class WearResult:
    """Degraded image, its evidence map, and the record of what produced them.

    `latent.LatentDegradation` returns a 2-tuple; this returns a named result because the
    evidence map is a first-class output rather than a diagnostic.
    """

    image: np.ndarray      # float32 [0,1], same shape as input
    evidence: np.ndarray   # float32 [0,1], fraction of ridge evidence surviving
    record: WearRecord


def _u(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _ridge_orientation(img: np.ndarray, period_px: float) -> np.ndarray:
    """Local ridge orientation in radians, via the structure tensor.

    Returns ridge direction (perpendicular to the gradient), in [0, pi).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(1.0, period_px)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    # Gradient orientation; ridges run perpendicular to it.
    gradient = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    return np.mod(gradient + np.pi / 2, np.pi).astype(np.float32)


def _oriented_noise(
    rng: np.random.Generator, orientation: np.ndarray, cfg: WearConfig
) -> np.ndarray:
    """Noise smoothed **along** the local ridge direction.

    Dry-skin breaks run along a ridge, so the field that places them must be elongated in
    the ridge direction. A per-pixel oriented blur is too slow; a small steerable bank --
    filter the same noise at N orientations and select per pixel -- is equivalent at this
    resolution and costs a handful of convolutions.
    """
    full_h, full_w = orientation.shape
    scale = float(np.clip(cfg.fragment_noise_scale, 0.05, 1.0))
    h = max(16, int(round(full_h * scale)))
    w = max(16, int(round(full_w * scale)))
    effective = (h / full_h + w / full_w) / 2

    small_orientation = (orientation if (h, w) == (full_h, full_w)
                         else cv2.resize(orientation, (w, h),
                                         interpolation=cv2.INTER_NEAREST))
    noise = rng.standard_normal((h, w)).astype(np.float32)

    n_bins = max(2, cfg.n_orientation_bins)
    # Correlation lengths are in image pixels, so they scale with the working resolution.
    length = max(1.0, cfg.fragment_length_px * effective)
    width = max(0.6, cfg.fragment_width_px * effective)
    ksize = _odd(int(6 * length))
    half = ksize // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    responses = np.empty((n_bins, h, w), dtype=np.float32)
    for i in range(n_bins):
        theta = np.pi * i / n_bins
        # Rotate into (along-ridge, across-ridge) coordinates.
        along = xx * np.cos(theta) + yy * np.sin(theta)
        across = -xx * np.sin(theta) + yy * np.cos(theta)
        kernel = np.exp(
            -0.5 * ((along / length) ** 2 + (across / width) ** 2)
        ).astype(np.float32)
        kernel /= kernel.sum()
        responses[i] = cv2.filter2D(noise, -1, kernel, borderType=cv2.BORDER_REFLECT)

    index = np.clip(
        np.round(small_orientation / np.pi * n_bins).astype(np.int32), 0, n_bins - 1
    )
    field = np.take_along_axis(responses, index[None], axis=0)[0]
    if (h, w) != (full_h, full_w):
        field = cv2.resize(field, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    spread = field.std()
    return (field / spread).astype(np.float32) if spread > 1e-8 else field


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], scale_px: float
                  ) -> np.ndarray:
    """A smooth random field normalised to [0, 1]. Used for patchy wear."""
    field = cv2.GaussianBlur(
        rng.standard_normal(shape).astype(np.float32), (0, 0), max(1.0, scale_px)
    )
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-8:
        return np.zeros(shape, dtype=np.float32)
    return ((field - lo) / (hi - lo)).astype(np.float32)


class WearDegradation:
    """Applies the four wear mechanisms. Seeded, monotone in severity, and self-reporting."""

    def __init__(self, config: WearConfig | None = None) -> None:
        self.config = config or WearConfig()

    def __call__(
        self,
        image: np.ndarray,
        seed: int,
        severity: float = 0.5,
        mask: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> WearResult:
        """Degrade one greyscale image.

        `image` may be any dtype; values above 1.0 are taken as 0-255. Ridges are assumed
        dark on a light background, which holds for every dataset in this project.

        `mask` marks the finger. **Supply it whenever it is available.** Creases and dry
        skin exist only where there is skin, and contact pressure only matters where the
        finger touches, so without a mask the model reports destroyed evidence over blank
        background -- supervision that would teach the confidence head to distrust empty
        paper. Outside the mask the image is returned untouched and evidence stays 1.0,
        meaning "nothing was there to lose"; consumers should still restrict the
        confidence loss to the foreground.

        `orientation` supplies a ridge orientation field if one is already known, which
        skips the internal structure-tensor estimate. Training callers have the teacher's
        field cached, so passing it is both faster and better grounded than re-deriving a
        rough one from the image.
        """
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")
        cfg = self.config
        rng = np.random.default_rng(seed)
        rec = WearRecord(severity=float(severity))

        img = np.asarray(image, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"expected a 2-D greyscale image, got shape {img.shape}")
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        h, w = img.shape
        evidence = np.ones((h, w), dtype=np.float32)

        clean = img.copy()
        foreground = self._foreground(mask, (h, w))

        # Local statistics, computed once on the clean image. `valley` is the local
        # bright level: what a patch looks like where a ridge failed to transfer.
        window = _odd(int(2 * cfg.ridge_period_px))
        local_mean = cv2.GaussianBlur(img, (0, 0), cfg.ridge_period_px * 1.2)
        valley = cv2.dilate(img, np.ones((window, window), np.uint8))

        # Sample every random quantity *before* severity is applied, so that raising
        # severity for a fixed seed can only ever add damage.
        if orientation is None:
            orientation = _ridge_orientation(img, cfg.ridge_period_px)
        elif orientation.shape != img.shape:
            raise ValueError(
                f"orientation shape {orientation.shape} does not match image {img.shape}"
            )
        else:
            orientation = np.mod(np.asarray(orientation, np.float32), np.pi)
        wear_field = _smooth_field(rng, (h, w), cfg.attenuation_scale_px)
        break_field = _oriented_noise(rng, orientation, cfg) if cfg.enable_fragmentation \
            else np.zeros((h, w), np.float32)
        creases = self._sample_creases(rng, (h, w))
        contact = self._contact_field(rng, (h, w))

        # -- 1. ridge amplitude attenuation ------------------------------------------
        if cfg.enable_attenuation and severity > 0:
            depth = severity * (1.0 - cfg.attenuation_floor)
            surviving = (1.0 - depth * wear_field).astype(np.float32)
            img = local_mean + (img - local_mean) * surviving
            evidence *= surviving
            rec.note("amplitude_attenuation", depth=float(depth),
                     min_surviving=float(surviving.min()))

        # -- 2. flexion creases --------------------------------------------------------
        if cfg.enable_creases and severity > 0:
            n = int(round(severity * cfg.max_creases))
            if n > 0:
                mask = self._crease_mask(creases[:n], (h, w), cfg)
                img = img * (1.0 - mask) + valley * mask
                evidence *= (1.0 - mask)
                rec.note("flexion_creases", count=n)

        # -- 3. dryness fragmentation ----------------------------------------------------
        if cfg.enable_fragmentation and severity > 0:
            fraction = severity * cfg.fragmentation_max
            if fraction > 0:
                # Threshold the oriented field at the quantile giving exactly `fraction`
                # coverage, so break area grows monotonically with severity.
                threshold = float(np.quantile(break_field, 1.0 - fraction))
                broken = (break_field >= threshold).astype(np.float32)
                broken = cv2.GaussianBlur(broken, (0, 0), 0.8)
                img = img * (1.0 - broken) + valley * broken
                evidence *= (1.0 - broken)
                rec.note("dryness_fragmentation", fraction=float(fraction),
                         actual=float((break_field >= threshold).mean()))

        # -- 4. pressure-dependent partial contact -----------------------------------------
        if cfg.enable_contact and severity > 0:
            retained = (1.0 - severity * cfg.contact_falloff * (1.0 - contact)
                        ).astype(np.float32)
            # Lost contact leaves background, not a darker ridge.
            img = img * retained + 1.0 * (1.0 - retained)
            evidence *= retained
            rec.note("partial_contact", falloff=float(severity * cfg.contact_falloff),
                     min_retained=float(retained.min()))

        if foreground is not None:
            # Restore the background exactly, and with it the claim that no evidence was
            # lost there. Compositing at the end rather than gating each mechanism keeps
            # the mechanisms simple and their spatial statistics unchanged.
            img = clean * (1.0 - foreground) + img * foreground
            evidence = 1.0 * (1.0 - foreground) + evidence * foreground
            rec.note("foreground_mask", coverage=float(foreground.mean()))

        return WearResult(
            image=np.clip(img, 0.0, 1.0).astype(np.float32),
            evidence=np.clip(evidence, 0.0, 1.0).astype(np.float32),
            record=rec,
        )

    # -- helpers ---------------------------------------------------------------------

    @staticmethod
    def _foreground(mask: np.ndarray | None, shape: tuple[int, int]
                    ) -> np.ndarray | None:
        """Normalise a supplied mask to float32 in [0,1], or None if absent."""
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.shape != shape:
            raise ValueError(f"mask shape {m.shape} does not match image shape {shape}")
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        return np.clip(m, 0.0, 1.0)

    def _sample_creases(
        self, rng: np.random.Generator, shape: tuple[int, int]
    ) -> list[dict]:
        """Pre-sample the full set of creases; severity decides how many are used."""
        h, w = shape
        out: list[dict] = []
        for _ in range(self.config.max_creases):
            # Enter and leave through different edges, so a crease crosses the print
            # rather than terminating inside it -- a fold does not stop mid-finger.
            edges = rng.permutation(4)[:2]
            points = [self._edge_point(rng, int(e), h, w) for e in edges]
            mid = (
                (points[0][0] + points[1][0]) / 2 + rng.uniform(-0.18, 0.18) * w,
                (points[0][1] + points[1][1]) / 2 + rng.uniform(-0.18, 0.18) * h,
            )
            out.append({
                "start": points[0],
                "control": mid,
                "end": points[1],
                "width": _u(rng, self.config.crease_width_px),
            })
        return out

    @staticmethod
    def _edge_point(rng: np.random.Generator, edge: int, h: int, w: int
                    ) -> tuple[float, float]:
        t = float(rng.uniform(0.1, 0.9))
        return [(t * w, 0.0), (t * w, h - 1.0), (0.0, t * h), (w - 1.0, t * h)][edge]

    @staticmethod
    def _crease_mask(creases: list[dict], shape: tuple[int, int], cfg: WearConfig
                     ) -> np.ndarray:
        """Soft mask of the crease bands, drawn as quadratic Beziers."""
        h, w = shape
        mask = np.zeros((h, w), np.float32)
        t = np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None]
        for crease in creases:
            p0 = np.asarray(crease["start"], dtype=np.float32)
            p1 = np.asarray(crease["control"], dtype=np.float32)
            p2 = np.asarray(crease["end"], dtype=np.float32)
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            cv2.polylines(mask, [curve.astype(np.int32)], False, 1.0,
                          max(1, int(round(crease["width"]))), cv2.LINE_AA)
        if cfg.crease_softness_px > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), cfg.crease_softness_px)
        return np.clip(mask, 0.0, 1.0)

    def _contact_field(self, rng: np.random.Generator, shape: tuple[int, int]
                       ) -> np.ndarray:
        """Smooth field, 1 at the contact core falling toward 0 at the periphery."""
        cfg = self.config
        h, w = shape
        cy = h / 2 + rng.uniform(-1, 1) * cfg.contact_centre_jitter * h
        cx = w / 2 + rng.uniform(-1, 1) * cfg.contact_centre_jitter * w
        extent = _u(rng, cfg.contact_extent) * np.hypot(h, w) / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Anisotropic: fingers are longer than they are wide.
        radius = np.sqrt(((xx - cx) / 0.85) ** 2 + (yy - cy) ** 2)
        return np.exp(-0.5 * (radius / max(extent, 1.0)) ** 2).astype(np.float32)
