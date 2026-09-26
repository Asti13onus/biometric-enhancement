"""WAFEN as a `run_baseline` preprocess hook, so it is scored exactly like GBFEN and SNFEN.

Same contract as `PyfingEnhancer`: called with an image path, returns the path of an
enhanced 8-bit PNG, caches by source path, and reports per-image CPU wall clock with the
warm-up image excluded. Running both through one hook is what makes the comparison fair --
identical extractor, matcher, pairs and bootstrap; only the enhancement differs.

The enhanced image is the ridge head rendered as ink (dark ridges on white, the polarity
mindtct expects). Background is whitened where the model's own segmentation head says there
is no finger -- the same treatment pyfing's baselines get from their SUFS mask. Confidence
gating (abstention) is deliberately *not* applied here: this is the coverage-1.0 point that
every competing method is scored at, and remains the default.

With `abstain_threshold` set (plan U9), foreground pixels whose confidence falls below it
are declined: written as background, and -- through `filter_template` -- any minutia within
`guard_px` of a declined pixel is dropped. The guard matters: blanking a region cuts every
ridge crossing its edge, and each cut is a ridge ending mindtct will report as a minutia.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

__all__ = ["WafenEnhancer"]


class WafenEnhancer:
    """Enhance images with a trained WAFEN checkpoint (`best.pt` from `fpe.train`)."""

    def __init__(self, checkpoint: str | Path, *, cache_dir: str | Path,
                 device: str = "cpu", mask_threshold: float = 0.5,
                 abstain_threshold: float | None = None, guard_px: int = 8,
                 block: int = 0) -> None:
        import torch

        from fpe.models.wafen import Wafen, WafenConfig

        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = Wafen(WafenConfig(**state["config"])).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = torch.device(device)
        self.stride = 2 ** (self.model.config.depth - 1)
        self.mask_threshold = mask_threshold
        self.abstain_threshold = abstain_threshold
        self.guard_px = guard_px
        self.block = block
        """0 = per-pixel gating; >0 = gate on mean confidence per block of this size."""
        self.cache_dir = Path(cache_dir) / "enhanced_wafen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._times: list[float] = []
        self._coverage: list[float] = []
        self._dropped: list[int] = []

    @property
    def abstention_metrics(self) -> dict[str, float]:
        """Mean foreground coverage and minutiae dropped per image; empty at coverage 1."""
        if self.abstain_threshold is None or not self._coverage:
            return {}
        return {
            "abstain_threshold": self.abstain_threshold,
            "coverage_mean": float(np.mean(self._coverage)),
            "minutiae_dropped_mean": float(np.mean(self._dropped)) if self._dropped else 0.0,
        }

    @property
    def inference_seconds(self) -> dict[str, float]:
        """Per-image wall clock over the images enhanced this run, warm-up excluded."""
        if len(self._times) <= 1:
            return {}
        t = np.asarray(self._times[1:], dtype=np.float64)
        return {
            "inference_mean_s": float(t.mean()),
            "inference_median_s": float(np.median(t)),
            "inference_max_s": float(t.max()),
            "inference_n": int(t.size),
        }

    def __call__(self, image_path: str | Path) -> Path:
        from PIL import Image

        from fpe.data.convert import cache_path, load_greyscale

        src = Path(image_path)
        out = cache_path(src, self.cache_dir, ".png")
        keep_path = self._keep_path(out)
        if out.is_file() and (self.abstain_threshold is None or keep_path.is_file()):
            return out

        img = load_greyscale(src).astype(np.float32) / 255.0
        start = time.perf_counter()
        ridge, mask, confidence = self.enhance(img)
        self._times.append(time.perf_counter() - start)

        foreground = mask >= self.mask_threshold
        shown = foreground
        if self.abstain_threshold is not None:
            import cv2

            from fpe.eval.abstain import block_confidence, coverage_fraction

            if self.block:
                confidence = block_confidence(confidence, foreground, self.block)
            declined = foreground & (confidence < self.abstain_threshold)
            shown = foreground & ~declined
            self._coverage.append(coverage_fraction(~declined, foreground))
            size = 2 * self.guard_px + 1
            near = cv2.dilate(declined.astype(np.uint8), np.ones((size, size), np.uint8))
            Image.fromarray(np.where(near > 0, 0, 255).astype(np.uint8), mode="L").save(
                keep_path, "PNG")

        ink = np.where(shown, 1.0 - ridge, 1.0)
        Image.fromarray(np.round(ink * 255).astype(np.uint8), mode="L").save(out, "PNG")
        return out

    @staticmethod
    def _keep_path(enhanced: Path) -> Path:
        return enhanced.with_name(enhanced.stem + ".keep.png")

    def filter_template(self, xyt: Path, enhanced: Path) -> Path:
        """`run_baseline` postprocess hook: drop minutiae in or near declined regions."""
        if self.abstain_threshold is None:
            return xyt
        from fpe.data.convert import load_greyscale
        from fpe.eval.abstain import filter_xyt_file

        keep = load_greyscale(self._keep_path(Path(enhanced))) > 127
        self._dropped.append(filter_xyt_file(xyt, keep))
        return xyt

    def enhance(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ridge probability, segmentation, confidence) for a [0, 1] image, at input size.

        The network needs sides divisible by its total stride; pad with white (background,
        the training convention) and crop the result back.
        """
        import torch

        h, w = img.shape
        ph, pw = -h % self.stride, -w % self.stride
        padded = np.pad(img, ((0, ph), (0, pw)), constant_values=1.0)
        with torch.no_grad():
            x = torch.from_numpy(padded)[None, None].to(self.device)
            output = self.model(x)
        ridge = output.ridge[0, 0, :h, :w].cpu().numpy()
        mask = output.segmentation[0, 0, :h, :w].cpu().numpy()
        confidence = output.confidence[0, 0, :h, :w].cpu().numpy()
        return ridge, mask, confidence
