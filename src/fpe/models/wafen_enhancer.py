"""WAFEN as a `run_baseline` preprocess hook, so it is scored exactly like GBFEN and SNFEN.

Same contract as `PyfingEnhancer`: called with an image path, returns the path of an
enhanced 8-bit PNG, caches by source path, and reports per-image CPU wall clock with the
warm-up image excluded. Running both through one hook is what makes the comparison fair --
identical extractor, matcher, pairs and bootstrap; only the enhancement differs.

The enhanced image is the ridge head rendered as ink (dark ridges on white, the polarity
mindtct expects). Background is whitened where the model's own segmentation head says there
is no finger -- the same treatment pyfing's baselines get from their SUFS mask. Confidence
gating (abstention) is deliberately *not* applied here: this is the coverage-1.0 point that
every competing method is scored at; the precision-coverage sweep is plan U9.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

__all__ = ["WafenEnhancer"]


class WafenEnhancer:
    """Enhance images with a trained WAFEN checkpoint (`best.pt` from `fpe.train`)."""

    def __init__(self, checkpoint: str | Path, *, cache_dir: str | Path,
                 device: str = "cpu", mask_threshold: float = 0.5) -> None:
        import torch

        from fpe.models.wafen import Wafen, WafenConfig

        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = Wafen(WafenConfig(**state["config"])).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = torch.device(device)
        self.stride = 2 ** (self.model.config.depth - 1)
        self.mask_threshold = mask_threshold
        self.cache_dir = Path(cache_dir) / "enhanced_wafen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._times: list[float] = []

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
        if out.is_file():
            return out

        img = load_greyscale(src).astype(np.float32) / 255.0
        start = time.perf_counter()
        ridge, mask = self.enhance(img)
        self._times.append(time.perf_counter() - start)

        ink = np.where(mask >= self.mask_threshold, 1.0 - ridge, 1.0)
        Image.fromarray(np.round(ink * 255).astype(np.uint8), mode="L").save(out, "PNG")
        return out

    def enhance(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(ridge probability, segmentation) for a [0, 1] image, both at input size.

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
        return ridge, mask
