"""Cappelli's released enhancement methods (GBFEN, SNFEN) as baselines.

`pyfing` is the reference implementation of the 2026 state of the art. We run
it ourselves on our own benchmark rather than quoting its published table,
because those numbers are on NIST SD27, which has been withdrawn since 2017
and which we cannot obtain.

Both methods need the same three upstream estimates, each produced by its own
network: segmentation (SUFS), orientation (SNFOE) and ridge frequency (SNFFE).
That four-network chain is what the thesis proposes to collapse into one, so
its cost is measured here rather than assumed -- see `inference_seconds` in
the returned metrics.

Backend note: pyfing is Keras 3 and we run it on the **PyTorch** backend, which
avoids TensorFlow entirely (native Windows TF has had no GPU support since
2.10). One incompatibility follows from that choice: pyfing calls `.numpy()`
on model outputs, which raises under torch because the tensors carry
`requires_grad`. Every call is therefore wrapped in `torch.no_grad()`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

__all__ = ["PyfingEnhancer", "ENHANCEMENT_METHODS"]

ENHANCEMENT_METHODS = ("SNFEN", "GBFEN")

# Keras reads its backend at import time, so this must be set before pyfing is
# imported anywhere in the process.
os.environ.setdefault("KERAS_BACKEND", "torch")


class PyfingEnhancer:
    """Enhance images with pyfing; callable as a `run_baseline` preprocess hook.

    Models load once on first use and are reused, so construct this once and
    reuse it across a dataset -- not per image.
    """

    def __init__(
        self,
        method: str = "SNFEN",
        *,
        cache_dir: str | Path,
        dpi: int = 500,
        segmentation: str = "SUFS",
        orientation: str = "SNFOE",
        frequency: str = "SNFFE",
        apply_mask: bool = True,
    ) -> None:
        if method not in ENHANCEMENT_METHODS:
            raise ValueError(f"method must be one of {ENHANCEMENT_METHODS}, got {method}")
        self.method = method
        self.dpi = dpi
        self.segmentation = segmentation
        self.orientation = orientation
        self.frequency = frequency
        self.apply_mask = apply_mask
        self.cache_dir = Path(cache_dir) / f"enhanced_{method.lower()}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._times: list[float] = []

    @property
    def inference_seconds(self) -> dict[str, float]:
        """Per-image wall clock over the images actually enhanced this run.

        The first image includes model warm-up and is excluded from the mean.
        """
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
        import torch
        from PIL import Image

        from fpe.data.convert import cache_path, load_greyscale

        src = Path(image_path)
        out = cache_path(src, self.cache_dir, ".png")
        if out.is_file():
            return out

        img = load_greyscale(src)
        start = time.perf_counter()
        enhanced, mask = self._enhance(img, torch)
        self._times.append(time.perf_counter() - start)

        if self.apply_mask and mask is not None:
            # Background to white: outside the mask there is no ridge evidence,
            # and leaving raw sensor noise there invites spurious minutiae.
            enhanced = np.where(mask > 0, enhanced, np.uint8(255)).astype(np.uint8)

        Image.fromarray(enhanced, mode="L").save(out, format="PNG")
        return out

    def _enhance(self, img: np.ndarray, torch) -> tuple[np.ndarray, np.ndarray | None]:
        import pyfing as pf

        with torch.no_grad():
            mask = pf.fingerprint_segmentation(img, dpi=self.dpi, method=self.segmentation)
            ori = pf.orientation_field_estimation(
                img, mask, dpi=self.dpi, method=self.orientation
            )
            per = pf.frequency_estimation(
                img, ori, mask, dpi=self.dpi, method=self.frequency
            )
            enhanced = pf.fingerprint_enhancement(
                img, ori, per, mask, dpi=self.dpi, method=self.method
            )
        return np.asarray(enhanced, dtype=np.uint8), mask
