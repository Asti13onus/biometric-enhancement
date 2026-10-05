"""Running pyfing's pretrained models safely alongside our own GPU training.

pyfing calls `.numpy()` directly on Keras model outputs. Under the TensorFlow backend that
works; under the PyTorch backend it raises twice over -- once because the tensors carry
`requires_grad`, and once, as soon as a CUDA device exists, because a CUDA tensor cannot be
converted to numpy without an explicit `.cpu()`.

The second failure is the nastier one, because it *appears when the project gets faster*:
every pyfing call in the codebase worked while torch was CPU-only and broke the moment a
CUDA build was installed, without a line of our code changing.

We do not want pyfing on the GPU anyway. Its models are teachers and baselines -- run once
and cached, at roughly 0.7 s per image on CPU -- while the GPU is wanted for training our
own network, and 4 GB does not want to host both.

Keras resolves its torch device **at import time**, from `KERAS_TORCH_DEVICE`. So the
device cannot be chosen after `import pyfing` has happened, and a caller that imports
pyfing before this module gets the GPU regardless of any later context manager. Rather than
depend on every caller getting import order right, `load_pyfing()` owns the import and sets
the variable first.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

__all__ = ["load_pyfing", "pyfing_cpu"]

# Both must precede any keras import. Keras reads its backend and its torch device once,
# at import, and caches them for the life of the process.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

_pyfing: Any = None


def load_pyfing() -> Any:
    """Import pyfing with its device pinned, and return the module.

    Always use this instead of `import pyfing`; a direct import elsewhere in the process
    first would fix Keras to the GPU and break every call with a numpy conversion error.
    """
    global _pyfing
    if _pyfing is None:
        if os.environ.get("KERAS_TORCH_DEVICE") != "cpu":
            raise RuntimeError(
                "KERAS_TORCH_DEVICE is not 'cpu'; pyfing would run on the GPU and its "
                "`.numpy()` calls would fail. Import fpe.models.pyfing_runtime before "
                "anything that imports keras."
            )
        import pyfing  # noqa: PLC0415 -- deliberately deferred until the device is set

        _pyfing = pyfing
    return _pyfing


@contextmanager
def pyfing_cpu():
    """Run pyfing calls with gradients disabled.

    Device placement is handled by `load_pyfing`; this covers the other half of the
    problem, the `requires_grad` flag that makes `.numpy()` raise under torch.
    """
    import torch

    with torch.no_grad():
        yield
