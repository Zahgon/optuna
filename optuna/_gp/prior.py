from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from optuna._gp import gp
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


