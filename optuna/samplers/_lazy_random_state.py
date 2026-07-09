from __future__ import annotations

import numpy as np


class LazyRandomState:

    def __init__(self, seed: int | None = None) -> None:
        self._rng: np.random.RandomState | None = None
        if seed is not None:
            self.rng.seed(seed=seed)


