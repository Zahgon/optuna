from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    import numpy as np

    from optuna.study import Study


@experimental_class("3.0.0")
class BLXAlphaCrossover(BaseCrossover):

    n_parents = 2

    def __init__(self, alpha: float = 0.5) -> None:
        self._alpha = alpha

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:

        parents_min = parents_params.min(axis=0)
        parents_max = parents_params.max(axis=0)
        diff = self._alpha * (parents_max - parents_min)  # Equation (1).
        low = parents_min - diff  # Equation (1).
        high = parents_max + diff  # Equation (1).
        r = rng.rand(len(search_space_bounds))
        child_params = (high - low) * r + low

        return child_params
