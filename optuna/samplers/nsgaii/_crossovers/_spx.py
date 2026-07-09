from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class SPXCrossover(BaseCrossover):

    n_parents = 3

    def __init__(self, epsilon: float | None = None) -> None:
        self._epsilon = epsilon

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:

        n = self.n_parents - 1
        G = np.mean(parents_params, axis=0)  # Equation (1).
        rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))  # Equation (2).

        epsilon = np.sqrt(len(search_space_bounds) + 2) if self._epsilon is None else self._epsilon
        xks = [G + epsilon * (pk - G) for pk in parents_params]  # Equation (3).

        ck = 0  # Equation (4).
        for k in range(1, self.n_parents):
            ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)

        child_params = xks[-1] + ck  # Equation (5).

        return child_params
