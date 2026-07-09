from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class SBXCrossover(BaseCrossover):

    n_parents = 2

    def __init__(
        self,
        eta: float | None = None,
        uniform_crossover_prob: float = 0.5,
        use_child_gene_prob: float = 0.5,
    ) -> None:
        if (eta is not None) and (eta < 0.0):
            raise ValueError("The value of `eta` must be greater than or equal to 0.0.")
        self._eta = eta

        if uniform_crossover_prob < 0.0 or uniform_crossover_prob > 1.0:
            raise ValueError(
                "The value of `uniform_crossover_prob` must be in the range [0.0, 1.0]."
            )
        if use_child_gene_prob <= 0.0 or use_child_gene_prob > 1.0:
            raise ValueError("The value of `use_child_gene_prob` must be in the range (0.0, 1.0].")
        self._uniform_crossover_prob = uniform_crossover_prob
        self._use_child_gene_prob = use_child_gene_prob

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:

        xls = search_space_bounds[..., 0]
        xus = search_space_bounds[..., 1]

        xs_min = np.min(parents_params, axis=0)
        xs_max = np.max(parents_params, axis=0)
        if self._eta is None:
            eta = 20.0 if study._is_multi_objective() else 2.0
        else:
            eta = self._eta

        xs_diff = np.clip(xs_max - xs_min, 1e-10, None)
        beta1 = 1 + 2 * (xs_min - xls) / xs_diff
        beta2 = 1 + 2 * (xus - xs_max) / xs_diff
        alpha1 = 2 - np.power(beta1, -(eta + 1))
        alpha2 = 2 - np.power(beta2, -(eta + 1))

        us = rng.rand(len(search_space_bounds))
        mask1 = us > 1 / alpha1  # Equation (3).
        betaq1 = np.power(us * alpha1, 1 / (eta + 1))  # Equation (3).
        betaq1[mask1] = np.power((1 / (2 - us * alpha1)), 1 / (eta + 1))[mask1]  # Equation (3).

        mask2 = us > 1 / alpha2  # Equation (3).
        betaq2 = np.power(us * alpha2, 1 / (eta + 1))  # Equation (3)
        betaq2[mask2] = np.power((1 / (2 - us * alpha2)), 1 / (eta + 1))[mask2]  # Equation (3).

        c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)  # Equation (4).
        c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)  # Equation (5).


        child1_params_list = []
        child2_params_list = []

        for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, parents_params[0], parents_params[1]):
            if rng.rand() < self._use_child_gene_prob:
                if rng.rand() >= self._uniform_crossover_prob:
                    child1_params_list.append(c1_i)
                    child2_params_list.append(c2_i)
                else:
                    child1_params_list.append(c2_i)
                    child2_params_list.append(c1_i)
            else:
                if rng.rand() >= self._uniform_crossover_prob:
                    child1_params_list.append(x1_i)
                    child2_params_list.append(x2_i)
                else:
                    child1_params_list.append(x2_i)
                    child2_params_list.append(x1_i)

        child_params_list = child1_params_list if rng.rand() < 0.5 else child2_params_list
        child_params = np.array(child_params_list)

        return child_params
