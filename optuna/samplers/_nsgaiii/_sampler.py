from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna._experimental import experimental_class
from optuna.samplers._ga import BaseGASampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    NSGAIIIElitePopulationSelectionStrategy,
)
from optuna.samplers._random import RandomSampler
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import numpy as np

    from optuna.distributions import BaseDistribution
    from optuna.study import Study


@experimental_class("3.2.0")
class NSGAIIISampler(BaseGASampler):

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        elite_population_selection_strategy: (
            Callable[[Study, list[FrozenTrial]], list[FrozenTrial]] | None
        ) = None,
        child_generation_strategy: (
            Callable[[Study, dict[str, BaseDistribution], list[FrozenTrial]], dict[str, Any]]
            | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None] | None
        ) = None,
    ) -> None:

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        if population_size < crossover.n_parents:
            raise ValueError(
                f"Using {crossover},"
                f" the population size should be greater than or equal to {crossover.n_parents}."
                f" The specified `population_size` is {population_size}."
            )

        super().__init__(population_size=population_size)
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._constraints_func = constraints_func
        self._search_space = IntersectionSearchSpace()

        self._elite_population_selection_strategy = (
            elite_population_selection_strategy
            or NSGAIIIElitePopulationSelectionStrategy(
                population_size=population_size,
                constraints_func=constraints_func,
                reference_points=reference_points,
                dividing_parameter=dividing_parameter,
                rng=self._rng,
            )
        )
        self._child_generation_strategy = (
            child_generation_strategy
            or NSGAIIChildGenerationStrategy(
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                swapping_prob=swapping_prob,
                crossover=crossover,
                constraints_func=constraints_func,
                rng=self._rng,
            )
        )
        self._after_trial_strategy = after_trial_strategy or NSGAIIAfterTrialStrategy(
            constraints_func=constraints_func
        )

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng.rng.seed()


    def select_parent(self, study: Study, generation: int) -> list[FrozenTrial]:
        return self._elite_population_selection_strategy(
            study,
            self.get_population(study, generation - 1)
            + self.get_parent_population(study, generation - 1),
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        generation = self.get_trial_generation(study, trial)
        parent_population = self.get_parent_population(study, generation)
        if len(parent_population) == 0:
            return {}
        return self._child_generation_strategy(study, search_space, parent_population)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._after_trial_strategy(study, trial, state, values)
        self._random_sampler.after_trial(study, trial, state, values)
