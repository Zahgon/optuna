from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._transform import _SearchSpaceTransform
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState


if TYPE_CHECKING:
    from typing import Any

    from optuna import distributions
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class RandomSampler(BaseSampler):

    def __init__(self, seed: int | None = None) -> None:
        self._rng = LazyRandomState(seed)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()


    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

        return trans.untransform(trans_params)[param_name]
