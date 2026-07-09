from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.samplers import BaseSampler


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState


@experimental_class("2.4.0")
class PartialFixedSampler(BaseSampler):

    def __init__(self, fixed_params: dict[str, Any], base_sampler: BaseSampler) -> None:
        self._fixed_params = fixed_params
        self._base_sampler = base_sampler

    def reseed_rng(self) -> None:
        self._base_sampler.reseed_rng()


    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return self._base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if param_name not in self._fixed_params:
            return self._base_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        else:
            param_value = self._fixed_params[param_name]

            param_value_in_internal_repr = param_distribution.to_internal_repr(param_value)
            contained = param_distribution._contains(param_value_in_internal_repr)

            if not contained:
                optuna_warn(
                    f"Fixed parameter '{param_name}' with value {param_value} is out of range "
                    f"for distribution {param_distribution}."
                )
            return param_value

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._base_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._base_sampler.after_trial(study, trial, state, values)
