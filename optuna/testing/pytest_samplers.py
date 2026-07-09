from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from _pytest.mark.structures import MarkDecorator

    from optuna.study import Study


def parametrize_suggest_method(name: str) -> MarkDecorator:
    return pytest.mark.parametrize(
        f"suggest_method_{name}",
        [
            lambda t: t.suggest_float(name, 0, 10),
            lambda t: t.suggest_int(name, 0, 10),
            lambda t: t.suggest_categorical(name, [0, 1, 2]),
            lambda t: t.suggest_float(name, 0, 10, step=0.5),
            lambda t: t.suggest_float(name, 1e-7, 10, log=True),
            lambda t: t.suggest_int(name, 1, 10, log=True),
        ],
    )


def _create_new_trial(study: Study) -> FrozenTrial:
    trial_id = study._storage.create_new_trial(study._study_id)
    return study._storage.get_trial(trial_id)


class FixedSampler(BaseSampler):
    def __init__(
        self,
        relative_search_space: dict[str, BaseDistribution],
        relative_params: dict[str, Any],
        unknown_param_value: Any,
    ) -> None:
        self.relative_search_space = relative_search_space
        self.relative_params = relative_params
        self.unknown_param_value = unknown_param_value

    pass

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self.relative_params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self.unknown_param_value


class _BaseSamplerTestCase:
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        raise NotImplementedError


class BasicSamplerTestCase(_BaseSamplerTestCase):
    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass


class RelativeSamplerTestCase(_BaseSamplerTestCase):
    pass

    pass

    pass

    pass


class MultiObjectiveSamplerTestCase(_BaseSamplerTestCase):
    pass


class SingleOnlySamplerTestCase(_BaseSamplerTestCase):
    pass
