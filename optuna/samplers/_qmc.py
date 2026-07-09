from __future__ import annotations

import hashlib
import json
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna import logging
from optuna._experimental import experimental_class
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = logging.get_logger(__name__)

_SUGGESTED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
_threading_lock = threading.Lock()


@experimental_class("3.0.0")
class QMCSampler(BaseSampler):

    def __init__(
        self,
        *,
        qmc_type: str = "sobol",
        scramble: bool = False,  # default is False for simplicity in distributed environment.
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        warn_asynchronous_seeding: bool = True,
        warn_independent_sampling: bool = True,
    ) -> None:
        self._scramble = scramble
        self._seed = np.random.PCG64().random_raw() if seed is None else seed
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling

        if qmc_type in ("halton", "sobol"):
            self._qmc_type = qmc_type
        else:
            raise ValueError(f"The `{qmc_type=}` is invalid. Choose either `halton` or `sobol`.")

        if seed is None and scramble and warn_asynchronous_seeding:
            self._log_asynchronous_seeding()

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()



    @staticmethod
    def _log_asynchronous_seeding() -> None:
        _logger.warning(
            "No seed is provided for `QMCSampler` and the seed is set randomly. "
            "If you are running multiple `QMCSampler`s in parallel and/or distributed "
            " environment, the same seed must be used in all samplers to ensure that resulting "
            "samples are taken from the same QMC sequence. "
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            _INDEPENDENT_SAMPLING_WARNING_TEMPLATE.format(
                param_name=param_name,
                trial_number=trial.number,
                independent_sampler_name=self._independent_sampler.__class__.__name__,
                sampler_name=self.__class__.__name__,
                fallback_reason=(
                    "dynamic search space and `CategoricalDistribution` are not supported "
                    "by `QMCSampler`"
                ),
            )
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if len(study._get_trials(deepcopy=False, states=_SUGGESTED_STATES, use_cache=True)):
            if self._warn_independent_sampling:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        sample = self._sample_qmc(study, search_space)
        trans = _SearchSpaceTransform(search_space)
        sample = trans.bounds[:, 0] + sample * (trans.bounds[:, 1] - trans.bounds[:, 0])
        return trans.untransform(sample[0, :])

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)

    def _sample_qmc(self, study: Study, search_space: dict[str, BaseDistribution]) -> np.ndarray:
        qmc_module = _LazyImport("scipy.stats.qmc")

        sample_id = self._find_sample_id(study, search_space)
        d = len(search_space)

        if self._qmc_type == "halton":
            qmc_engine = qmc_module.Halton(d, seed=self._seed, scramble=self._scramble)
        elif self._qmc_type == "sobol":
            with _threading_lock:
                qmc_engine = qmc_module.Sobol(d, seed=self._seed, scramble=self._scramble)
        else:
            raise ValueError("Invalid `qmc_type`")

        forward_size = sample_id  # `sample_id` starts from 0.
        if forward_size > 0:
            qmc_engine.fast_forward(forward_size)
        sample = qmc_engine.random(1)

        return sample

    def _find_sample_id(self, study: Study, search_space: dict[str, BaseDistribution]) -> int:
        search_space_str = {
            param_name: str(dist) for param_name, dist in sorted(search_space.items())
        }
        qmc_vars: dict[str, Any] = {"qmc_type": self._qmc_type, "search_space": search_space_str}
        if self._scramble:
            qmc_vars.update(scramble=True, seed=self._seed)
        else:
            qmc_vars.update(scramble=False)

        key_qmc_id = "qmc:" + hashlib.sha256(json.dumps(qmc_vars).encode()).hexdigest()
        sample_id = study._storage.get_study_system_attrs(study._study_id).get(key_qmc_id, -1) + 1
        study._storage.set_study_system_attr(study._study_id, key_qmc_id, sample_id)
        return sample_id
