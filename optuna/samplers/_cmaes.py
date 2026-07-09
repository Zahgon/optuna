from __future__ import annotations

import copy
import math
import pickle
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna import _deprecated
from optuna import logging
from optuna._convert_positional_args import convert_positional_args
from optuna._experimental import warn_experimental_argument
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna._warnings import optuna_warn
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    import cmaes

    from optuna.distributions import BaseDistribution
    from optuna.trial import FrozenTrial

    CmaClass: TypeAlias = cmaes.CMA | cmaes.SepCMA | cmaes.CMAwM
else:
    cmaes = _LazyImport("cmaes")

_logger = logging.get_logger(__name__)

_EPS = 1e-10
_SYSTEM_ATTR_MAX_LENGTH = 2045


class CmaEsSampler(BaseSampler):

    @convert_positional_args(
        previous_positional_arg_names=[
            "self",
            "x0",
            "sigma0",
            "n_startup_trials",
            "independent_sampler",
            "warn_independent_sampling",
            "seed",
        ],
        deprecated_version="4.9.0",
        removed_version="6.0.0",
    )
    def __init__(
        self,
        *,
        x0: dict[str, Any] | None = None,
        sigma0: float | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: int | None = None,
        consider_pruned_trials: bool = False,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = -1,
        use_separable_cma: bool = False,
        with_margin: bool = False,
        lr_adapt: bool = False,
        source_trials: list[FrozenTrial] | None = None,
    ) -> None:
        if restart_strategy is not None or inc_popsize != -1:
            msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
                name="`restart_strategy`", d_ver="4.4.0", r_ver="6.0.0"
            )
            optuna_warn(
                f"{msg} From v4.4.0 onward, `restart_strategy` automatically falls back to "
                "`None`. `restart_strategy` will be supported in OptunaHub.",
                FutureWarning,
            )
        if x0 is not None:
            msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
                name="`x0`", d_ver="4.9.0", r_ver="6.0.0"
            )
            optuna_warn(msg, FutureWarning)
        if sigma0 is not None:
            msg = _deprecated._DEPRECATION_WARNING_TEMPLATE.format(
                name="`sigma0`", d_ver="4.9.0", r_ver="6.0.0"
            )
            optuna_warn(msg, FutureWarning)

        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._cma_rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._popsize = popsize
        self._use_separable_cma = use_separable_cma
        self._with_margin = with_margin
        self._lr_adapt = lr_adapt
        self._source_trials = source_trials

        if self._use_separable_cma:
            self._attr_prefix = "sepcma:"
        elif self._with_margin:
            self._attr_prefix = "cmawm:"
        else:
            self._attr_prefix = "cma:"

        if self._consider_pruned_trials:
            warn_experimental_argument("consider_pruned_trials")

        if self._use_separable_cma:
            warn_experimental_argument("use_separable_cma")

        if self._source_trials is not None:
            warn_experimental_argument("source_trials")

        if self._with_margin:
            warn_experimental_argument("with_margin")

        if self._lr_adapt:
            warn_experimental_argument("lr_adapt")

        if source_trials is not None and (x0 is not None or sigma0 is not None):
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when x0 or sigma0 is specified."
            )

        if source_trials is not None and use_separable_cma:
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES."
            )

        if lr_adapt and (use_separable_cma or with_margin):
            raise ValueError(
                "It is prohibited to pass `use_separable_cma` or `with_margin` argument when "
                "using `lr_adapt`."
            )

        if self._use_separable_cma and self._with_margin:
            raise ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`."
            )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()


    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)
        if len(completed_trials) < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(
            search_space, transform_step=not self._with_margin, transform_0_1=True
        )

        optimizer = self._restore_optimizer(completed_trials)
        if optimizer is None:
            optimizer = self._init_optimizer(trans, study.direction)

        if optimizer.dim != len(trans.bounds):
            if self._warn_independent_sampling:
                ind_sampler_name = self._independent_sampler.__class__.__name__
                _logger.warning(
                    "`CmaEsSampler` does not support dynamic search space. "
                    f"`{ind_sampler_name}` is used instead of `CmaEsSampler`."
                )
                self._warn_independent_sampling = False
            return {}

        solution_trials = self._get_solution_trials(completed_trials, optimizer.generation)

        if len(solution_trials) >= optimizer.population_size:
            solutions: list[tuple[np.ndarray, float]] = []
            for t in solution_trials[: optimizer.population_size]:
                assert t.value is not None, "completed trials must have a value"
                if isinstance(optimizer, cmaes.CMAwM):
                    x = np.array(t.system_attrs["x_for_tell"])
                else:
                    x = trans.transform(t.params)
                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value
                solutions.append((x, y))

            optimizer.tell(solutions)

            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        if isinstance(optimizer, cmaes.CMAwM):
            params, x_for_tell = optimizer.ask()
            study._storage.set_trial_system_attr(
                trial._trial_id, "x_for_tell", x_for_tell.tolist()
            )
        else:
            params = optimizer.ask()

        generation_attr_key = self._attr_key_generation
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )

        external_values = trans.untransform(params)

        return external_values



    def _concat_optimizer_attrs(self, optimizer_attrs: dict[str, str]) -> str:
        return "".join(
            optimizer_attrs[f"{self._attr_key_optimizer}:{i}"] for i in range(len(optimizer_attrs))
        )

    def _split_optimizer_str(self, optimizer_str: str) -> dict[str, str]:
        optimizer_len = len(optimizer_str)
        attrs = {}
        for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
            start = i * _SYSTEM_ATTR_MAX_LENGTH
            end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
            attrs[f"{self._attr_key_optimizer}:{i}"] = optimizer_str[start:end]
        return attrs

    def _restore_optimizer(
        self,
        completed_trials: "list[optuna.trial.FrozenTrial]",
    ) -> "CmaClass" | None:
        for trial in reversed(completed_trials):
            optimizer_attrs = {
                key: value
                for key, value in trial.system_attrs.items()
                if key.startswith(self._attr_key_optimizer)
            }
            if len(optimizer_attrs) == 0:
                continue

            optimizer_str = self._concat_optimizer_attrs(optimizer_attrs)
            return pickle.loads(bytes.fromhex(optimizer_str))
        return None

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
    ) -> "CmaClass":
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._source_trials is None:
            if self._x0 is None:
                mean = lower_bounds + (upper_bounds - lower_bounds) / 2
            else:
                mean = trans.transform(self._x0)

            if self._sigma0 is None:
                sigma0 = np.min((upper_bounds - lower_bounds) / 6)
            else:
                sigma0 = self._sigma0

            cov = None
        else:
            expected_states = [TrialState.COMPLETE]
            if self._consider_pruned_trials:
                expected_states.append(TrialState.PRUNED)

            sign = 1 if direction == StudyDirection.MINIMIZE else -1
            source_solutions = [
                (trans.transform(t.params), sign * cast("float", t.value))
                for t in self._source_trials
                if t.state in expected_states
                and _is_compatible_search_space(trans, t.distributions)
            ]
            if len(source_solutions) == 0:
                raise ValueError("No compatible source_trials")

            mean, sigma0, cov = cmaes.get_warm_start_mgd(source_solutions)

        sigma0 = max(sigma0, _EPS)

        if self._use_separable_cma:
            if len(trans.bounds) == 1:
                optuna_warn(
                    "Separable CMA-ES does not operate meaningfully on single-dimensional "
                    "search spaces. The setting `use_separable_cma=True` will be ignored.",
                    UserWarning,
                )
            else:
                return cmaes.SepCMA(
                    mean=mean,
                    sigma=sigma0,
                    bounds=trans.bounds,
                    seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                    n_max_resampling=10 * n_dimension,
                    population_size=self._popsize,
                )

        if self._with_margin:
            steps = np.empty(len(trans._search_space), dtype=float)
            for i, dist in enumerate(trans._search_space.values()):
                assert isinstance(dist, (IntDistribution, FloatDistribution))
                if dist.step is None or dist.log:
                    steps[i] = 0.0
                elif dist.low == dist.high:
                    steps[i] = 1.0
                else:
                    steps[i] = dist.step / (dist.high - dist.low)

            return cmaes.CMAwM(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                steps=steps,
                cov=cov,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=self._popsize,
            )

        return cmaes.CMA(
            mean=mean,
            sigma=sigma0,
            cov=cov,
            bounds=trans.bounds,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=self._popsize,
            lr_adapt=self._lr_adapt,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
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
                    "by `CmaEsSampler`"
                ),
            )
        )

    def _get_trials(self, study: "optuna.Study") -> list[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def _get_solution_trials(
        self, trials: list[FrozenTrial], generation: int
    ) -> list[FrozenTrial]:
        generation_attr_key = self._attr_key_generation
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    def before_trial(self, study: optuna.Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


def _is_compatible_search_space(
    trans: _SearchSpaceTransform, search_space: dict[str, BaseDistribution]
) -> bool:
    intersection_size = len(set(trans._search_space.keys()).intersection(search_space.keys()))
    return intersection_size == len(trans._search_space) == len(search_space)
