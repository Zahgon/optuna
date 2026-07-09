from __future__ import annotations

import json
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna._experimental import warn_experimental_argument
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.study._multi_objective import _is_pareto_front
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch

    import optuna._gp.acqf as acqf_module
    import optuna._gp.gp as gp
    import optuna._gp.optim_mixed as optim_mixed
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    acqf_module = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")

import logging


_logger = logging.getLogger(__name__)

EPS = 1e-10

_RELATIVE_PARAMS_KEY = "gp:relative_params"
_SYSTEM_ATTR_MAX_LENGTH = 2045


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class GPSampler(BaseSampler):

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        warn_independent_sampling: bool = True,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[[gp.GPRegressor], torch.Tensor] = prior.default_log_prior
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        self._gprs_cache_list: list[gp.GPRegressor] | None = None
        self._constraints_gprs_cache_list: list[gp.GPRegressor] | None = None
        self._deterministic = deterministic_objective
        self._constraints_func = constraints_func
        self._warn_independent_sampling = warn_independent_sampling

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")
        if deterministic_objective:
            warn_experimental_argument("deterministic_objective")

        self._n_preliminary_samples: int = 2048
        self._n_local_search = 10
        self._tol = 1e-4
        self._n_qmc_samples_qei = 128
        self._n_qmc_samples_ehvi = 128  # NOTE(nabenabe): The BoTorch default value.

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        msg = _INDEPENDENT_SAMPLING_WARNING_TEMPLATE.format(
            param_name=param_name,
            trial_number=trial.number,
            independent_sampler_name=self._independent_sampler.__class__.__name__,
            sampler_name=self.__class__.__name__,
            fallback_reason="dynamic search space is not supported by GPSampler",
        )
        _logger.warning(msg)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()


    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params

    def _get_constraints_acqf_args(
        self,
        constraint_vals: np.ndarray,
        internal_search_space: gp_search_space.SearchSpace,
        normalized_params: np.ndarray,
    ) -> tuple[list[gp.GPRegressor], list[float]]:
        standardized_constraint_vals, means, stds = _standardize_values(-constraint_vals)
        if (
            self._gprs_cache_list is not None
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)
            != internal_search_space.dim
        ):
            self._constraints_gprs_cache_list = None

        is_categorical = internal_search_space.is_categorical
        constraints_gprs = []
        constraints_threshold_list = []
        constraints_threshold_list = (-means / np.maximum(EPS, stds)).tolist()
        for i, vals in enumerate(standardized_constraint_vals.T):
            cache = (
                self._constraints_gprs_cache_list[i]
                if self._constraints_gprs_cache_list is not None
                else None
            )
            gpr = gp.fit_kernel_params(
                X=normalized_params,
                Y=vals,
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                gpr_cache=cache,
                deterministic_objective=self._deterministic,
            )
            constraints_gprs.append(gpr)

        self._constraints_gprs_cache_list = constraints_gprs
        return constraints_gprs, constraints_threshold_list

    def _get_best_params_for_multi_objective(
        self,
        normalized_params: np.ndarray,
        standardized_score_vals: np.ndarray,
    ) -> np.ndarray:
        pareto_params = normalized_params[
            _is_pareto_front(-standardized_score_vals, assume_unique_lexsorted=False)
        ]
        n_pareto_sols = len(pareto_params)
        size = min(self._n_local_search // 2, n_pareto_sols)
        chosen_indices = self._rng.rng.choice(n_pareto_sols, size=size, replace=False)
        return pareto_params[chosen_indices]

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.RUNNING)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=False)
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        running_trials = [
            t
            for t in trials
            if t.state == TrialState.RUNNING and search_space.keys() <= _get_params(t).keys()
        ]

        if len(completed_trials) < self._n_startup_trials:
            return {}

        with torch.device("cpu"):
            params = self._sample_relative_impl(
                study, completed_trials, running_trials, search_space
            )

        if params != {}:
            params_str = json.dumps(params)
            for i in range(0, len(params_str), _SYSTEM_ATTR_MAX_LENGTH):
                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    f"{_RELATIVE_PARAMS_KEY}:{i // _SYSTEM_ATTR_MAX_LENGTH}",
                    params_str[i : i + _SYSTEM_ATTR_MAX_LENGTH],
                )

        return params

    def _sample_relative_impl(
        self,
        study: Study,
        completed_trials: list[FrozenTrial],
        running_trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        internal_search_space = gp_search_space.SearchSpace(search_space)
        normalized_params = internal_search_space.get_normalized_params(completed_trials)
        normalized_params_of_running_trials = (
            internal_search_space.get_normalized_params(
                running_trials, [_get_params(t) for t in running_trials]
            )
            if len(running_trials) > 0
            else None
        )

        _sign = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.values for trial in completed_trials])
        )

        if (
            self._gprs_cache_list is not None
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)
            != internal_search_space.dim
        ):
            self._gprs_cache_list = None

        gprs_list = []
        n_objectives = standardized_score_vals.shape[-1]
        is_categorical = internal_search_space.is_categorical
        for i in range(n_objectives):
            cache = self._gprs_cache_list[i] if self._gprs_cache_list is not None else None
            gprs_list.append(
                gp.fit_kernel_params(
                    X=normalized_params,
                    Y=standardized_score_vals[:, i],
                    is_categorical=is_categorical,
                    log_prior=self._log_prior,
                    minimum_noise=self._minimum_noise,
                    gpr_cache=cache,
                    deterministic_objective=self._deterministic,
                )
            )
        self._gprs_cache_list = gprs_list

        best_params: np.ndarray | None
        acqf: acqf_module.BaseAcquisitionFunc
        if self._constraints_func is None:
            if n_objectives == 1:
                assert len(gprs_list) == 1
                if normalized_params_of_running_trials is None:
                    acqf = acqf_module.LogEI(
                        gpr=gprs_list[0],
                        search_space=internal_search_space,
                        threshold=standardized_score_vals[:, 0].max(),
                    )
                else:
                    acqf = acqf_module.qLogEI(
                        gpr=gprs_list[0],
                        search_space=internal_search_space,
                        threshold=standardized_score_vals[:, 0].max(),
                        n_qmc_samples=self._n_qmc_samples_qei,
                        qmc_seed=self._rng.rng.randint(1 << 30),
                        normalized_params_of_running_trials=normalized_params_of_running_trials,
                    )
                best_params = normalized_params[np.argmax(standardized_score_vals), np.newaxis]
            else:
                acqf = acqf_module.LogEHVI(
                    gpr_list=gprs_list,
                    search_space=internal_search_space,
                    Y_train=torch.from_numpy(standardized_score_vals),
                    n_qmc_samples=self._n_qmc_samples_ehvi,
                    qmc_seed=self._rng.rng.randint(1 << 30),
                    normalized_params_of_running_trials=normalized_params_of_running_trials,
                )
                best_params = self._get_best_params_for_multi_objective(
                    normalized_params, standardized_score_vals
                )
        else:
            if n_objectives == 1:
                assert len(gprs_list) == 1
                constraint_vals, is_feasible = _get_constraint_vals_and_feasibility(
                    study, completed_trials
                )
                y_with_neginf = np.where(is_feasible, standardized_score_vals[:, 0], -np.inf)
                constr_gpr_list, constr_threshold_list = self._get_constraints_acqf_args(
                    constraint_vals, internal_search_space, normalized_params
                )
                i_opt = np.argmax(y_with_neginf)
                best_feasible_y = y_with_neginf[i_opt]
                acqf = acqf_module.ConstrainedLogEI(
                    gpr=gprs_list[0],
                    search_space=internal_search_space,
                    threshold=best_feasible_y,
                    constraints_gpr_list=constr_gpr_list,
                    constraints_threshold_list=constr_threshold_list,
                    normalized_params_of_running_trials=normalized_params_of_running_trials,
                )
                assert normalized_params.shape[:-1] == y_with_neginf.shape
                best_params = (
                    None if np.isneginf(best_feasible_y) else normalized_params[i_opt, np.newaxis]
                )
            else:
                constraint_vals, is_feasible = _get_constraint_vals_and_feasibility(
                    study, completed_trials
                )
                constr_gpr_list, constr_threshold_list = self._get_constraints_acqf_args(
                    constraint_vals, internal_search_space, normalized_params
                )
                is_all_infeasible = not any(is_feasible)
                acqf = acqf_module.ConstrainedLogEHVI(
                    gpr_list=gprs_list,
                    search_space=internal_search_space,
                    Y_feasible=(
                        torch.from_numpy(standardized_score_vals[is_feasible])
                        if not is_all_infeasible
                        else None
                    ),
                    n_qmc_samples=self._n_qmc_samples_ehvi,
                    qmc_seed=self._rng.rng.randint(1 << 30),
                    constraints_gpr_list=constr_gpr_list,
                    constraints_threshold_list=constr_threshold_list,
                    normalized_params_of_running_trials=normalized_params_of_running_trials,
                )
                best_params = (
                    self._get_best_params_for_multi_objective(
                        normalized_params[is_feasible],
                        standardized_score_vals[is_feasible],
                    )
                    if not is_all_infeasible
                    else None
                )

        normalized_param = self._optimize_acqf(acqf, best_params)
        return internal_search_space.get_unnormalized_param(normalized_param)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._warn_independent_sampling:
            states = (TrialState.COMPLETE,)
            completed_trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
            if len(completed_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)


def _get_constraint_vals_and_feasibility(
    study: Study, trials: list[FrozenTrial]
) -> tuple[np.ndarray, np.ndarray]:
    _constraint_vals = [
        study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY, ())
        for trial in trials
    ]
    if any(len(_constraint_vals[0]) != len(c) for c in _constraint_vals):
        raise ValueError("The number of constraints must be the same for all trials.")

    constraint_vals = np.array(_constraint_vals)
    assert len(constraint_vals.shape) == 2, "constraint_vals must be a 2d array."
    is_feasible = np.all(constraint_vals <= 0, axis=1)
    assert not isinstance(is_feasible, np.bool_), "MyPy Redefinition for NumPy v2.2.0."
    return constraint_vals, is_feasible


def _get_params(trial: FrozenTrial) -> dict[str, Any]:
    if trial.state.is_finished():
        return trial.params

    params_strs = []
    i = 0
    while params_str_i := trial.system_attrs.get(f"{_RELATIVE_PARAMS_KEY}:{i}"):
        params_strs.append(params_str_i)
        i += 1

    if len(params_strs) == 0:
        return trial.params
    params = json.loads("".join(params_strs))
    params.update(trial.params)
    return params
