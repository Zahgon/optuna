from __future__ import annotations

from collections import defaultdict
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.importance._base import _check_evaluate_args
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova.scott_parzen_estimator import build_parzen_estimator_on_grid
from optuna.samplers._tpe.sampler import _split_complete_trials_multi_objective
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class _QuantileFilter:
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        target: Callable[[FrozenTrial], float] | None,
    ) -> None:
        assert 0 < quantile <= 1, "quantile must be in (0, 1]."
        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._target = target



@experimental_class("3.6.0")
class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):

    def __init__(
        self,
        *,
        target_quantile: float = 0.1,  # gamma' in the original paper
        region_quantile: float = 1.0,  # gamma in the original paper
        baseline_quantile: float | None = None,
        evaluate_on_local: bool = True,
    ) -> None:
        assert 0.0 < target_quantile < region_quantile <= 1.0, (
            "condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied"
        )
        if baseline_quantile is not None:
            msg = _DEPRECATION_WARNING_TEMPLATE.format(
                name="`baseline_quantile`", d_ver="4.7.0", r_ver="5.0.0"
            )
            optuna_warn(
                f"{msg} `baseline_quantile` is currently ignored. Use `target_quantile` instead.",
            )
        if region_quantile != 1.0 and not evaluate_on_local:
            optuna_warn("If `evaluate_on_local` is False, `region_quantile` has no effect.")

        self._target_quantile = target_quantile
        self._region_quantile = region_quantile
        self._evaluate_on_local = evaluate_on_local

        self._n_steps: int = 50
        self._prior_weight = 1.0
        self._min_n_trials_in_regime = 2

    def _get_top_quantile_trials(
        self,
        study: Study,
        trials: list[FrozenTrial],
        quantile: float,
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        if quantile == 1.0:
            return trials
        if study._is_multi_objective() and target is None:
            n_below = math.ceil(quantile * len(trials))
            top_trials, _ = _split_complete_trials_multi_objective(trials, study, n_below)
            return top_trials
        is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
        if target is not None:
            optuna_warn(
                f"{self.__class__.__name__} computes the importances of params to achieve "
                "low `target` values. If this is not what you want, "
                "please modify target, e.g., by multiplying the output by -1."
            )
            is_lower_better = True

        top_trials = _QuantileFilter(quantile, is_lower_better, target).filter(trials)

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        target_trials: list[FrozenTrial],
        region_trials: list[FrozenTrial],
    ) -> float:
        prior_weight = self._prior_weight
        pe_top, grid_size = build_parzen_estimator_on_grid(
            param_name, dist, target_trials, self._n_steps, prior_weight
        )
        grids = np.arange(grid_size)
        pdf_top = pe_top.pdf({param_name: grids}) + 1e-12

        if self._evaluate_on_local:  # The importance of param during the study.
            pe_local, _ = build_parzen_estimator_on_grid(
                param_name, dist, region_trials, self._n_steps, prior_weight
            )
            pdf_local = pe_local.pdf({param_name: grids}) + 1e-12
        else:  # The importance of param in the search space.
            pdf_local = np.full(grid_size, 1.0 / grid_size)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        """Evaluate parameter importances based on completed trials in the given study.

        .. note::

            This method is not meant to be called by library users.

        .. seealso::

            Please refer to :func:`~optuna.importance.get_param_importances` for how a concrete
            evaluator should implement this method.

        Args:
            study:
                An optimized study.
            params:
                A list of names of parameters to assess.
                If :obj:`None`, all parameters that are present in all of the completed trials are
                assessed.
            target:
                A function to specify the value to evaluate importances.
                If it is :obj:`None` and ``study`` is being used for single-objective optimization,
                the objective values are used. If it is :obj:`None` and ``study`` is being used for
                multi-objective optimization, the importance of reaching the Pareto front is
                evaluated by selecting top-quantile trials without preference for any particular
                objective, using non-domination rank and HSSP tie-breaking. To evaluate importance
                against a single objective or another trial attribute, specify ``target``
                explicitly, for example ``target=lambda t: t.values[0]`` or
                ``target=lambda t: t.duration.total_seconds()``.

                .. note::
                    :class:`PedAnovaImportanceEvaluator` assumes lower ``target`` values are
                    better.

        Returns:
            A :obj:`dict` where the keys are parameter names and the values are assessed
            importances.

        """
        dists = _get_distributions_list(study, params=params)
        if params is None:
            params = list(dict.fromkeys(k for d in dists for k in d))

        assert params is not None

        trials = _get_filtered_trials(study, target)
        if len(trials) <= 1:
            optuna_warn(
                "The number of trials is too small to compute importances. "
                "Parameter importances will be equal."
            )
            return {k: 0.0 for k in params}

        target_trials = self._get_top_quantile_trials(study, trials, self._target_quantile, target)
        region_trials = self._get_top_quantile_trials(study, trials, self._region_quantile, target)
        if len(target_trials) == len(region_trials):
            optuna_warn(
                "Target and region quantiles select the same set of trials. "
                "Parameter importances will be equal."
            )
        if len(target_trials) == 0:
            return {k: 0.0 for k in params}
        target_trial_ids = set(t._trial_id for t in target_trials)
        region_trial_ids = set(t._trial_id for t in region_trials)
        assert target_trial_ids.issubset(region_trial_ids)

        quantile = len(target_trials) / len(region_trials)  # gamma' / gamma
        param_importances = {k: 0.0 for k in params}
        for param_name in params:
            regime_trials = _partition_by_regime(
                param_name, region_trials, self._min_n_trials_in_regime
            )
            for dist, region_trials_regime in regime_trials.items():
                target_trials_regime = [
                    t for t in region_trials_regime if t._trial_id in target_trial_ids
                ]
                target_prob_regime = len(target_trials_regime) / len(target_trials)  # alpha_i
                region_prob_regime = len(region_trials_regime) / len(region_trials)  # beta_i
                if dist is not None and not dist.single() and len(target_trials_regime):
                    param_importances[param_name] += (
                        target_prob_regime**2
                        / region_prob_regime
                        * self._compute_pearson_divergence(
                            param_name,
                            dist,
                            target_trials=target_trials_regime,
                            region_trials=region_trials_regime,
                        )
                    )
        param_importances = {k: v * quantile**2 for k, v in param_importances.items()}
        return _sort_dict_by_importance(param_importances)


def _partition_by_regime(
    param_name: str, trials: list[FrozenTrial], min_n_trials_in_regime: int
) -> dict[BaseDistribution | None, list[FrozenTrial]]:
    regime_trials: dict[BaseDistribution | None, list[FrozenTrial]] = defaultdict(list)
    for trial in trials:
        regime_trials[trial.distributions.get(param_name)].append(trial)

    if any(len(v) < min_n_trials_in_regime for v in regime_trials.values()):
        optuna_warn(
            f"Some regimes for parameter `{param_name}` have less than "
            f"{min_n_trials_in_regime} trials. "
            "The importance of the parameter may be inaccurate."
        )
    regime_trials = {k: v for k, v in regime_trials.items() if len(v) >= min_n_trials_in_regime}

    return regime_trials


def _get_filtered_trials(
    study: Study, target: Callable[[FrozenTrial], float] | None
) -> list[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        trial
        for trial in trials
        if (
            math.isfinite(target(trial))
            if target is not None
            else all(math.isfinite(v) for v in trial.values)
        )
    ]


def _get_distributions_list(
    study: Study, params: list[str] | None
) -> list[dict[str, BaseDistribution]]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    _check_evaluate_args(trials, params)
    params_set = set(params) if params is not None else None
    return [
        {k: v for k, v in t.distributions.items() if params_set is None or k in params_set}
        for t in trials
    ]
