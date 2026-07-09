from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _get_trans_params
from optuna.importance._base import _param_importances_to_dict
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.study import Study
    from optuna.trial import FrozenTrial


class FanovaImportanceEvaluator(BaseImportanceEvaluator):

    def __init__(self, *, n_trees: int = 64, max_depth: int = 64, seed: int | None = None) -> None:
        self._evaluator = _Fanova(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=seed,
        )

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        distributions = _get_distributions(study, params=params)
        if params is None:
            params = list(distributions.keys())
        assert params is not None

        non_single_distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }
        single_distributions = {
            name: dist for name, dist in distributions.items() if dist.single()
        }

        if len(non_single_distributions) == 0:
            return {k: 0.0 for k in single_distributions}

        trials: list[FrozenTrial] = _get_filtered_trials(study, params=params, target=target)
        if len(trials) <= 1:
            return {k: 0.0 for k in params}

        trans = _SearchSpaceTransform(
            non_single_distributions, transform_log=False, transform_step=False
        )

        trans_params: np.ndarray = _get_trans_params(trials, trans)
        target_values: np.ndarray = _get_target_values(trials, target)

        evaluator = self._evaluator
        evaluator.fit(
            X=trans_params,
            y=target_values,
            search_spaces=trans.bounds,
            column_to_encoded_columns=trans.column_to_encoded_columns,
        )
        param_importances = np.array(
            [evaluator.get_importance(i)[0] for i in range(len(non_single_distributions))]
        )

        return _sort_dict_by_importance(
            {
                **_param_importances_to_dict(non_single_distributions.keys(), param_importances),
                **_param_importances_to_dict(single_distributions.keys(), 0.0),
            }
        )
