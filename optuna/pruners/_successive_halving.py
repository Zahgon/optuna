from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optuna.pruners._base import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    import optuna


class SuccessiveHalvingPruner(BasePruner):

    def __init__(
        self,
        min_resource: str | int = "auto",
        reduction_factor: int = 4,
        min_early_stopping_rate: int = 0,
        bootstrap_count: int = 0,
    ) -> None:
        if isinstance(min_resource, str) and min_resource != "auto":
            raise ValueError(
                f"The value of `min_resource` is {min_resource}, "
                "but must be either `min_resource` >= 1 or 'auto'"
            )

        if isinstance(min_resource, int) and min_resource < 1:
            raise ValueError(
                f"The value of `min_resource` is {min_resource}, "
                "but must be either `min_resource >= 1` or 'auto'"
            )

        if reduction_factor < 2:
            raise ValueError(
                f"The value of `reduction_factor` is {reduction_factor}, "
                "but must be `reduction_factor >= 2`"
            )

        if min_early_stopping_rate < 0:
            raise ValueError(
                f"The value of `min_early_stopping_rate` is {min_early_stopping_rate}, "
                "but must be `min_early_stopping_rate >= 0`"
            )

        if bootstrap_count < 0:
            raise ValueError(
                "The value of `bootstrap_count` is "
                f"{bootstrap_count}, but must be `bootstrap_count >= 0`"
            )

        if bootstrap_count > 0 and min_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and min_resource == 'auto' "
                f"are mutually incompatible, bootstrap_count is {bootstrap_count}"
            )

        self._min_resource: int | None = None
        if isinstance(min_resource, int):
            self._min_resource = min_resource
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate = min_early_stopping_rate
        self._bootstrap_count = bootstrap_count











