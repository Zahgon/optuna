from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from collections.abc import KeysView

    import optuna






def _is_first_in_interval_step(
    step: int, intermediate_steps: KeysView[int], n_warmup_steps: int, interval_steps: int
) -> bool:
    nearest_lower_pruning_step = (
        step - n_warmup_steps
    ) // interval_steps * interval_steps + n_warmup_steps
    assert nearest_lower_pruning_step >= 0

    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step


class PercentilePruner(BasePruner):

    def __init__(
        self,
        percentile: float,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        if not 0.0 <= percentile <= 100:
            raise ValueError(
                f"Percentile must be between 0 and 100 inclusive, but got {percentile=}."
            )
        if n_startup_trials < 0:
            raise ValueError(
                f"Number of startup trials cannot be negative, but got {n_startup_trials=}."
            )
        if n_warmup_steps < 0:
            raise ValueError(
                f"Number of warmup steps cannot be negative, but got {n_warmup_steps=}."
            )
        if interval_steps < 1:
            raise ValueError(
                f"Pruning interval steps must be at least 1, but got {interval_steps=}."
            )
        if n_min_trials < 1:
            raise ValueError(
                f"Number of trials for pruning must be at least 1, but got {n_min_trials=}."
            )

        self._percentile = percentile
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps
        self._n_min_trials = n_min_trials

