from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Container

    from optuna.study import Study
    from optuna.trial import FrozenTrial


class MaxTrialsCallback:

    def __init__(
        self, n_trials: int, states: Container[TrialState] | None = (TrialState.COMPLETE,)
    ) -> None:
        self._n_trials = n_trials
        self._states = states

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        trials = study.get_trials(deepcopy=False, states=self._states)
        n_complete = len(trials)
        if n_complete >= self._n_trials:
            study.stop()
