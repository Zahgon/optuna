from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import optuna


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study


def _calculate(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
    search_space: dict[str, BaseDistribution] | None = None,
    cached_trial_number: int = -1,
) -> tuple[dict[str, BaseDistribution] | None, int]:
    states_of_interest = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.WAITING,
        optuna.trial.TrialState.RUNNING,
    ]

    if include_pruned:
        states_of_interest.append(optuna.trial.TrialState.PRUNED)

    next_cached_trial_number = -1

    for trial in reversed(trials):
        if trial.state not in states_of_interest:
            continue

        if next_cached_trial_number == -1:
            next_cached_trial_number = trial.number + 1

        if cached_trial_number > trial.number:
            break

        if not trial.state.is_finished():
            next_cached_trial_number = trial.number
            continue

        if search_space is None:
            search_space = copy.copy(trial.distributions)
            continue

        search_space = {
            name: distribution
            for name, distribution in search_space.items()
            if trial.distributions.get(name) == distribution
        }

    return search_space, next_cached_trial_number


class IntersectionSearchSpace:

    def __init__(self, include_pruned: bool = False) -> None:
        self._cached_trial_number: int = -1
        self._search_space: dict[str, BaseDistribution] | None = None
        self._study_id: int | None = None

        self._include_pruned = include_pruned

    def calculate(self, study: Study, use_cache: bool = False) -> dict[str, BaseDistribution]:
        pass


def intersection_search_space(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
) -> dict[str, BaseDistribution]:
    """Return the intersection search space of the given trials.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    .. note::
        :class:`~optuna.search_space.IntersectionSearchSpace` provides the same functionality with
        a much faster way. Please consider using it if you want to reduce execution time
        as much as possible.

    Args:
        trials:
            A list of trials.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions sorted by
        parameter names.
    """

    search_space, _ = _calculate(trials, include_pruned)
    search_space = search_space or {}
    search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))
    return search_space
