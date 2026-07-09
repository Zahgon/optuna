from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna import logging
from optuna._warnings import optuna_warn


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from optuna.study._study_direction import StudyDirection
    from optuna.trial import FrozenTrial

_logger = logging.get_logger(__name__)


class StudySummary:

    def __init__(
        self,
        study_name: str,
        direction: StudyDirection | None,
        best_trial: FrozenTrial | None,
        user_attrs: dict[str, Any],
        system_attrs: dict[str, Any],
        n_trials: int,
        datetime_start: datetime | None,
        study_id: int,
        *,
        directions: Sequence[StudyDirection] | None = None,
    ):
        self.study_name = study_name
        if direction is None and directions is None:
            raise ValueError("Specify one of `direction` and `directions`.")
        elif directions is not None:
            self._directions = list(directions)
        elif direction is not None:
            self._directions = [direction]
        else:
            raise ValueError("Specify only one of `direction` and `directions`.")
        self.best_trial = best_trial
        self.user_attrs = user_attrs
        self._system_attrs = system_attrs
        self.n_trials = n_trials
        self.datetime_start = datetime_start
        self._study_id = study_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id <= other._study_id



