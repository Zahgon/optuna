from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection


if TYPE_CHECKING:
    import optuna


@experimental_class("2.8.0")
class PatientPruner(BasePruner):

    def __init__(
        self, wrapped_pruner: BasePruner | None, patience: int, min_delta: float = 0.0
    ) -> None:
        if patience < 0:
            raise ValueError(f"patience cannot be negative but got {patience}.")

        if min_delta < 0:
            raise ValueError(f"min_delta cannot be negative but got {min_delta}.")

        self._wrapped_pruner = wrapped_pruner
        self._patience = patience
        self._min_delta = min_delta

