from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection


if TYPE_CHECKING:
    from typing import Literal

    import scipy.stats as ss

    import optuna
    from optuna.trial import FrozenTrial
else:
    from optuna._imports import _LazyImport

    ss = _LazyImport("scipy.stats")


@experimental_class("3.6.0")
class WilcoxonPruner(BasePruner):

    def __init__(
        self,
        *,
        p_threshold: float = 0.1,
        n_startup_steps: int = 2,
    ) -> None:
        if n_startup_steps < 0:  # TODO: Consider changing the RHS to 2.
            raise ValueError(f"n_startup_steps must be nonnegative but got {n_startup_steps}.")
        if not 0.0 <= p_threshold <= 1.0:
            raise ValueError(f"p_threshold must be between 0 and 1 but got {p_threshold}.")

        self._n_startup_steps = n_startup_steps
        self._p_threshold = p_threshold

