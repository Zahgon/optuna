from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

from optuna.pruners import BasePruner
from optuna.pruners._percentile import _is_first_in_interval_step


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _check_value(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        message = (
            f"The `value` argument is of type '{type(value).__name__}' but supposed to be a float."
        )
        raise TypeError(message) from None

    return value


class ThresholdPruner(BasePruner):

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
    ) -> None:
        if lower is None and upper is None:
            raise TypeError("Either lower or upper must be specified.")
        if lower is not None:
            lower = _check_value(lower)
        if upper is not None:
            upper = _check_value(upper)

        lower = lower if lower is not None else -float("inf")
        upper = upper if upper is not None else float("inf")

        if lower > upper:
            raise ValueError("lower should be smaller than upper.")
        if n_warmup_steps < 0:
            raise ValueError(
                f"Number of warmup steps cannot be negative but got {n_warmup_steps}."
            )
        if interval_steps < 1:
            raise ValueError(
                f"Pruning interval steps must be at least 1 but got {interval_steps}."
            )

        self._lower = lower
        self._upper = upper
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

