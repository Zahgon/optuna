from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna._deprecated import deprecated_class
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func


if TYPE_CHECKING:
    from optuna.trial import FrozenTrial


@experimental_class("2.8.0")
class RetryHeartbeatStaleTrialCallback:

    def __init__(
        self, max_retry: int | None = None, inherit_intermediate_values: bool = False
    ) -> None:
        self._max_retry = max_retry
        self._inherit_intermediate_values = inherit_intermediate_values

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        system_attrs: dict[str, Any] = {
            "failed_trial": trial.number,
            "retry_history": [],
            **trial.system_attrs,
        }
        system_attrs["retry_history"].append(trial.number)
        if self._max_retry is not None:
            if self._max_retry < len(system_attrs["retry_history"]):
                return

        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.WAITING,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=system_attrs,
                intermediate_values=(
                    trial.intermediate_values if self._inherit_intermediate_values else None
                ),
            )
        )

    @staticmethod
    @experimental_func("2.8.0")
    def retried_trial_number(trial: FrozenTrial) -> int | None:
        pass

    @staticmethod
    @experimental_func("3.0.0")
    def retry_history(trial: FrozenTrial) -> list[int]:
        pass


@deprecated_class(
    "4.9.0",
    "6.0.0",
    text="Use `RetryHeartbeatStaleTrialCallback` instead.",
)
class RetryFailedTrialCallback(RetryHeartbeatStaleTrialCallback):

    pass
