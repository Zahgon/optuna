from __future__ import annotations

import binascii
import math
from typing import TYPE_CHECKING

import optuna


if TYPE_CHECKING:
    from collections.abc import Container
from optuna import logging
from optuna.pruners._base import BasePruner
from optuna.pruners._successive_halving import SuccessiveHalvingPruner
from optuna.trial._state import TrialState


_logger = logging.get_logger(__name__)


class HyperbandPruner(BasePruner):

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._pruners: list[SuccessiveHalvingPruner] = []
        self._bootstrap_count = bootstrap_count
        self._total_trial_allocation_budget = 0
        self._trial_allocation_budgets: list[int] = []
        self._n_brackets: int | None = None

        if not isinstance(self._max_resource, int) and self._max_resource != "auto":
            raise ValueError(
                "The 'max_resource' should be integer or 'auto'. "
                f"But max_resource = {self._max_resource}"
            )

        if self._bootstrap_count > 0 and self._max_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and max_resource == 'auto' "
                f"are mutually incompatible, bootstrap_count is {self._bootstrap_count}"
            )



    def _calculate_trial_allocation_budget(self, bracket_id: int) -> int:
        pass

    def _get_bracket_id(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> int:
        """Compute the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__.
        """

        if len(self._pruners) == 0:
            return 0

        assert self._n_brackets is not None
        n = (
            binascii.crc32(f"{study.study_name}_{trial.number}".encode())
            % self._total_trial_allocation_budget
        )
        for bracket_id in range(self._n_brackets):
            n -= self._trial_allocation_budgets[bracket_id]
            if n < 0:
                return bracket_id

        assert False, "This line should be unreachable."

    def _create_bracket_study(
        self, study: "optuna.study.Study", bracket_id: int
    ) -> "optuna.study.Study":
        class _BracketStudy(optuna.study.Study):
            _VALID_ATTRS = (
                "get_trials",
                "_get_trials",
                "directions",
                "direction",
                "_directions",
                "_storage",
                "_study_id",
                "pruner",
                "study_name",
                "_bracket_id",
                "sampler",
                "trials",
                "_is_multi_objective",
                "stop",
                "_study",
                "_thread_local",
            )

            def __init__(
                self, study: "optuna.study.Study", pruner: HyperbandPruner, bracket_id: int
            ) -> None:
                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=pruner,
                )
                self._study = study
                self._bracket_id = bracket_id

            def get_trials(
                self,
                deepcopy: bool = True,
                states: Container[TrialState] | None = None,
            ) -> list["optuna.trial.FrozenTrial"]:
                trials = super()._get_trials(deepcopy=deepcopy, states=states)
                pruner = self.pruner
                assert isinstance(pruner, HyperbandPruner)
                return [t for t in trials if pruner._get_bracket_id(self, t) == self._bracket_id]

            def stop(self) -> None:
                self._study.stop()

            def __getattribute__(self, attr_name):  # type: ignore
                if attr_name not in _BracketStudy._VALID_ATTRS:
                    raise AttributeError(f"_BracketStudy does not have attribute of '{attr_name}'")
                else:
                    return object.__getattribute__(self, attr_name)

        return _BracketStudy(study, self, bracket_id)
