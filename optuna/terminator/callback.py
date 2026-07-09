from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._warnings import optuna_warn
from optuna.logging import get_logger
from optuna.terminator.terminator import Terminator


if TYPE_CHECKING:
    from optuna.study.study import Study
    from optuna.terminator.terminator import BaseTerminator
    from optuna.trial import FrozenTrial


_logger = get_logger(__name__)

_DEPRECATION_WARNING_MESSAGE = _DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.terminator` module",
    d_ver="4.9.0",
    r_ver="6.0.0",
)


class TerminatorCallback:

    def __init__(self, terminator: BaseTerminator | None = None) -> None:
        optuna_warn(_DEPRECATION_WARNING_MESSAGE, FutureWarning)
        self._terminator = terminator or Terminator()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        should_terminate = self._terminator.should_terminate(study=study)

        if should_terminate:
            _logger.info("The study has been stopped by the terminator.")
            study.stop()
