from __future__ import annotations

import abc

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._warnings import optuna_warn
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.trial import TrialState


_DEPRECATION_WARNING_MESSAGE = _DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.terminator` module",
    d_ver="4.9.0",
    r_ver="6.0.0",
)


class BaseTerminator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


class Terminator(BaseTerminator):

    def __init__(
        self,
        improvement_evaluator: BaseImprovementEvaluator | None = None,
        error_evaluator: BaseErrorEvaluator | None = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        optuna_warn(_DEPRECATION_WARNING_MESSAGE, FutureWarning)

        if min_n_trials <= 0:
            raise ValueError("`min_n_trials` is expected to be a positive integer.")

        self._improvement_evaluator = improvement_evaluator or RegretBoundEvaluator()
        self._error_evaluator = error_evaluator or self._initialize_error_evaluator()
        self._min_n_trials = min_n_trials

    def _initialize_error_evaluator(self) -> BaseErrorEvaluator:
        if isinstance(self._improvement_evaluator, BestValueStagnationEvaluator):
            return StaticErrorEvaluator(constant=0)
        return CrossValidationErrorEvaluator()

    def should_terminate(self, study: Study) -> bool:
        pass
