from __future__ import annotations

import datetime
from typing import Any
from typing import overload
from typing import TYPE_CHECKING

from optuna import distributions
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_func
from optuna._warnings import optuna_warn
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial._base import _SUGGEST_INT_POSITIONAL_ARGS
from optuna.trial._base import BaseTrial


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.distributions import CategoricalChoiceType

_suggest_deprecated_msg = "Use suggest_float{args} instead."


class FixedTrial(BaseTrial):

    def __init__(self, params: dict[str, Any], number: int = 0) -> None:
        self._params = params
        self._suggested_params: dict[str, Any] = {}
        self._distributions: dict[str, BaseDistribution] = {}
        self._user_attrs: dict[str, Any] = {}
        self._system_attrs: dict[str, Any] = {}
        self._datetime_start = datetime.datetime.now()
        self._number = number

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        return self._suggest(name, FloatDistribution(low, high, log=log, step=step))




    @convert_positional_args(
        previous_positional_arg_names=_SUGGEST_INT_POSITIONAL_ARGS,
        deprecated_version="3.5.0",
        removed_version="5.0.0",
    )
    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        return int(self._suggest(name, IntDistribution(low, high, log=log, step=step)))

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float: ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str: ...

    @overload
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType: ...

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        pass


    def set_user_attr(self, key: str, value: Any) -> None:
        self._user_attrs[key] = value

    @deprecated_func("3.1.0", "5.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:
        self._system_attrs[key] = value

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        if name not in self._params:
            raise ValueError(
                f"The value of the parameter '{name}' is not found. Please set it at "
                "the construction of the FixedTrial object."
            )

        value = self._params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            optuna_warn(
                f"The value {value} of the parameter '{name}' is out of "
                f"the range of the distribution {distribution}."
            )

        if name in self._distributions:
            distributions.check_distribution_compatibility(self._distributions[name], distribution)

        self._suggested_params[name] = value
        self._distributions[name] = distribution

        return value


    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        return self._distributions




