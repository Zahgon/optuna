from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution


class _SearchSpaceTransform:

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        transform_log: bool = True,
        transform_step: bool = True,
        transform_0_1: bool = False,
    ) -> None:
        bounds, column_to_encoded_columns, encoded_column_to_column = _transform_search_space(
            search_space, transform_log, transform_step
        )
        self._raw_bounds = bounds
        self._column_to_encoded_columns = column_to_encoded_columns
        self._encoded_column_to_column = encoded_column_to_column
        self._search_space = search_space
        self._transform_log = transform_log
        self._transform_0_1 = transform_0_1




    def transform(self, params: dict[str, Any]) -> np.ndarray:
        """Transform a parameter configuration from actual values to continuous space.

        Args:
            params:
                A parameter configuration to transform.

        Returns:
            A 1-dimensional ``numpy.ndarray`` holding the transformed parameters in the
            configuration.

        """
        trans_params = np.zeros(self._raw_bounds.shape[0], dtype=np.float64)

        bound_idx = 0
        for name, distribution in self._search_space.items():
            assert name in params, "Parameter configuration must contain all distributions."
            param = params[name]

            if isinstance(distribution, CategoricalDistribution):
                choice_idx = int(distribution.to_internal_repr(param))
                trans_params[bound_idx + choice_idx] = 1
                bound_idx += len(distribution.choices)
            else:
                trans_params[bound_idx] = _transform_numerical_param(
                    param, distribution, self._transform_log
                )
                bound_idx += 1

        if self._transform_0_1:
            single_mask = self._raw_bounds[:, 0] == self._raw_bounds[:, 1]
            trans_params[single_mask] = 0.5
            trans_params[~single_mask] = (
                trans_params[~single_mask] - self._raw_bounds[~single_mask, 0]
            ) / (self._raw_bounds[~single_mask, 1] - self._raw_bounds[~single_mask, 0])

        return trans_params

    def untransform(self, trans_params: np.ndarray) -> dict[str, Any]:
        """Untransform a parameter configuration from continuous space to actual values.

        Args:
            trans_params:
                A 1-dimensional ``numpy.ndarray`` in the transformed space corresponding to a
                parameter configuration.

        Returns:
            A dictionary of an untransformed parameter configuration. Keys are parameter names.
            Values are untransformed parameter values.

        """
        assert trans_params.shape == (self._raw_bounds.shape[0],)

        if self._transform_0_1:
            trans_params = self._raw_bounds[:, 0] + trans_params * (
                self._raw_bounds[:, 1] - self._raw_bounds[:, 0]
            )

        params = {}

        for (name, distribution), encoded_columns in zip(
            self._search_space.items(), self.column_to_encoded_columns
        ):
            trans_param = trans_params[encoded_columns]

            if isinstance(distribution, CategoricalDistribution):
                param = distribution.to_external_repr(trans_param.argmax())
            else:
                param = _untransform_numerical_param(
                    trans_param.item(), distribution, self._transform_log
                )

            params[name] = param

        return params


def _transform_search_space(
    search_space: dict[str, BaseDistribution], transform_log: bool, transform_step: bool
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    assert len(search_space) > 0, "Cannot transform if no distributions are given."

    n_bounds = sum(
        len(d.choices) if isinstance(d, CategoricalDistribution) else 1
        for d in search_space.values()
    )

    bounds = np.empty((n_bounds, 2), dtype=np.float64)
    column_to_encoded_columns: list[np.ndarray] = []
    encoded_column_to_column = np.empty(n_bounds, dtype=np.int64)

    bound_idx = 0
    for distribution in search_space.values():
        d = distribution
        if isinstance(d, CategoricalDistribution):
            n_choices = len(d.choices)
            bounds[bound_idx : bound_idx + n_choices] = (0, 1)  # Broadcast across all choices.
            encoded_columns = np.arange(bound_idx, bound_idx + n_choices)
            encoded_column_to_column[encoded_columns] = len(column_to_encoded_columns)
            column_to_encoded_columns.append(encoded_columns)
            bound_idx += n_choices
        elif isinstance(
            d,
            (
                FloatDistribution,
                IntDistribution,
            ),
        ):
            if isinstance(d, FloatDistribution):
                if d.step is not None:
                    half_step = 0.5 * d.step if transform_step else 0.0
                    bds = (
                        _transform_numerical_param(d.low, d, transform_log) - half_step,
                        _transform_numerical_param(d.high, d, transform_log) + half_step,
                    )
                else:
                    bds = (
                        _transform_numerical_param(d.low, d, transform_log),
                        _transform_numerical_param(d.high, d, transform_log),
                    )
            elif isinstance(d, IntDistribution):
                half_step = 0.5 * d.step if transform_step else 0.0
                if d.log:
                    bds = (
                        _transform_numerical_param(d.low - half_step, d, transform_log),
                        _transform_numerical_param(d.high + half_step, d, transform_log),
                    )
                else:
                    bds = (
                        _transform_numerical_param(d.low, d, transform_log) - half_step,
                        _transform_numerical_param(d.high, d, transform_log) + half_step,
                    )
            else:
                assert False, "Should not reach. Unexpected distribution."

            bounds[bound_idx] = bds
            encoded_column = np.atleast_1d(bound_idx)
            encoded_column_to_column[encoded_column] = len(column_to_encoded_columns)
            column_to_encoded_columns.append(encoded_column)
            bound_idx += 1
        else:
            assert False, "Should not reach. Unexpected distribution."

    assert bound_idx == n_bounds

    return bounds, column_to_encoded_columns, encoded_column_to_column


def _transform_numerical_param(
    param: int | float, distribution: BaseDistribution, transform_log: bool
) -> float:
    d = distribution

    if isinstance(d, CategoricalDistribution):
        assert False, "Should not reach. Should be one-hot encoded."
    elif isinstance(d, FloatDistribution):
        if d.log:
            trans_param = math.log(param) if transform_log else float(param)
        else:
            trans_param = float(param)
    elif isinstance(d, IntDistribution):
        if d.log:
            trans_param = math.log(param) if transform_log else float(param)
        else:
            trans_param = float(param)
    else:
        assert False, "Should not reach. Unexpected distribution."

    return trans_param


def _untransform_numerical_param(
    trans_param: float, distribution: BaseDistribution, transform_log: bool
) -> int | float:
    d = distribution

    if isinstance(d, CategoricalDistribution):
        assert False, "Should not reach. Should be one-hot encoded."
    elif isinstance(d, FloatDistribution):
        if d.log:
            param = math.exp(trans_param) if transform_log else trans_param
            if d.single():
                pass
            else:
                param = min(param, np.nextafter(d.high, d.high - 1))
        elif d.step is not None:
            param = float(
                np.clip(np.round((trans_param - d.low) / d.step) * d.step + d.low, d.low, d.high)
            )
        else:
            if d.single():
                param = trans_param
            else:
                param = min(trans_param, np.nextafter(d.high, d.high - 1))
    elif isinstance(d, IntDistribution):
        if d.log:
            if transform_log:
                param = int(np.clip(np.round(math.exp(trans_param)), d.low, d.high))
            else:
                param = int(trans_param)
        else:
            param = int(
                np.clip(np.round((trans_param - d.low) / d.step) * d.step + d.low, d.low, d.high)
            )
    else:
        assert False, "Should not reach. Unexpected distribution."

    return param
