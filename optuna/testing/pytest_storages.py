from __future__ import annotations

import copy
from datetime import datetime
import math
import pickle
import random
import sys
from time import sleep
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna._typing import JSONSerializable
    from optuna.storages import BaseStorage


@pytest.fixture
def storage() -> BaseStorage:
    raise NotImplementedError


class StorageTestCase:
    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass


ALL_STATES = list(TrialState)

EXAMPLE_ATTRS: dict[str, JSONSerializable] = {
    "dataset": "MNIST",
    "none": None,
    "json_serializable": {"baseline_score": 0.001, "tags": ["image", "classification"]},
}

FLOAT_ATTRS = {
    "zero": 0,
    "pi": math.pi,
    "max": sys.float_info.max,
    "negative max": -sys.float_info.max,
    "min": sys.float_info.min,
    "negative min": -sys.float_info.min,
    "inf": float("inf"),
    "negative inf": -float("inf"),
    "nan": float("nan"),
}


def is_equal_floats(a: float, b: float) -> bool:
    if math.isnan(a):
        return math.isnan(b)
    if math.isnan(b):
        return False
    return a == b


def _test_set_and_get_study_user_attrs_for_floats(storage: BaseStorage) -> None:
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    for key, value in FLOAT_ATTRS.items():
        storage.set_study_user_attr(study_id, key, value)
        assert is_equal_floats(storage.get_study_user_attrs(study_id)[key], value)


def _test_set_and_get_study_system_attrs_for_floats(storage: BaseStorage) -> None:
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    for key, value in FLOAT_ATTRS.items():
        storage.set_study_system_attr(study_id, key, value)
        assert is_equal_floats(storage.get_study_system_attrs(study_id)[key], value)


def _test_set_trial_state_values_for_floats(storage: BaseStorage) -> None:
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    for value in FLOAT_ATTRS.values():
        if math.isnan(value):  # NOTE: Optuna does not accept `nan` as `value`.
            continue
        trial_id = storage.create_new_trial(study_id)
        storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=(value,))
        set_value = storage.get_trial(trial_id).value
        assert set_value is not None
        assert is_equal_floats(set_value, value)


def _test_set_and_get_trial_param_for_floats(storage: BaseStorage) -> None:
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)

    for key, value in FLOAT_ATTRS.items():
        float_distribution = FloatDistribution(low=value, high=value)
        categorical_distribution = CategoricalDistribution(choices=(value,))
        for distribution in (float_distribution, categorical_distribution):
            if isinstance(distribution, FloatDistribution) and not math.isfinite(value):
                continue
            param_name = distribution.__class__.__name__ + key
            internal_repr = distribution.to_internal_repr(value)
            storage.set_trial_param(trial_id, param_name, internal_repr, distribution)
            assert is_equal_floats(storage.get_trial_param(trial_id, param_name), internal_repr)
            assert storage.get_trial(trial_id).distributions[param_name] == distribution


def _test_set_trial_intermediate_value_for_floats(storage: BaseStorage) -> None:
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)
    for i, value in enumerate(FLOAT_ATTRS.values()):
        storage.set_trial_intermediate_value(trial_id, i, value)
        assert is_equal_floats(storage.get_trial(trial_id).intermediate_values[i], value)


def _test_set_and_get_trial_user_attr_for_floats(storage: BaseStorage) -> None:
    trial_id = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    for key, value in FLOAT_ATTRS.items():
        storage.set_trial_user_attr(trial_id, key, value)
        assert is_equal_floats(storage.get_trial_user_attrs(trial_id)[key], value)


def _test_set_and_get_trial_system_attr_for_floats(storage: BaseStorage) -> None:
    trial_id = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    for key, value in FLOAT_ATTRS.items():
        storage.set_trial_system_attr(trial_id, key, value)
        assert is_equal_floats(storage.get_trial_system_attrs(trial_id)[key], value)


pass


pass
