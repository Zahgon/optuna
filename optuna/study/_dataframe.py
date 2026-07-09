from __future__ import annotations

import collections
from typing import Any

import optuna
from optuna._imports import try_import
from optuna.trial._state import TrialState


with try_import() as _imports:
    import pandas as pd

if not _imports.is_successful():
    pd = object  # NOQA

__all__ = ["pd"]






