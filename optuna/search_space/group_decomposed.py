from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study


class _SearchSpaceGroup:
    def __init__(self) -> None:
        self._search_spaces: list[dict[str, BaseDistribution]] = []




class _GroupDecomposedSearchSpace:
    def __init__(self, include_pruned: bool = False) -> None:
        self._search_space = _SearchSpaceGroup()
        self._study_id: int | None = None
        self._include_pruned = include_pruned

