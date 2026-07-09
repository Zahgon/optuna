from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.pruners import BasePruner


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class NopPruner(BasePruner):

    pass
