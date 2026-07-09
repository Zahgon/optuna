from __future__ import annotations

from dataclasses import dataclass
import decimal
from functools import lru_cache
import math
from numbers import Real
import sys
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import create_trial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    ChoicesArgsType = tuple[int | float, int | float, int | float | None]  # low, high, step


@dataclass(frozen=True)
class _UnexpandedTreeNode:
    is_running: bool = False

    def is_any_expandable(self, exclude_running: bool) -> bool:
        return True

    def count_unexpanded(self, exclude_running: bool) -> int:
        return 1


_UNEXPANDED_NODE = _UnexpandedTreeNode()


@dataclass(**({"slots": True} if sys.version_info >= (3, 10) else {}))
class _TreeNode:

    param_name: str | None = None
    children: dict[float, _TreeNode | _UnexpandedTreeNode] | None = None
    is_running: bool = False
    choices_args: ChoicesArgsType | None = None

    def _validate_search_space_consistency(
        self, param_name: str | None, choices_args: ChoicesArgsType | None
    ) -> None:
        if self.param_name != param_name:
            raise ValueError(f"param_name mismatch: {self.param_name} != {param_name}")
        if choices_args != self.choices_args:
            assert self.children is not None and choices_args is not None
            choices_old = list(self.children)
            choices_new = _enumerate_candidates(*choices_args)
            raise ValueError(
                f"search_space mismatch in {param_name}: {choices_old} != {choices_new}"
            )

    def expand(self, param_name: str | None, choices_args: ChoicesArgsType) -> None:
        if self.children is None:
            self.param_name = param_name
            choices = _enumerate_candidates(*choices_args)
            self.children = {value: _UNEXPANDED_NODE for value in choices}
            self.choices_args = choices_args
        else:
            self._validate_search_space_consistency(param_name, choices_args)

    def set_running(self) -> None:
        self.is_running = True

    def set_leaf(self) -> None:
        if self.children is not None:
            self._validate_search_space_consistency(None, None)
        self.children = {}

    def add_path(self, trial_path: list[tuple[str, ChoicesArgsType, float]]) -> _TreeNode | None:
        current_node = self
        for param_name, choices_args, value in trial_path:
            current_node.expand(param_name, choices_args)
            if not (children := current_node.children):  # children is empty or None.
                return None
            elif (next_node := children.get(value)) is None:
                return None
            elif next_node is _UNEXPANDED_NODE:
                next_node = _TreeNode()
                children[value] = next_node
            current_node = cast(_TreeNode, next_node)
        return current_node

    def is_any_expandable(self, exclude_running: bool) -> bool:
        if (children := self.children) is None:
            return not exclude_running or not self.is_running
        return any(child.is_any_expandable(exclude_running) for child in children.values())

    def count_unexpanded(self, exclude_running: bool) -> int:
        if (children := self.children) is None:
            return 0 if exclude_running and self.is_running else 1
        return sum(child.count_unexpanded(exclude_running) for child in children.values())

    def sample_child(self, rng: np.random.RandomState, exclude_running: bool) -> float:
        assert (children := self.children) is not None
        unexpanded_counts = np.array(
            [child.count_unexpanded(exclude_running) for child in children.values()], dtype=float
        )

        alpha = 0.5
        weights_orig = unexpanded_counts / unexpanded_counts.sum()
        weights_flat = np.where(unexpanded_counts > 0, 1.0, 0.0)
        weights_flat /= weights_flat.sum()

        weights = (1.0 - alpha) * weights_orig + alpha * weights_flat
        if any(
            not value.is_running and weights[i] > 0 for i, value in enumerate(children.values())
        ):
            for i, child in enumerate(children.values()):
                if child.is_running:
                    weights[i] = 0.0
        weights /= weights.sum()
        return rng.choice(list(children.keys()), p=weights).item()


def _get_non_waiting_trials_and_current_trial_index(
    study: Study, current_trial_number: int
) -> tuple[list[FrozenTrial], int]:
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)
    for i in range(1, len(trials) + 1):
        t = trials[-i]
        if t.number == current_trial_number:
            return trials, len(trials) - i
    assert False, "Should not reach"


@experimental_class("3.1.0")
class BruteForceSampler(BaseSampler):

    def __init__(self, seed: int | None = None, avoid_premature_stop: bool = False) -> None:
        self._rng = LazyRandomState(seed)
        self._avoid_premature_stop = avoid_premature_stop


    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    @staticmethod
    def _populate_tree(tree: _TreeNode, trials: list[FrozenTrial], params: dict[str, Any]) -> None:
        cat_internal_repr_cache: dict[str, dict[CategoricalChoiceType, float]] = {}
        params_items = params.items()
        nonnan_params_items = {k: v for k, v in params_items if not _is_nan(v)}.items()
        nan_param_names = [k for k, v in params_items if _is_nan(v)]

        def _get_trial_path(trial: FrozenTrial) -> list[tuple[str, ChoicesArgsType, float]]:
            trial_path: list[tuple[str, ChoicesArgsType, float]] = []
            trial_params = trial.params
            for name, dist in trial.distributions.items():
                if name in params:
                    continue
                if name not in cat_internal_repr_cache:
                    cat_internal_repr_cache[name] = {}
                    if isinstance(dist, CategoricalDistribution):
                        cat_internal_repr_cache[name] = {c: i for i, c in enumerate(dist.choices)}
                if cat_repr := cat_internal_repr_cache[name]:
                    if (value := cat_repr.get(param_val := trial_params[name])) is None:
                        value = dist.to_internal_repr(param_val)  # most likely param_val is nan.
                    dist = cast(CategoricalDistribution, dist)  # mypy redefinition.
                    trial_path.append((name, (0, len(dist.choices) - 1, 1), value))
                else:
                    dist = cast("IntDistribution | FloatDistribution", dist)  # mypy redefinition.
                    trial_path.append((name, (dist.low, dist.high, dist.step), trial_params[name]))
            return trial_path

        for trial in trials:
            if params:
                trial_params = trial.params
                if not (nonnan_params_items <= trial_params.items()):
                    continue
                if not all(_is_nan(trial_params.get(p)) for p in nan_param_names):
                    continue
            if (leaf := tree.add_path(_get_trial_path(trial))) is not None:
                if trial.state.is_finished():
                    leaf.set_leaf()
                else:
                    leaf.set_running()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        exclude_running = not self._avoid_premature_stop
        trials, current_idx = _get_non_waiting_trials_and_current_trial_index(study, trial.number)
        trials.pop(current_idx)
        tree = _TreeNode()
        if isinstance(param_distribution, CategoricalDistribution):
            c_args: ChoicesArgsType = (0, len(param_distribution.choices) - 1, 1)
        elif isinstance(param_distribution, (IntDistribution, FloatDistribution)):
            c_args = (param_distribution.low, param_distribution.high, param_distribution.step)
        else:
            assert False, "Should not reach."
        tree.expand(param_name, c_args)
        self._populate_tree(tree, trials, trial.params)
        if tree.is_any_expandable(exclude_running):
            param_val = tree.sample_child(self._rng.rng, exclude_running)
        else:
            choices = _enumerate_candidates(*c_args)
            param_val = self._rng.rng.choice(choices).item()
        return param_distribution.to_external_repr(param_val)

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        exclude_running = not self._avoid_premature_stop
        trials, current_idx = _get_non_waiting_trials_and_current_trial_index(study, trial.number)
        trials[current_idx] = create_trial(
            state=state, values=values, params=trial.params, distributions=trial.distributions
        )
        params = trial.params.copy()
        for param_name in reversed(trial.params.keys()):
            params.pop(param_name)
            tree = _TreeNode()
            self._populate_tree(tree, trials, params)
            if tree.is_any_expandable(exclude_running):
                return
        study.stop()


def _is_nan(v: CategoricalChoiceType) -> bool:
    return isinstance(v, Real) and math.isnan(float(v))


@lru_cache
def _enumerate_candidates(
    low: int | float, high: int | float, step: int | float | None
) -> tuple[float, ...]:
    if step is None:
        raise ValueError(
            "FloatDistribution.step must be given for BruteForceSampler"
            " (otherwise, the search space will be infinite)."
        )
    if isinstance(low, int) and isinstance(high, int) and isinstance(step, int):
        return tuple(range(low, high + 1, step))
    else:
        low_ = decimal.Decimal(str(low))
        high_ = decimal.Decimal(str(high))
        step_ = decimal.Decimal(str(step))
        ret = []
        while low_ <= high_:
            ret.append(float(low_))
            low_ += step_
        return tuple(ret)
