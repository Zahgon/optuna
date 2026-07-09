
from __future__ import annotations

import numpy as np

from optuna._warnings import optuna_warn
from optuna.study._multi_objective import _is_pareto_front


def _get_upper_bound_set(
    sorted_pareto_sols: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function follows Algorithm 2 of Lacour17.

    Args:
        sorted_pareto_sols: Pareto solutions sorted with respect to the first objective.
        ref_point: The reference point.

    Returns:
        upper_bound_set: The upper bound set, which is ``U(N)`` in the paper. The shape is
        ``(n_bounds, n_objectives)``.
        def_points: The defining points of each vector in ``U(N)``. The shape is
        ``(n_bounds, n_objectives, n_objectives)``.

    NOTE:
        ``pareto_sols`` corresponds to ``N`` and ``upper_bound_set`` corresponds to ``U(N)`` in the
        paper.
        ``def_points`` (the shape is ``(n_bounds, n_objectives, n_objectives)``) is not well
        explained in the paper, but basically, ``def_points[i, j] = z[j]`` of
        ``upper_bound_set[i]``.
    """
    (_, n_objectives) = sorted_pareto_sols.shape
    objective_indices = np.arange(n_objectives)
    skip_ineq_judge = np.eye(n_objectives, dtype=bool)
    skip_ineq_judge[:, 0] = True

    def update(sol: np.ndarray, ubs: np.ndarray, dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        is_dominated = np.all(sol < ubs, axis=-1)
        if not any(is_dominated):
            return ubs, dps

        dominated_dps = dps[is_dominated]
        n_bounds = dominated_dps.shape[0]
        update = sol >= np.max(np.where(skip_ineq_judge, -np.inf, dominated_dps), axis=-2)
        ubs_indices_to_update = np.tile(np.arange(n_bounds)[:, np.newaxis], n_objectives)[update]
        dimensions_to_update = np.tile(objective_indices, (n_bounds, 1))[update]
        assert ubs_indices_to_update.size == dimensions_to_update.size
        indices_for_sweeping = np.arange(dimensions_to_update.size)
        new_dps = dominated_dps[ubs_indices_to_update]
        new_dps[indices_for_sweeping, dimensions_to_update] = sol
        new_ubs = ubs[is_dominated][ubs_indices_to_update]
        new_ubs[indices_for_sweeping, dimensions_to_update] = sol[dimensions_to_update]
        return np.vstack([ubs[~is_dominated], new_ubs]), np.vstack([dps[~is_dominated], new_dps])

    upper_bound_set = np.asarray([ref_point])  # Line 1 of Alg. 2.
    def_points = np.full((1, n_objectives, n_objectives), -np.inf)  # z^k(z^r) = \hat{z}^k
    def_points[0, objective_indices, objective_indices] = ref_point  # \hat{z}^k is a dummy point.
    for solution in sorted_pareto_sols:  # NOTE(nabenabe): Sorted must be fulfilled.
        upper_bound_set, def_points = update(solution, upper_bound_set, def_points)

    return upper_bound_set, def_points


def _get_box_bounds(
    upper_bound_set: np.ndarray, def_points: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    n_objectives = upper_bound_set.shape[-1]
    assert n_objectives > 1, "This function is used only for multi-objective problems."
    bounds = np.empty((2, *upper_bound_set.shape))
    bounds[0, :, 0] = def_points[:, 0, 0]
    bounds[1, :, 0] = ref_point[0]
    row, col = np.diag_indices(n_objectives - 1)
    bounds[0, :, 1:] = np.maximum.accumulate(def_points, axis=-2)[:, row, col + 1]
    bounds[1, :, 1:] = upper_bound_set[:, 1:]
    not_empty = ~np.any(bounds[1] <= bounds[0], axis=-1)  # Remove [inf, inf] or [-inf, -inf].
    return bounds[:, not_empty]


def _get_non_dominated_box_bounds(
    sorted_pareto_sols: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    neg_upper_bound_set = -_get_upper_bound_set(sorted_pareto_sols, ref_point)[0]
    sorted_neg_upper_bound_set = np.unique(neg_upper_bound_set, axis=0)  # lexsort by np.unique.
    point_at_infinity = np.full_like(ref_point, np.inf)
    neg_lower_bound_set, neg_def_points = _get_upper_bound_set(
        sorted_pareto_sols=sorted_neg_upper_bound_set[
            _is_pareto_front(sorted_neg_upper_bound_set, assume_unique_lexsorted=True)
        ],
        ref_point=point_at_infinity,
    )
    box_upper_bounds, box_lower_bounds = -_get_box_bounds(
        neg_lower_bound_set, neg_def_points, point_at_infinity
    )
    return box_lower_bounds, box_upper_bounds


def get_non_dominated_box_bounds(
    loss_vals: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    assert np.all(np.isfinite(loss_vals)), "loss_vals must be clipped before box decomposition."
    unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
    sorted_pareto_sols = unique_lexsorted_loss_vals[
        _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
    ]
    n_objectives = loss_vals.shape[-1]
    assert n_objectives > 1, "This function is used only for multi-objective problems."
    if n_objectives > 4:
        optuna_warn(
            "Box decomposition (typically used by `GPSampler`) might be significantly slow for "
            "n_objectives > 4. Please consider using another sampler instead."
        )

    return _get_non_dominated_box_bounds(sorted_pareto_sols, ref_point)
