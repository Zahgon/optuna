from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch

    from optuna._gp import batched_lbfgsb
    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")
    batched_lbfgsb = _LazyImport("optuna._gp.batched_lbfgsb")


_logger = get_logger(__name__)


def _gradient_ascent_batched(
    acqf: BaseAcquisitionFunc,
    initial_params_batched: np.ndarray,
    initial_fvals: np.ndarray,
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function optimizes the acquisition function using preconditioning.
    Preconditioning equalizes the variances caused by each parameter and
    speeds up the convergence.

    In Optuna, acquisition functions use Matern 5/2 kernel, which is a function of `x / l`
    where `x` is `normalized_params` and `l` is the corresponding lengthscales.
    Then acquisition functions are a function of `x / l`, i.e. `f(x / l)`.
    As `l` has different values for each param, it makes the function ill-conditioned.
    By transforming `x / l` to `zl / l = z`, the function becomes `f(z)` and has
    equal variances w.r.t. `z`.
    So optimization w.r.t. `z` instead of `x` is the preconditioning here and
    speeds up the convergence.
    As the domain of `x` is [0, 1], that of `z` becomes [0, 1/l].
    """
    assert initial_params_batched.ndim == 2
    if len(continuous_indices) == 0:
        return initial_params_batched, initial_fvals, np.zeros(len(initial_fvals), dtype=bool)


    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_cont_xs_opt, neg_fvals_opt, n_iterations = batched_lbfgsb.batched_lbfgsb(
            func_and_grad=negative_acqf_with_grad,
            x0_batched=initial_params_batched[:, continuous_indices] / lengthscales,
            batched_args=([param for param in initial_params_batched.copy()],),
            bounds=[(0, 1 / s) for s in lengthscales],
            pgtol=math.sqrt(tol),
            max_iters=200,
        )

    xs_opt = initial_params_batched.copy()
    xs_opt[:, continuous_indices] = scaled_cont_xs_opt * lengthscales
    fvals_opt = -neg_fvals_opt
    is_updated_batch = (fvals_opt > initial_fvals) & (n_iterations > 0)

    return (
        np.where(is_updated_batch[:, None], xs_opt, initial_params_batched),
        np.where(is_updated_batch, fvals_opt, initial_fvals),
        is_updated_batch,
    )








def _local_search_discrete_batched(
    acqf: BaseAcquisitionFunc,
    initial_params_batched: np.ndarray,
    initial_fvals: np.ndarray,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_normalized_params_batched = initial_params_batched.copy()
    best_fvals = initial_fvals.copy()

    is_updated_batch = np.zeros(len(initial_fvals), dtype=bool)
    for batch, normalized_params in enumerate(initial_params_batched):
        best_normalized_params, best_fval, updated = _local_search_discrete(
            acqf, normalized_params, best_fvals[batch], param_idx, choices, xtol
        )
        best_normalized_params_batched[batch] = best_normalized_params
        best_fvals[batch] = best_fval
        is_updated_batch[batch] = updated

    return best_normalized_params_batched, best_fvals, is_updated_batch


def local_search_mixed_batched(
    acqf: BaseAcquisitionFunc, xs0: np.ndarray, *, tol: float = 1e-4, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    lengthscales = acqf.length_scales[(cont_inds := acqf.search_space.continuous_indices)]
    discrete_indices = acqf.search_space.discrete_indices
    choices_of_discrete_params = acqf.search_space.get_choices_of_discrete_params()
    discrete_xtols = [
        np.min(np.diff(choices), initial=np.inf) / 4
        for choices in choices_of_discrete_params
    ]
    best_fvals = acqf.eval_acqf_no_grad((best_xs := xs0.copy()))
    CONTINUOUS = -1
    last_changed_dims = np.full(len(best_xs), CONTINUOUS, dtype=int)
    remaining_inds = np.arange(len(best_xs))
    for _ in range(max_iter):
        best_xs[remaining_inds], best_fvals[remaining_inds], updated = _gradient_ascent_batched(
            acqf, best_xs[remaining_inds], best_fvals[remaining_inds], cont_inds, lengthscales, tol
        )
        last_changed_dims = np.where(updated, CONTINUOUS, last_changed_dims)
        for i, choices, xtol in zip(discrete_indices, choices_of_discrete_params, discrete_xtols):
            last_changed_dims = last_changed_dims[~(is_converged := last_changed_dims == i)]
            remaining_inds = remaining_inds[~is_converged]
            if remaining_inds.size == 0:
                return best_xs, best_fvals
            best_xs[remaining_inds], best_fvals[remaining_inds], updated = (
                _local_search_discrete_batched(
                    acqf, best_xs[remaining_inds], best_fvals[remaining_inds], i, choices, xtol
                )
            )
            last_changed_dims = np.where(updated, i, last_changed_dims)

        remaining_inds = remaining_inds[~(is_converged := last_changed_dims == CONTINUOUS)]
        last_changed_dims = last_changed_dims[~is_converged]
        if remaining_inds.size == 0:
            return best_xs, best_fvals
    else:
        _logger.warning("local_search_mixed: Local search did not converge.")
    return best_xs, best_fvals


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.RandomState()

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert len(warmstart_normalized_params_array) <= n_local_search - 1, (
        "We must choose at least 1 best sampled point + given_initial_xs as start points."
    )

    sampled_xs = acqf.search_space.sample_normalized_params(n_preliminary_samples, rng=rng)

    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1, n_non_zero_probs_improvement
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning("Study already converged, so the number of local search is reduced.")
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    x_warmstarts = np.vstack([sampled_xs[chosen_idxs, :], warmstart_normalized_params_array])
    best_xs, best_fvals = local_search_mixed_batched(acqf, x_warmstarts, tol=tol)
    best_idx = np.argmax(best_fvals).item()
    return best_xs[best_idx], best_fvals[best_idx]
