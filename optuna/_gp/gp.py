
from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna._warnings import optuna_warn
from optuna.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy
    import torch
else:
    from optuna._imports import _LazyImport

    scipy = _LazyImport("scipy")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


def warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    optuna_warn("Clip non-finite values to the min/max finite values for GP fittings.")
    is_any_finite = np.any(is_values_finite, axis=0)
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


def _solve_cholesky(L: torch.Tensor, B: torch.Tensor, *, left: bool = True) -> torch.Tensor:
    """
    This function returns the tensor `X` by solving the linear system `A @ X = B`,
    where `A = L @ L.T`.

    if ``left=False``, solve `X @ A = B` instead.

    NOTE(nabenabe): torch.cholesky_solve is legacy and slower based on my benchmarking.
    NOTE(nabenabe): Don't use np.linalg.inv because it is too slow und unstable.
    cf. https://github.com/optuna/optuna/issues/6230
    """
    if left:
        return torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, B, upper=False), upper=True
        )
    else:
        return torch.linalg.solve_triangular(
            L,
            torch.linalg.solve_triangular(L.T, B, upper=True, left=False),
            upper=False,
            left=False,
        )


def _extend_cholesky(L11: torch.Tensor, K21: torch.Tensor, K22: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the Cholesky decompsition L of K=[[K11,K12],[K21,K22]] by
    extending L11 where K11 = L11 @ L11.T. Note that K12 = K21.T.

    The solution L = chol(K) is calculated as:
        chol(K) = [[L11, 0], [K21 @ inv(L11).T, chol(K22 - K21 @ inv(K11) @ K21.T)]].

    Denote L21 := K21 @ inv(L11).T.
    Since inv(L11.T).T = inv(L11), L21 = K21 @ inv(L11.T) --> L21.T = inv(L11) @ K21.T
    --> Solving L11 @ L21.T = K21.T yields L21.T.
    Note that L21 = K21 @ inv(L11).T = K21 @ inv(L11.T).

    Since inv(K11) = inv(L11 @ L11.T) = inv(L11.T) @ inv(L11),
    K21 @ inv(K11) @ K21.T = K21 @ inv(L11.T) @ inv(L11) @ K21.T = L21 @ L21.T.
    """
    n1 = L11.shape[-1]
    n2 = K22.shape[-1]
    batch_shape = L11.shape[:-2]
    L = torch.zeros(batch_shape + (n1 + n2, n1 + n2), dtype=torch.float64)
    L21_T = torch.linalg.solve_triangular(L11, K21.transpose(-1, -2), upper=False)
    L21 = L21_T.transpose(-1, -2)
    L[..., :n1, :n1] = L11
    L[..., n1:, n1:] = torch.linalg.cholesky(K22 - L21 @ L21_T)
    L[..., n1:, :n1] = L21
    return L


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        """
        Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function, then
        deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        """
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class GPRegressor:
    def __init__(
        self,
        is_categorical: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        inverse_squared_lengthscales: torch.Tensor,  # (len(params), )
        kernel_scale: torch.Tensor,  # Scalar
        noise_var: torch.Tensor,  # Scalar
    ) -> None:
        assert len(X_train.shape) == 2 and len(y_train.shape) == 1
        self._is_categorical = is_categorical
        self._X_train = X_train
        self._y_train = y_train.unsqueeze(-1)
        self._X_all = X_train
        self._y_all = y_train.unsqueeze(-1)
        self._squared_X_diff = (X_train.unsqueeze(-2) - X_train.unsqueeze(-3)).square_()
        if self._is_categorical.any():
            self._squared_X_diff[..., self._is_categorical] = (
                self._squared_X_diff[..., self._is_categorical] > 0.0
            ).type(torch.float64)
        self._cov_Y_Y_chol: torch.Tensor | None = None
        self._cov_Y_Y_inv_Y: torch.Tensor | None = None
        self.inverse_squared_lengthscales = inverse_squared_lengthscales
        self.kernel_scale = kernel_scale
        self.noise_var = noise_var


    def _cache_matrix(self) -> None:
        assert self._cov_Y_Y_chol is None and self._cov_Y_Y_inv_Y is None, (
            "Cannot call cache_matrix more than once."
        )
        self.inverse_squared_lengthscales = self.inverse_squared_lengthscales.detach()
        self.kernel_scale = self.kernel_scale.detach()
        self.noise_var = self.noise_var.detach()
        with torch.no_grad():
            cov_Y_Y = self.kernel()
        cov_Y_Y.diagonal().add_(self.noise_var)
        self._cov_Y_Y_chol = torch.linalg.cholesky(cov_Y_Y)
        self._cov_Y_Y_inv_Y = _solve_cholesky(self._cov_Y_Y_chol, self._y_train).squeeze(-1)

    def append_running_data(self, X_running: torch.Tensor, y_running: torch.Tensor) -> None:
        assert self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None, (
            "Call _cache_matrix before append_running_data"
        )
        with torch.no_grad():
            kernel_running_train = self.kernel(X_running)
            kernel_running_running = self.kernel(X_running, X_running)
        self._X_all = torch.cat([self._X_train, X_running], dim=0)
        self._y_all = torch.cat([self._y_train, y_running.unsqueeze(-1)], dim=0)
        kernel_running_running.diagonal().add_(self.noise_var)
        self._cov_Y_Y_chol = _extend_cholesky(
            L11=self._cov_Y_Y_chol, K21=kernel_running_train, K22=kernel_running_running
        )
        self._cov_Y_Y_inv_Y = _solve_cholesky(self._cov_Y_Y_chol, self._y_all).squeeze(-1)

    def kernel(
        self, X1: torch.Tensor | None = None, X2: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Return the kernel matrix with the shape of (..., n_A, n_B) given X1 and X2 each with the
        shapes of (..., n_A, len(params)) and (..., n_B, len(params)).

        If x1 and x2 have the shape of (len(params), ), kernel(x1, x2) is computed as:
            kernel_scale * Matern52Kernel.apply(
                sqd(x1, x2) @ inverse_squared_lengthscales
            )
        where if x1[i] is continuous, sqd(x1, x2)[i] = (x1[i] - x2[i]) ** 2 and if x1[i] is
        categorical, sqd(x1, x2)[i] = int(x1[i] != x2[i]).
        Note that the distance for categorical parameters is the Hamming distance.
        """
        if X1 is None:
            assert X2 is None
            sqd = self._squared_X_diff
        else:
            if X2 is None:
                X2 = self._X_train

            sqd = (X1 - X2 if X1.ndim == 1 else X1.unsqueeze(-2) - X2.unsqueeze(-3)).square_()
            if self._is_categorical.any():
                sqd[..., self._is_categorical] = (sqd[..., self._is_categorical] > 0.0).type(
                    torch.float64
                )
        sqdist = sqd.matmul(self.inverse_squared_lengthscales)
        return Matern52Kernel.apply(sqdist) * self.kernel_scale  # type: ignore

    def posterior(self, x: torch.Tensor, joint: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes the posterior mean and variance given the points `x` where both mean
        and variance tensors will have the shape of x.shape[:-1].
        If ``joint=True``, the joint posterior will be computed.

        The posterior mean and variance are computed as:
            mean = cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ y, and
            var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ cov_fx_fX.T.

        Please note that we clamp the variance to avoid negative values due to numerical errors.
        """
        assert self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None, (
            "Call cache_matrix before calling posterior."
        )
        is_single_point = x.ndim == 1
        x_ = x if not is_single_point else x.unsqueeze(0)
        mean = torch.linalg.vecdot(cov_fx_fX := self.kernel(x_, self._X_all), self._cov_Y_Y_inv_Y)
        V = _solve_cholesky(self._cov_Y_Y_chol, cov_fx_fX, left=False)
        if joint:
            assert not is_single_point, "Call posterior with joint=False for a single point."
            cov_fx_fx = self.kernel(x_, x_)
            var_ = cov_fx_fx - V.matmul(cov_fx_fX.transpose(-1, -2))
            var_.diagonal(dim1=-2, dim2=-1).clamp_min_(0.0)
        else:
            cov_fx_fx = self.kernel_scale  # kernel(x, x) = kernel_scale
            var_ = cov_fx_fx - torch.linalg.vecdot(cov_fx_fX, V)
            var_.clamp_min_(0.0)
        return (mean.squeeze(0), var_.squeeze(0)) if is_single_point else (mean, var_)

    def marginal_log_likelihood(self) -> torch.Tensor:  # Scalar
        """
        This method computes the marginal log-likelihood of the kernel hyperparameters given the
        training dataset (X, y).
        Assume that N = len(X) in this method.

        Mathematically, the closed form is given as:
            -0.5 * log((2*pi)**N * det(C)) - 0.5 * y.T @ inv(C) @ y
            = -0.5 * log(det(C)) - 0.5 * y.T @ inv(C) @ y + const,
        where C = cov_Y_Y = cov_fX_fX + noise_var * I and inv(...) is the inverse operator.

        We exploit the full advantages of the Cholesky decomposition (C = L @ L.T) in this method:
            1. The determinant of a lower triangular matrix is the diagonal product, which can be
               computed with N flops where log(det(C)) = log(det(L.T @ L)) = 2 * log(det(L)).
            2. Solving linear system L @ u = y, which yields u = inv(L) @ y, costs N**2 flops.
        Note that given `u = inv(L) @ y` and `inv(C) = inv(L @ L.T) = inv(L).T @ inv(L)`,
        y.T @ inv(C) @ y is calculated as (inv(L) @ y) @ (inv(L) @ y).

        In principle, we could invert the matrix C first, but in this case, it costs:
            1. 1/3*N**3 flops for the determinant of inv(C).
            2. 2*N**2-N flops to solve C @ alpha = y, which is alpha = inv(C) @ y.

        Since the Cholesky decomposition costs 1/3*N**3 flops and the matrix inversion costs
        2/3*N**3 flops, the overall cost for the former is 1/3*N**3+N**2+N flops and that for the
        latter is N**3+2*N**2-N flops.
        """
        cov_Y_Y = self.kernel()
        cov_Y_Y.diagonal().add_(self.noise_var)
        L = torch.linalg.cholesky(cov_Y_Y)
        logdet_part = -L.diagonal().log().sum()
        inv_L_y = torch.linalg.solve_triangular(L, self._y_train, upper=False).squeeze(-1)
        quad_part = -0.5 * (inv_L_y @ inv_L_y)
        return logdet_part + quad_part

    def _fit_kernel_params(
        self,
        log_prior: Callable[[GPRegressor], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> GPRegressor:
        n_params = self._X_train.shape[1]

        initial_raw_params = np.concatenate(
            [
                np.log(self.inverse_squared_lengthscales.detach().cpu().numpy()),
                [
                    np.log(self.kernel_scale.item()),
                    np.log(self.noise_var.item() - 0.99 * minimum_noise),
                ],
            ]
        )


        with single_blas_thread_if_scipy_v1_15_or_newer():
            res = scipy.optimize.minimize(
                loss_func,
                initial_raw_params,
                jac=True,
                method="l-bfgs-b",
                options={"gtol": gtol},
            )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        raw_params_opt_tensor = torch.from_numpy(res.x)
        self.inverse_squared_lengthscales = torch.exp(raw_params_opt_tensor[:n_params])
        self.kernel_scale = torch.exp(raw_params_opt_tensor[n_params])
        self.noise_var = (
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        )
        self._cache_matrix()
        return self


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    gpr_cache: GPRegressor | None = None,
    gtol: float = 1e-2,
) -> GPRegressor:
    default_kernel_params = torch.ones(X.shape[1] + 2, dtype=torch.float64)

    def _default_gpr() -> GPRegressor:
        return GPRegressor(
            is_categorical=torch.from_numpy(is_categorical),
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            inverse_squared_lengthscales=default_kernel_params[:-2].clone(),
            kernel_scale=default_kernel_params[-2].clone(),
            noise_var=default_kernel_params[-1].clone(),
        )

    default_gpr_cache = _default_gpr()
    if gpr_cache is None:
        gpr_cache = _default_gpr()

    error = None
    for gpr_cache_to_use in [gpr_cache, default_gpr_cache]:
        try:
            return GPRegressor(
                is_categorical=torch.from_numpy(is_categorical),
                X_train=torch.from_numpy(X),
                y_train=torch.from_numpy(Y),
                inverse_squared_lengthscales=gpr_cache_to_use.inverse_squared_lengthscales,
                kernel_scale=gpr_cache_to_use.kernel_scale,
                noise_var=gpr_cache_to_use.noise_var,
            )._fit_kernel_params(
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel parameters failed: \n{error}\n"
        "The default initial kernel parameters will be used instead."
    )
    default_gpr = _default_gpr()
    default_gpr._cache_matrix()
    return default_gpr
