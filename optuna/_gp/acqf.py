from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna._hypervolume import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import torch

    from optuna._gp.gp import GPRegressor
    from optuna._gp.search_space import SearchSpace
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
_EPS = 1e-12  # NOTE(nabenabe): grad becomes nan when EPS=0.


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
    sobol_samples = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=dim, scramble=True, seed=seed
    ).draw(n_samples, dtype=torch.float64)
    samples = 2.0 * (sobol_samples - 0.5)  # The Sobol sequence in [-1, 1].
    return torch.erfinv(samples) * float(np.sqrt(2))


def logehvi(
    Y_post: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
    non_dominated_box_lower_bounds: torch.Tensor,  # (n_boxes, n_objectives)
    non_dominated_box_intervals: torch.Tensor,  # (n_boxes, n_objectives)
) -> torch.Tensor:  # (..., )
    log_n_qmc_samples = float(np.log(Y_post.shape[-2]))
    diff = Y_post.unsqueeze(-2) - non_dominated_box_lower_bounds
    diff.clamp_(min=torch.tensor(_EPS, dtype=torch.float64), max=non_dominated_box_intervals)
    return torch.special.logsumexp(diff.log().sum(dim=-1), dim=(-2, -1)) - log_n_qmc_samples


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    """
    Return E_{x ~ N(0, 1)}[max(0, x+z)]
    The calculation depends on the value of z for numerical stability.
    Please refer to Eq. (9) in the following paper for more details:
        https://arxiv.org/pdf/2310.20708.pdf

    NOTE: We do not use the third condition because [-10**100, 10**100] is an overly high range.
    """
    out = (
        (z_half := 0.5 * z) * torch.special.erfc(-_SQRT_HALF * z)  # z * cdf(z)
        + (-z_half * z).exp() * _INV_SQRT_2PI  # pdf(z)
    ).log()
    if (z_small := z[(small := z < -25)]).numel():
        out[small] = (
            -0.5 * z_small**2
            - _LOG_SQRT_2PI
            + (1 + _SQRT_HALF_PI * z_small * torch.special.erfcx(-_SQRT_HALF * z_small)).log()
        )
    return out


def logei(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    return standard_logei((mean - f0) / (sigma := var.sqrt_())) + sigma.log()


class BaseAcquisitionFunc(ABC):
    def __init__(self, length_scales: np.ndarray, search_space: SearchSpace) -> None:
        self.length_scales = length_scales
        self.search_space = search_space

    @abstractmethod
    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.eval_acqf(torch.from_numpy(x)).detach().numpy()



class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold

        if normalized_params_of_running_trials is not None:
            normalized_params_of_running_trials_tensor = torch.from_numpy(
                normalized_params_of_running_trials
            )

            constant_liar_value = self._gpr._y_train.max()
            constant_liar_y = constant_liar_value.expand(
                normalized_params_of_running_trials_tensor.shape[0]
            )

            self._gpr.append_running_data(
                normalized_params_of_running_trials_tensor,
                constant_liar_y,
            )

        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return (
            logei(mean=mean, var=var + self._stabilizing_noise, f0=self._threshold)
            if not np.isneginf(self._threshold)
            else torch.zeros(x.shape[:-1], dtype=torch.float64)
        )


class qLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        n_qmc_samples: int,
        qmc_seed: int | None,
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold
        self._X_running = torch.from_numpy(normalized_params_of_running_trials)
        self._fixed_samples = _sample_from_normal_sobol(
            dim=1 + normalized_params_of_running_trials.shape[0],
            n_samples=n_qmc_samples,
            seed=qmc_seed,
        )
        super().__init__(gpr.length_scales, search_space)

    def _get_posterior_samples(self, x: torch.Tensor) -> torch.Tensor:
        mean, cov = self._gpr.posterior(x, joint=True)
        cov.diagonal(dim1=-2, dim2=-1).add_(self._stabilizing_noise)
        return mean.unsqueeze(-2) + torch.matmul(
            self._fixed_samples, torch.linalg.cholesky(cov).transpose(-1, -2)
        )

    def _get_joint_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return torch.cat([self._X_running, x.unsqueeze(0)], dim=0)
        if x.ndim == 2:
            running = self._X_running.unsqueeze(0).expand(x.shape[0], -1, -1)
            return torch.cat([running, x.unsqueeze(-2)], dim=-2)
        raise ValueError(f"{x.ndim=} must be 1 or 2.")

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        if np.isneginf(self._threshold):
            return torch.zeros(x.shape[:-1], dtype=torch.float64)

        joint_x = self._get_joint_input(x)
        y_post = self._get_posterior_samples(joint_x)
        log_improvement = y_post.clamp_(min=torch.tensor(_EPS, dtype=torch.float64)).log()
        max_log_improvement_in_q_batch = torch.amax(log_improvement, dim=-1)
        return torch.special.logsumexp(max_log_improvement_in_q_batch, dim=-1) - math.log(
            max_log_improvement_in_q_batch.shape[-1]
        )


class LogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold

        if normalized_params_of_running_trials is not None:
            normalized_params_of_running_trials_tensor = torch.from_numpy(
                normalized_params_of_running_trials
            )


            self._gpr.append_running_data(
                normalized_params_of_running_trials_tensor,
                gpr.posterior(normalized_params_of_running_trials_tensor)[0],
            )
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        sigma = torch.sqrt(var + self._stabilizing_noise)
        return torch.special.log_ndtr((mean - self._threshold) / sigma)


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean + torch.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean - torch.sqrt(self._beta * var)


class ConstrainedLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = LogEI(
            gpr, search_space, threshold, normalized_params_of_running_trials, stabilizing_noise
        )
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return self._acqf.eval_acqf(x) + sum(
            acqf.eval_acqf(x) for acqf in self._constraints_acqf_list
        )


class LogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_train: torch.Tensor,
        n_qmc_samples: int,
        qmc_seed: int | None,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        def _get_non_dominated_box_bounds() -> tuple[torch.Tensor, torch.Tensor]:
            loss_vals = -Y_train.numpy()
            pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]
            ref_point = np.max(loss_vals, axis=0)
            ref_point = np.nextafter(np.maximum(1.1 * ref_point, 0.9 * ref_point), np.inf)
            lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
            return torch.from_numpy(-ubs), torch.from_numpy(-lbs)

        self._stabilizing_noise = stabilizing_noise
        self._gpr_list = gpr_list
        if normalized_params_of_running_trials is not None:
            normalized_params_of_running_trials_tensor = torch.from_numpy(
                normalized_params_of_running_trials
            )


            for gpr in self._gpr_list:
                gpr.append_running_data(
                    normalized_params_of_running_trials_tensor,
                    gpr.posterior(normalized_params_of_running_trials_tensor)[0],
                )

        self._fixed_samples = _sample_from_normal_sobol(
            dim=Y_train.shape[-1], n_samples=n_qmc_samples, seed=qmc_seed
        )
        self._non_dominated_box_lower_bounds, non_dominated_box_upper_bounds = (
            _get_non_dominated_box_bounds()
        )
        self._non_dominated_box_intervals = (
            non_dominated_box_upper_bounds - self._non_dominated_box_lower_bounds
        ).clamp_min_(_EPS)
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        Y_post = []
        for i, gpr in enumerate(self._gpr_list):
            mean, var = gpr.posterior(x)
            stdev = torch.sqrt(var + self._stabilizing_noise)
            Y_post.append(mean[..., None] + stdev[..., None] * self._fixed_samples[..., i])

        return logehvi(
            Y_post=torch.stack(Y_post, dim=-1),
            non_dominated_box_lower_bounds=self._non_dominated_box_lower_bounds,
            non_dominated_box_intervals=self._non_dominated_box_intervals,
        )


class ConstrainedLogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_feasible: torch.Tensor | None,
        n_qmc_samples: int,
        qmc_seed: int | None,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = (
            LogEHVI(
                gpr_list,
                search_space,
                Y_feasible,
                n_qmc_samples,
                qmc_seed,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            if Y_feasible is not None
            else None
        )
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        constraints_acqf_values = sum(acqf.eval_acqf(x) for acqf in self._constraints_acqf_list)
        if self._acqf is None:
            return cast("torch.Tensor", constraints_acqf_values)
        return constraints_acqf_values + self._acqf.eval_acqf(x)
