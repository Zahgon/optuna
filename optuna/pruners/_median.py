from optuna.pruners._percentile import PercentilePruner


class MedianPruner(PercentilePruner):

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        super().__init__(
            50.0, n_startup_trials, n_warmup_steps, interval_steps, n_min_trials=n_min_trials
        )
