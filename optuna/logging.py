from __future__ import annotations

import logging
from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
from logging import WARNING
import sys
import threading

import colorlog


__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "WARN",
    "WARNING",
]

_lock: threading.Lock = threading.Lock()
_default_handler: logging.Handler | None = None


def create_default_formatter() -> logging.Formatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """
    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    return colorlog.TTYColoredFormatter(
        f"%(log_color)s{header}%(reset)s {message}",
        stream=sys.stderr,
    )


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.setFormatter(create_default_formatter())

        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False




def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name.

    This function is not supposed to be directly accessed by library users.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    pass


def set_verbosity(verbosity: int) -> None:
    pass


def disable_default_handler() -> None:
    """Disable the default handler of the Optuna's root logger.

    Example:

        Stop and then resume logging to :obj:`sys.stderr`.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x**2 + y

        .. testcode::

            import optuna

            study = optuna.create_study()

            # There are no logs in sys.stderr.
            optuna.logging.disable_default_handler()
            study.optimize(objective, n_trials=10)

            # There are logs in sys.stderr.
            optuna.logging.enable_default_handler()
            study.optimize(objective, n_trials=10)
            # [I 2020-02-23 17:00:54,314] Trial 10 finished with value: ...
            # [I 2020-02-23 17:00:54,356] Trial 11 finished with value: ...
            # ...

    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the Optuna's root logger.

    Please refer to the example shown in :func:`~optuna.logging.disable_default_handler()`.
    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    pass


def enable_propagation() -> None:
    pass
