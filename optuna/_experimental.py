from __future__ import annotations

import functools
import textwrap
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings

from optuna._warnings import optuna_warn
from optuna.exceptions import ExperimentalWarning


if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec

    FT = TypeVar("FT")
    FP = ParamSpec("FP")
    CT = TypeVar("CT")


_EXPERIMENTAL_NOTE_TEMPLATE = """

.. note::
    Added in v{ver} as an experimental feature. The interface may change in newer versions
    without prior notice. See https://github.com/optuna/optuna/releases/tag/v{ver}.
"""


def warn_experimental_argument(option_name: str) -> None:
    optuna_warn(
        f"Argument ``{option_name}`` is an experimental feature."
        " The interface can change in the future.",
        ExperimentalWarning,
    )


def _validate_version(version: str) -> None:
    if not isinstance(version, str) or len(version.split(".")) != 3:
        raise ValueError(
            f"Invalid version specification. Must follow `x.y.z` format but `{version}` is given"
        )


def _get_docstring_indent(docstring: str) -> str:
    return docstring.split("\n")[-1] if "\n" in docstring else ""


def experimental_func(
    version: str,
    name: str | None = None,
) -> Callable[[Callable[FP, FT]], Callable[FP, FT]]:
    """Decorate function as experimental.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to fully qualified name of
        the function, i.e. `f"{func.__module__}.{func.__qualname__}"`. Optional.
    """

    _validate_version(version)


    return decorator


def experimental_class(
    version: str,
    name: str | None = None,
) -> Callable[[CT], CT]:
    """Decorate class as experimental.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to the class name. Optional.
    """

    _validate_version(version)


    return decorator
