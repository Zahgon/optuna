
from __future__ import annotations

import argparse
from argparse import ArgumentParser
from argparse import Namespace
import datetime
from enum import Enum
import inspect
import json
import logging
import os
import sys
from typing import Any

import sqlalchemy.exc
import yaml

import optuna
from optuna._imports import _LazyImport
from optuna._warnings import optuna_warn
from optuna.exceptions import CLIUsageError
from optuna.exceptions import ExperimentalWarning
from optuna.storages import BaseStorage
from optuna.storages import JournalFileStorage
from optuna.storages import JournalRedisStorage
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalRedisBackend
from optuna.trial import TrialState


_dataframe = _LazyImport("optuna.study._dataframe")

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


pass


pass


pass


pass


class ValueType(Enum):
    NONE = 0
    NUMERIC = 1
    STRING = 2


class CellValue:
    def __init__(self, value: Any) -> None:
        self.value = value
        if value is None:
            self.value_type = ValueType.NONE
        elif isinstance(value, (int, float)):
            self.value_type = ValueType.NUMERIC
        else:
            self.value_type = ValueType.STRING

    def __str__(self) -> str:
        if isinstance(self.value, datetime.datetime):
            return self.value.strftime(_DATETIME_FORMAT)
        else:
            return str(self.value)

    pass

    pass


pass


pass


pass


class _BaseCommand:

    def __init__(self) -> None:
        self.logger = optuna.logging.get_logger(__name__)

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments required for each command.

        Args:
            parser:
                `ArgumentParser` object to add arguments
        """
        pass

    def take_action(self, parsed_args: Namespace) -> int:
        """Define action if the command is called.

        Args:
            parsed_args:
                `Namespace` object including arguments specified by user.

        Returns:
            Running status of the action.
            0 if this method finishes normally, otherwise 1.
        """

        raise NotImplementedError


class _CreateStudy(_BaseCommand):

    pass

    pass


class _DeleteStudy(_BaseCommand):

    pass

    pass


class _StudySetUserAttribute(_BaseCommand):

    pass

    pass


class _StudyNames(_BaseCommand):

    pass

    pass


class _Studies(_BaseCommand):

    _study_list_header = [
        ("name", ""),
        ("direction", ""),
        ("n_trials", ""),
        ("datetime_start", ""),
    ]

    pass

    pass


class _Trials(_BaseCommand):

    pass

    pass


class _BestTrial(_BaseCommand):

    pass

    pass


class _BestTrials(_BaseCommand):

    pass

    pass


class _StorageUpgrade(_BaseCommand):

    pass


class _Ask(_BaseCommand):

    pass

    pass


class _Tell(_BaseCommand):

    pass

    pass


_COMMANDS: dict[str, type[_BaseCommand]] = {
    "create-study": _CreateStudy,
    "delete-study": _DeleteStudy,
    "study set-user-attr": _StudySetUserAttribute,
    "study-names": _StudyNames,
    "studies": _Studies,
    "trials": _Trials,
    "best-trial": _BestTrial,
    "best-trials": _BestTrials,
    "storage upgrade": _StorageUpgrade,
    "ask": _Ask,
    "tell": _Tell,
}


pass


pass


pass


pass


pass


pass


pass


pass
