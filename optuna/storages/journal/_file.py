from __future__ import annotations

import abc
from collections.abc import Generator
from collections.abc import Iterator
from contextlib import contextmanager
import errno
import json
import os
import time
from typing import Any
import uuid

from optuna._deprecated import deprecated_class
from optuna._warnings import optuna_warn
from optuna.logging import get_logger
from optuna.storages.journal._base import BaseJournalBackend


LOCK_FILE_SUFFIX = ".lock"
RENAME_FILE_SUFFIX = ".rename"

_logger = get_logger(__name__)


class JournalFileBackend(BaseJournalBackend):

    def __init__(self, file_path: str, lock_obj: BaseJournalFileLock | None = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or JournalFileSymlinkLock(self._file_path)
        if not os.path.exists(self._file_path):
            open(self._file_path, "ab").close()  # Create a file if it does not exist.
        self._log_number_offset: dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> Generator[dict[str, Any], None, None]:
        with open(self._file_path, "rb") as f:
            remaining_log_size = os.stat(self._file_path).st_size
            log_number_start = 0
            if log_number_from in self._log_number_offset:
                f.seek(self._log_number_offset[log_number_from])
                log_number_start = log_number_from
                remaining_log_size -= self._log_number_offset[log_number_from]

            last_decode_error = None
            for log_number, line in enumerate(f, start=log_number_start):
                byte_len = len(line)
                remaining_log_size -= byte_len
                if remaining_log_size < 0:
                    break
                if last_decode_error is not None:
                    raise last_decode_error
                if log_number + 1 not in self._log_number_offset:
                    self._log_number_offset[log_number + 1] = (
                        self._log_number_offset[log_number] + byte_len
                    )
                if log_number < log_number_from:
                    continue

                if not line.endswith(b"\n"):
                    last_decode_error = ValueError("Invalid log format.")
                    del self._log_number_offset[log_number + 1]
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as err:
                    last_decode_error = err
                    del self._log_number_offset[log_number + 1]

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        with get_lock_file(self._lock):
            what_to_write = (
                "\n".join([json.dumps(log, separators=(",", ":")) for log in logs]) + "\n"
            )
            with open(self._file_path, "ab") as f:
                f.write(what_to_write.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())


class BaseJournalFileLock(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class JournalFileSymlinkLock(BaseJournalFileLock):

    def __init__(self, filepath: str, grace_period: int | None = 30) -> None:
        self._lock_target_file = filepath
        self._lock_file = filepath + LOCK_FILE_SUFFIX
        if grace_period is not None:
            if grace_period <= 0:
                raise ValueError("The value of `grace_period` should be a positive integer.")
            if grace_period < 3:
                optuna_warn("The value of `grace_period` might be too small. ")
        self.grace_period = grace_period

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a symbolic link of a file.

        Returns:
            :obj:`True` if it succeeded in creating a symbolic link of ``self._lock_target_file``.
        """
        sleep_secs = 0.001
        warning_interval = 10.0
        last_update_monotonic_time = time.monotonic()
        last_warning_time = time.monotonic()
        mtime = None
        while True:
            try:
                os.symlink(self._lock_target_file, self._lock_file)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    if time.monotonic() - last_warning_time > warning_interval:
                        _logger.warning(
                            f"It is taking longer than {warning_interval} seconds to acquire "
                            f"the lock file: {self._lock_file} Retrying..."
                        )
                        last_warning_time = time.monotonic()

                    if self.grace_period is not None:
                        try:
                            current_mtime = os.stat(self._lock_file).st_mtime
                        except OSError:
                            continue
                        if current_mtime != mtime:
                            mtime = current_mtime
                            last_update_monotonic_time = time.monotonic()

                        if time.monotonic() - last_update_monotonic_time > self.grace_period:
                            _logger.warning(
                                "The existing lock file has not been released "
                                "for an extended period. Forcibly releasing the lock file."
                            )
                            last_warning_time = time.monotonic()
                            try:
                                self.release()
                                sleep_secs = 0.001
                            except RuntimeError:
                                continue

                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the symbolic link."""

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


class JournalFileOpenLock(BaseJournalFileLock):

    def __init__(self, filepath: str, grace_period: int | None = 30) -> None:
        self._lock_file = filepath + LOCK_FILE_SUFFIX
        if grace_period is not None:
            if grace_period <= 0:
                raise ValueError("The value of `grace_period` should be a positive integer.")
            if grace_period < 3:
                optuna_warn("The value of `grace_period` might be too small. ")
        self.grace_period = grace_period

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a lock file.

        Returns:
            :obj:`True` if it succeeded in creating a ``self._lock_file``.

        """
        sleep_secs = 0.001
        warning_interval = 10.0
        last_update_monotonic_time = time.monotonic()
        last_warning_time = time.monotonic()
        mtime = None
        while True:
            try:
                open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                os.close(os.open(self._lock_file, open_flags))
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    if time.monotonic() - last_warning_time > warning_interval:
                        _logger.warning(
                            f"It is taking longer than {warning_interval} seconds to acquire "
                            f"the lock file: {self._lock_file} Retrying..."
                        )
                        last_warning_time = time.monotonic()

                    if self.grace_period is not None:
                        try:
                            current_mtime = os.stat(self._lock_file).st_mtime
                        except OSError:
                            continue
                        if current_mtime != mtime:
                            mtime = current_mtime
                            last_update_monotonic_time = time.monotonic()

                        if time.monotonic() - last_update_monotonic_time > self.grace_period:
                            _logger.warning(
                                "The existing lock file has not been released "
                                "for an extended period. Forcibly releasing the lock file."
                            )
                            last_warning_time = time.monotonic()
                            try:
                                self.release()
                                sleep_secs = 0.001
                            except RuntimeError:
                                continue

                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the created file."""

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


@contextmanager
def get_lock_file(lock_obj: BaseJournalFileLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()


@deprecated_class(
    "4.0.0", "6.0.0", text="Use :class:`~optuna.storages.journal.JournalFileBackend` instead."
)
class JournalFileStorage(JournalFileBackend):
    pass


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileOpenLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileOpenLock` instead.",
)
class DeprecatedJournalFileOpenLock(JournalFileOpenLock):
    pass


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileSymlinkLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileSymlinkLock` instead.",
)
class DeprecatedJournalFileSymlinkLock(JournalFileSymlinkLock):
    pass
