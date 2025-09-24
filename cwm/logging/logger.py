# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import math
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import timedelta
from pathlib import Path
from typing import TypeVar

from cwm.common.environment import get_global_rank, get_is_slurm_job

T = TypeVar("T")


class LogFormatter(logging.Formatter):
    """Custom logger for distributed jobs, displaying rank
    and preserving indent from the custom prefix format.
    """

    def __init__(self, display_name: bool = False) -> None:
        self.start_time = time.time()
        self.rank = get_global_rank()
        self.display_rank = not get_is_slurm_job()  # srun has --label
        self.display_name = display_name  # useful for finer debugging

    def format_time(self, record: logging.LogRecord) -> str:
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))
        return f"{curr_date} - {delta}"

    def format_prefix(self, record: logging.LogRecord) -> str:
        fmt_time = self.format_time(record)
        prefix = ""
        if self.display_rank:
            prefix += f"{self.rank}: "
        prefix += f"{record.levelname:<7} {fmt_time} - "
        if self.display_name:
            prefix += f"{record.name} - "
        return prefix

    def format_message_with_indent(self, record: logging.LogRecord, indent: str) -> str:
        content = record.getMessage()
        content = content.replace("\n", "\n" + indent)
        # Exception handling as in the default formatter, albeit with indenting
        # according to our custom prefix

        # Cache the traceback text to avoid converting it multiple times
        # (it's constant anyway)
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join(
                [line + "\n" for line in record.exc_text.splitlines()],
            )
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join(
                [line + "\n" for line in stack_text.splitlines()],
            )
            if content[-1:] == "\n":
                content = content[:-1]

        return content

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.format_prefix(record)
        indent = " " * len(prefix)
        content = self.format_message_with_indent(record, indent)
        return prefix + content


def set_root_log_level(log_level: str) -> None:
    logger = logging.getLogger()
    level: int | str = log_level.upper()
    with suppress(ValueError):
        level = int(log_level)
    try:
        logger.setLevel(level)
    except (TypeError, ValueError):
        logger.warning(
            f"Failed to set logging level to {log_level}, using default 'NOTSET'",
        )
        logger.setLevel(logging.NOTSET)


def initialize_logger(name: str | None = None, level: str = "NOTSET") -> None:
    """Setup logging.

    Args:
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
    """
    set_root_log_level(level)
    logger = logging.getLogger()
    log_formatter = LogFormatter(display_name=False)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(log_formatter)

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(log_formatter)

    # set stream handlers
    logger.handlers.clear()
    assert len(logger.handlers) == 0, logger.handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)


def add_logger_file_handler(log_file: str | Path) -> None:
    logger = logging.getLogger()
    if get_global_rank() == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # build file handler
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(LogFormatter())
        # update logger
        logger.handlers.append(file_handler)


@contextmanager
def log_timing(
    message: str,
    level: int = logging.INFO,
    clock: Callable[[], float] = time.monotonic,
    precision: int = 3,
) -> Iterator[None]:
    logger = logging.getLogger()
    logger.log(level, f"{message}: starting")
    t0 = clock()
    try:
        yield
    finally:
        t1 = clock()
        logger.log(level, f"{message}: completed in {t1 - t0:.{precision}f}s")
