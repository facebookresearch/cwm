# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from enum import Enum


class ExecStatus(Enum):
    UNKNOWN = -1
    SUCCESS = 0
    FAILURE = 1
    EXCEPTION = 2
    SYNTAX_ERROR = 3
    TIMEOUT = 4
    OUT_OF_MEMORY = 5
    INFRA_FAILURE = 6


@dataclass
class ExecResult:
    status: ExecStatus = ExecStatus.UNKNOWN
    info: str = ""
    duration: float = -1  # Execution time in seconds
