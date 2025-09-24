# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import time
from typing import Literal, TextIO

from cwm.common.environment import get_global_rank

METRICS: TextIO | Literal["disabled"] | None = None


def dump(kind: str, **kwargs) -> None:
    global METRICS

    if METRICS == "disabled":
        return

    if METRICS is None:
        if path := os.environ.get("FG_METRICS"):
            METRICS = open(path, "a", encoding="utf8")
        else:
            METRICS = "disabled"
        return dump(kind, **kwargs)

    record = {
        "kind": kind,
        "rank": get_global_rank(),
        "timestamp": time.time(),
        **kwargs,
    }
    record_str = json.dumps(record) + "\n"
    METRICS.write(record_str)
    METRICS.flush()


def recording() -> bool:
    return METRICS is not None
