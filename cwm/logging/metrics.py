# Copyright (c) Meta Platforms, Inc. and affiliates.

import fcntl
import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import torch
from typing_extensions import Self  # noqa

from cwm.utils.distributed import rank_zero_only

logger = logging.getLogger()


def _sanitize(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.detach().item() if isinstance(v, torch.Tensor) else v
        for k, v in metrics.items()
    }


class MetricsLogger(ABC):
    """Base interface for logging experiment metrics."""

    def __init__(
        self,
        *,
        tag: str | None = None,
        enable_media_logging: bool = False,
    ) -> None:
        self._tag = tag
        self._enable_media_logging = enable_media_logging

    @property
    def tag(self) -> str | None:
        return self._tag

    @property
    def media_logging_enabled(self) -> bool:
        return self._enable_media_logging

    @rank_zero_only
    @abstractmethod
    def open(self) -> None: ...

    @rank_zero_only
    @abstractmethod
    def close(self) -> None: ...

    @rank_zero_only
    @abstractmethod
    def is_disabled(self) -> bool: ...

    @rank_zero_only
    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        *,
        sanitize: bool = True,
    ) -> None: ...

    @rank_zero_only
    @abstractmethod
    def log_text(
        self,
        key: str,
        text: str,
        step: int | None = None,
        *,
        sanitize: bool = True,
    ) -> None: ...

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class JsonlMetricsLogger(MetricsLogger):
    """Metrics logging to JSONL file of name 'metrics.<tag>.jsonl'."""

    def __init__(
        self,
        save_dir: Path,
        *,
        tag: str | None = None,
        enable_media_logging: bool = False,
    ) -> None:
        super().__init__(tag=tag, enable_media_logging=enable_media_logging)
        self.save_dir = save_dir
        self.fname = f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"
        self.fpath = save_dir / self.fname
        self.jsonl_writer: TextIO | None = None

    @rank_zero_only
    def is_disabled(self) -> bool:
        return self.writer is None

    @rank_zero_only
    def open(self) -> None:
        if self.jsonl_writer is None:
            self.jsonl_writer = self.fpath.open("a")

    @rank_zero_only
    def close(self) -> None:
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    @rank_zero_only
    def log_metrics(
        self, metrics: dict[str, Any], step: int | None = None, *, sanitize: bool = True
    ) -> None:
        # check / sanitize inputs
        assert all(type(k) is str for k in metrics)

        if sanitize:
            metrics = _sanitize(metrics)

        json_metrics = {
            "global_step": step,
            "created_at": datetime.now(UTC).isoformat(),
        }
        json_metrics.update(metrics)
        if self.jsonl_writer is not None:
            fcntl.flock(self.jsonl_writer, fcntl.LOCK_EX)
        try:
            print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)
        finally:
            if self.jsonl_writer is not None:
                fcntl.flock(self.jsonl_writer, fcntl.LOCK_UN)

    @rank_zero_only
    def log_text(
        self, key: str, text: str, step: int | None = None, *, sanitize: bool = True
    ) -> None:
        self.log_metrics(metrics={key: text}, step=step, sanitize=sanitize)
