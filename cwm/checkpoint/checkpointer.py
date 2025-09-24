# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.metadata import Metadata
from upath import UPath

from cwm.checkpoint.stateful_utils import DCPModelWrapper
from cwm.logging.logger import log_timing

logger = logging.getLogger()


class Checkpointer:
    """Simple checkpointer to save/load the model."""

    def __init__(self, model: torch.nn.Module | list[torch.nn.Module]) -> None:
        self.states = {"model": DCPModelWrapper(model)}

    def load_from_path(self, path: UPath, step: int = -1) -> None:
        with log_timing(f"Load checkpoint from {path}"):
            path = UPath(path)  # linting does not catch if type(path) != UPath
            storage_reader = FsspecReader(path, **path.storage_options)
            dcp.load(self.states, storage_reader=storage_reader)

    def save_to_path(self, path: UPath) -> Metadata:
        logger.info(f"Saving checkpoint to: {path}")
        path = UPath(path)  # linting does not catch if type(path) != UPath
        storage_writer = FsspecWriter(
            path, sync_files=path.protocol == "", **path.storage_options
        )
        begin_time = time.monotonic()
        meta = dcp.save(self.states, storage_writer=storage_writer)
        logger.info(
            f"Checkpoint {meta.storage_meta.checkpoint_id} saved in {time.monotonic() - begin_time} seconds."
        )
        return meta
