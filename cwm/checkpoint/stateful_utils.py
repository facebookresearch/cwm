# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Model wrapper for distributed checkpoints.
Derived from https://github.com/pytorch/torchtitan
"""

import logging
from functools import partial
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger()


class DCPModelWrapper(Stateful):
    def __init__(self, model: torch.nn.Module | list[torch.nn.Module]) -> None:
        self.model = [model] if isinstance(model, torch.nn.Module) else model

    def state_dict(self) -> dict[str, Any]:
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
