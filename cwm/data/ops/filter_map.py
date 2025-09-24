# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Callable
from typing import Any

import dill

from cwm.data.dataset import Dataset, T, T_in, T_out


class FilterMap(Dataset[T_out]):
    def __init__(
        self, dataset: Dataset[T_in], fn: Callable[[T_in], tuple[bool, T_out]]
    ) -> None:
        self.dataset = dataset
        self.fn = fn

    def reset(self) -> None:
        self.dataset.reset()

    def __next__(self) -> T_out:
        # Assuming that the function `fn` is stateless, the
        # position of a FilterMap operator is fully encoded
        # in the position of `dataset`.

        while True:
            keep, b = self.fn(next(self.dataset))
            if keep:
                return b

    def state_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.dataset.load_state_dict(state_dict["dataset"])

    def __getstate__(self) -> str:
        return dill.dumps((self.dataset, self.fn))

    def __setstate__(self, state: str) -> None:
        self.dataset, self.fn = dill.loads(state)


@Dataset.register_op("filter")
def filter(dataset: Dataset[T], fn: Callable[[T], bool]) -> Dataset[T]:
    return FilterMap(dataset, lambda x: (fn(x), x))


@Dataset.register_op("map")
def map(dataset: Dataset[T_in], fn: Callable[[T_in], T_out]) -> Dataset[T_out]:
    return FilterMap(dataset, lambda x: (True, fn(x)))
