# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any

from cwm.data.dataset import Dataset, T


class ChainPolicy:
    def __init__(self) -> None:
        self.idx: int | None = None
        self.cnt: int | None = None

        self.reset()

    def initialize(self, datasets: list[Dataset[T]]) -> None:
        self.cnt = len(datasets)

    def reset(self) -> None:
        self.idx = 0

    def select_dataset(self) -> int | None:
        assert self.cnt is not None
        if self.idx is not None and self.idx >= self.cnt:
            self.idx = None
        return self.idx

    def exhaust(self, _: int) -> bool:
        assert self.idx is not None
        assert self.cnt is not None
        self.idx += 1
        if self.idx >= self.cnt:
            self.idx = None
        return self.idx is not None

    def state_dict(self) -> dict[str, Any]:
        return {"idx": self.idx, "cnt": self.cnt}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.idx = state_dict["idx"]
        self.cnt = state_dict["cnt"]


@Dataset.register_op("chain")
def chain(datasets: list[Dataset[T]]) -> Dataset[T]:
    return Dataset.mix(datasets, ChainPolicy())
