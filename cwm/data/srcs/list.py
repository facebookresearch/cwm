# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any

from cwm.data.dataset import Dataset, T


@Dataset.register_src("list")
class List(Dataset[T]):
    def __init__(self, values: list[T]) -> None:
        self.values = values
        self.reset()

    def reset(self) -> None:
        self.idx = 0

    def __next__(self) -> T:
        if self.idx >= len(self.values):
            raise StopIteration

        idx, self.idx = self.idx, self.idx + 1
        return self.values[idx]

    def state_dict(self) -> dict[str, Any]:
        return {"idx": self.idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.idx = state_dict["idx"]
