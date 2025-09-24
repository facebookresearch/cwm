# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any

from cwm.data.dataset import Dataset


@Dataset.register_src("range")
class Range(Dataset[int]):
    def __init__(self, begin: int, end: int) -> None:
        self.begin = begin
        self.end = end
        self.reset()

    def reset(self) -> None:
        self.curr = self.begin

    def __next__(self) -> int:
        if self.curr >= self.end:
            raise StopIteration

        curr, self.curr = self.curr, self.curr + 1
        return curr

    def state_dict(self) -> dict[str, Any]:
        return {"curr": self.curr}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.curr = state_dict["curr"]
