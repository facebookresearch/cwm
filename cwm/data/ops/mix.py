# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Protocol

from cwm.data.dataset import Dataset, T


class MixPolicy(Protocol):
    def initialize(self, datasets: list[Dataset[T]]) -> None: ...

    def reset(self) -> None: ...

    def select_dataset(self) -> int | None: ...

    def exhaust(self, dataset_idx: int) -> bool: ...


@Dataset.register_op("mix")
class Mix(Dataset[T]):
    def __init__(
        self,
        datasets: list[Dataset[T]],
        policy: MixPolicy,
    ) -> None:
        self.datasets = datasets
        self.policy = policy
        self.policy.initialize(datasets)

    def reset(self) -> None:
        for dataset in self.datasets:
            dataset.reset()
        self.policy.reset()

    def __next__(self) -> T:
        while True:
            idx = self.policy.select_dataset()
            if idx is None:
                raise StopIteration

            try:
                return next(self.datasets[idx])
            except StopIteration:
                if self.policy.exhaust(idx):
                    self.datasets[idx].reset()

    def state_dict(self) -> dict[str, Any]:
        return {
            "datasets": [dataset.state_dict() for dataset in self.datasets],
            "policy": self.policy.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for dataset, state in zip(self.datasets, state_dict["datasets"], strict=True):
            dataset.load_state_dict(state)
        self.policy.load_state_dict(state_dict["policy"])
