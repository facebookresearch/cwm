# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from functools import wraps
from types import ModuleType
from typing import TypeVar

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class Dataset(Stateful, IterableDataset[T], ABC):
    @abstractmethod
    def reset(self) -> None: ...

    def __iter__(self) -> Iterator[T]:
        # Important: Do not override this method!
        # Implement the __next__() method instead.
        return self

    @abstractmethod
    def __next__(self) -> T: ...

    @staticmethod
    def register_op(name: str) -> Callable:
        if hasattr(Dataset, name):
            raise KeyError(f"Dataset operator '{name}' already defined.")

        def decorator(
            cls: type["Dataset"] | Callable[..., "Dataset"],
        ) -> type["Dataset"] | Callable[..., "Dataset"]:
            @wraps(cls)
            def wrapper(self: "Dataset", *args: list, **kwargs: dict) -> "Dataset":
                return cls(self, *args, **kwargs)  # type: ignore[call-arg]

            wrapper.__signature__ = inspect.signature(wrapper).replace(
                return_annotation=Dataset
            )

            setattr(Dataset, name, wrapper)
            return cls

        return decorator

    @staticmethod
    def register_src(
        name: str,
    ) -> Callable[
        [type["Dataset"] | Callable[..., "Dataset"]],
        type["Dataset"] | Callable[..., "Dataset"],
    ]:
        return Dataset.register_op(f"from_{name}")

    @staticmethod
    def register_package(package: ModuleType, subpackage_names: list[str]) -> None:
        for subpackage_name in subpackage_names:
            full_subpackage_name = f"{package.__name__}.{subpackage_name}"
            subpackage = importlib.import_module(full_subpackage_name)
            for _, module_name, _ in pkgutil.iter_modules(subpackage.__path__):
                importlib.import_module(f"{full_subpackage_name}.{module_name}")


import cwm.data  # noqa: E402

Dataset.register_package(cwm.data, ["ops", "srcs"])
