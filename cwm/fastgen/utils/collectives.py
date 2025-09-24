# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

has_custom_allreduce = False
with suppress(ModuleNotFoundError, ImportError):
    from fgcuda.custom_all_reduce import CustomAllreduce

    has_custom_allreduce = True


class Collectives:
    """
    Collectives and their associated state.
    If possible, dispatch to fast intra-node
    implementations.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        max_custom_allreduce_size: int = 8192 * 1024,
    ) -> None:
        self.group = mesh.get_group()
        self.rank = mesh.get_local_rank()
        self.car: Any = None

        use_car = has_custom_allreduce
        if mesh.size() > 8:
            # Crude approximation for devices being
            # on the same host; works for us
            use_car = False

        if use_car:
            self.car = CustomAllreduce(  # type: ignore
                group=self.group,
                device=torch.cuda.current_device(),
                max_size=max_custom_allreduce_size,
            )

    @property
    def tp_size(self) -> int:
        return self.group.size()

    def local_split(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split ``x`` along dimension 1 over devices in
        the collectives group.
        """
        size = self.group.size()
        dim_size = x.size(1)
        assert dim_size % size == 0, (dim_size, size)
        start_idx = self.rank * (dim_size // size)
        end_idx = start_idx + dim_size // size
        return x[:, start_idx:end_idx].contiguous()

    def all_gather(
        self,
        x: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        size = self.group.size()
        gather = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(gather, x, group=self.group)
        return torch.cat(gather, dim=dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.car is not None:
            out = self.car.custom_all_reduce(x)
            if out is not None:
                return out
            # else, fall back to nccl
        dist.all_reduce(x, group=self.group)
        return x

    @contextmanager
    def capture(self) -> Iterator[None]:
        if self.car is not None:
            with self.car.capture():
                yield
        else:
            yield

    def close(self) -> None:
        if self.car is not None:
            self.car.close()
            self.car = None
