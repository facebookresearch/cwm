# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh

from cwm.common.environment import get_is_rank_zero

logger = logging.getLogger()


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs) -> Any | None:
        if get_is_rank_zero():
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def dist_max(x: float | np.floating, mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x, device="cuda")
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh).item()


def dist_min(x: float | np.floating, mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x, device="cuda")
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MIN.name, group=mesh).item()


def dist_mean(x: float | np.floating, mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x, device="cuda")
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.AVG.name, group=mesh).item()


def dist_max_list(x: list[float | np.floating], mesh: DeviceMesh) -> list[float]:
    tensor = [torch.tensor(xi, device="cuda", dtype=torch.float64) for xi in x]
    reduced = funcol.all_reduce_coalesced(
        tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh
    )
    return [r.item() for r in reduced]


def dist_min_list(x: list[float | np.floating], mesh: DeviceMesh) -> list[float]:
    tensor = [torch.tensor(xi, device="cuda", dtype=torch.float64) for xi in x]
    reduced = funcol.all_reduce_coalesced(
        tensor, reduceOp=c10d.ReduceOp.MIN.name, group=mesh
    )
    return [r.item() for r in reduced]


def dist_sum_list(x: list[float | np.floating], mesh: DeviceMesh) -> list[float]:
    tensor = [torch.tensor(xi, device="cuda", dtype=torch.float64) for xi in x]
    reduced = funcol.all_reduce_coalesced(
        tensor, reduceOp=c10d.ReduceOp.SUM.name, group=mesh
    )
    return [r.item() for r in reduced]


def all_gather_object(object: Any, group: dist.ProcessGroup) -> list:
    size = dist.get_world_size(group)
    obj_list = [None] * size
    dist.all_gather_object(obj_list, object, group=group)
    return obj_list


def div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def _max(xs: list[float]) -> float:
    return max(xs) if xs else float("-inf")


class Metrics:
    def __init__(self):
        self.sum = defaultdict(float)
        self.max = defaultdict(lambda: float("-inf"))
        self.count = defaultdict(int)

    @property
    def keys(self) -> set[str]:
        return set(self.sum) | set(self.max) | set(self.count)

    def add(self, metrics: dict[str, list[float]]) -> None:
        "Update with a list of values per metric key."
        sums = {k: sum(v) for k, v in metrics.items()}
        maxs = {k: _max(v) for k, v in metrics.items()}
        counts = {k: len(v) for k, v in metrics.items()}
        self.add_aggregates(sums, counts, maxs)

    def add_single(self, metrics: dict[str, float]) -> None:
        "Update with a single value per metric key."
        self.add_aggregates(metrics)

    def add_aggregates(
        self,
        sums: dict[str, float],
        counts: dict[str, int] | None = None,
        maxs: dict[str, float] | None = None,
    ) -> None:
        """
        Update sums, maxima and counts for each metric key.

        If `counts` is None, use 1 for every key that is updated.
        If `maxs` is None, use the value from `sums`.
        """
        counts = counts or {}
        maxs = maxs or sums
        for k in set(sums) | set(counts) | set(maxs):
            self.sum[k] += float(sums.get(k, 0))
            self.max[k] = max(self.max[k], maxs.get(k, float("-inf")))
            self.count[k] += counts.get(k, 1)

    def dist_sum(self, mesh: DeviceMesh) -> dict[str, float]:
        "Calculate averages over a 1D mesh."
        group = mesh.get_group()
        rank_keys = all_gather_object(self.keys, group)
        keys = sorted(list(set.union(*rank_keys)))
        sums = {}
        for k in keys:
            sm, _ = dist_sum_list([self.sum[k], self.count[k]], mesh)
            sums[k] = sm
        return sums

    def dist_mean(self, mesh: DeviceMesh) -> dict[str, float]:
        "Calculate averages over a 1D mesh."
        group = mesh.get_group()
        rank_keys = all_gather_object(self.keys, group)
        keys = sorted(list(set.union(*rank_keys)))
        avgs = {}
        for k in keys:
            sm, cnt = dist_sum_list([self.sum[k], self.count[k]], mesh)
            avgs[k] = div(sm, cnt)
        return avgs

    def dist_max(self, mesh: DeviceMesh) -> dict[str, float]:
        "Calculate maxima over a 1D mesh."
        group = mesh.get_group()  # only support 1D meshes
        rank_keys = all_gather_object(self.keys, group)
        keys = sorted(list(set.union(*rank_keys)))
        maxs = {}
        for k in keys:
            maxs[k] = dist_max(self.max[k], mesh)
        return maxs

    def state_dict(self) -> dict[str, Any]:
        return {
            "sum": dict(self.sum),
            "max": dict(self.max),
            "count": dict(self.count),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sum.update(state_dict["sum"])
        self.max.update(state_dict["max"])
        self.count.update(state_dict["count"])
