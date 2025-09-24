# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Model cache implementations."""

import heapq
import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

from . import metrics
from .misc import hashints

if TYPE_CHECKING:
    from cwm.model.transformer import Transformer as CWMTransformerModel

    from .model import Transformer as FastgenTransformerModel

    TransformerModel: TypeAlias = CWMTransformerModel | FastgenTransformerModel


logger = logging.getLogger()


class ModelCache:
    """Cache stub interface."""

    def cache_kv(self, n: int) -> tuple[torch.Tensor, ...]:
        """
        The key-value cache for layer n.
        """
        raise NotImplementedError

    def page_in(self, n: int) -> None:
        """
        Page in (host to device) the cache for layer n.
        """

    def page_out(self, n: int) -> None:
        """
        Page out (device to host) the cache for layer n.
        """


@dataclass
class RawCache(ModelCache):
    """Inference key-value caches"""

    caches: list[torch.Tensor]
    split: list[int]
    length: int
    start: int = 0

    @staticmethod
    def build(
        model: "TransformerModel",
        length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RawCache":
        """
        Allocate a cache to be used with the decoding functions.

        Args:
            model (Model): the model to build a cache for.
            length (int): per layer cache size.
                It is usually budgeted as ``max_batch * seq_len``.
            device (torch.device, optional): the device on which
                the cache should be allocated (defaults to the
                default device).
            dtype (torch.dtype, optional): the dtype to use for
                cache entries (defaults to the default dtype).
        """

        if model.kv_lora_rank > 0:
            lora_dim = model.kv_lora_rank
            rope_dim = model.qk_rope_head_dim
            cache_dim = lora_dim + rope_dim
            shape = (length, cache_dim)
            split = [cache_dim]
        else:
            n_heads = model.layers[0].attention.n_kv_heads
            k_dim = model.qk_head_dim
            v_dim = model.head_dim
            shape = (length, n_heads * (k_dim + v_dim))
            split = [n_heads * k_dim, n_heads * v_dim]
        return RawCache(
            [
                torch.zeros(shape, device=device, dtype=dtype)
                for _ in range(model.n_layers)
            ],
            split,
            length,
        )

    @staticmethod
    def build_like(
        other: "RawCache",
        length: int,
        device: str | None = None,
        pin_memory: bool = False,
    ) -> "RawCache":
        """
        Allocate a cache for the associated model of ``other``.
        The underlying tensors may be pinned to enable async
        copies.
        """
        c = other.caches[0]
        depth = len(other.caches)
        shape = (depth, length, c.shape[1])
        new_c = torch.empty(
            shape,
            dtype=c.dtype,
            device=device if device is not None else c.device,
            pin_memory=pin_memory,
        )
        caches = [new_c[n] for n in range(depth)]
        return RawCache(caches, other.split, length)

    def _cache(self, n: int) -> torch.Tensor:
        """
        A single tensor with all the cached data
        for layer n; the tensor is writeable.
        """
        start, end = self.start, self.start + self.length
        return self.caches[n][start:end]

    def cache_kv(self, n: int) -> tuple[torch.Tensor, ...]:
        return torch.split(self._cache(n), self.split, dim=1)

    def view(self, start: int, length: int) -> "RawCache":
        """
        Take a view along the sequence axis of a larger cache.

        The original cache object remains of identical size and valid
        after the shrunk alias has been used.

        Args:
            start (int): the start position in each layer's cache
            length (int): the number of items to keep in the view

        Returns:
            A view in the cache object.
        """
        assert start + length <= self.length
        return RawCache(
            self.caches,
            self.split,
            start=self.start + start,
            length=length,
        )

    def bytes_per_token(self) -> int:
        """
        Return the number of bytes required to cache one token.
        """
        caches = self.caches
        if len(caches) == 0:
            return 0

        c = caches[0]
        return len(caches) * c[0].numel() * c.element_size()

    def clear(self) -> None:
        for c in self.caches:
            c.zero_()


@dataclass
class DynCacheLane:
    """Dynamic cache."""

    gpu_blocks: list[int]
    cpu_blocks: list[RawCache]
    ready_count: int

    def h2d_blocks(self) -> Iterable[tuple[int, RawCache]]:
        return itertools.islice(
            zip(self.gpu_blocks, self.cpu_blocks, strict=False),
            self.ready_count,
        )

    def d2h_blocks(self) -> Iterable[tuple[int, RawCache]]:
        return itertools.islice(
            zip(self.gpu_blocks, self.cpu_blocks, strict=False),
            self.ready_count,
            None,
        )


@dataclass
class DynCache(ModelCache):
    """
    A model cache class that dynamically pages
    in and out kv-caches from/to host memory.
    A ``DynCache`` object is intended to be
    used by a single ``fwd.prefill()`` call.
    """

    Mode = Literal["page_in", "page_out"]
    MODE_BITS = {
        "page_in": 1,
        "page_out": 2,
    }

    gpu_cache: RawCache
    host_cache: "Cache"
    cache_lanes: list[DynCacheLane]
    h2d_events: list[tuple[torch.cuda.Event, int]] = field(default_factory=list)
    mode: int = 3

    def disable(self, mode: Mode) -> None:
        self.mode &= ~DynCache.MODE_BITS[mode]

    def enable(self, mode: Mode) -> None:
        self.mode |= DynCache.MODE_BITS[mode]

    def page_in(self, n: int) -> None:
        if self.host_cache.disabled or not self.mode & 1:
            return

        assert all(nn < n for _, nn in self.h2d_events)

        with torch.cuda.stream(self.host_cache.h2d_stream):
            node_len = self.host_cache.node_len
            for lane in self.cache_lanes:
                dc = self.gpu_cache._cache(n)
                for gpu_block, cpu_block in lane.h2d_blocks():
                    start = gpu_block * node_len
                    end = start + node_len
                    sc = cpu_block._cache(n)
                    dc[start:end].copy_(sc, non_blocking=True)

        ev = torch.cuda.Event()
        ev.record(self.host_cache.h2d_stream)
        self.h2d_events.append((ev, n))

    def page_out(self, n: int) -> None:
        if self.host_cache.disabled or not self.mode & 2:
            return

        ev = torch.cuda.Event()
        ev.record(torch.cuda.default_stream())

        with torch.cuda.stream(self.host_cache.d2h_stream):
            ev.wait()
            node_len = self.host_cache.node_len
            for lane in self.cache_lanes:
                sc = self.gpu_cache._cache(n)
                for gpu_block, cpu_block in lane.d2h_blocks():
                    start = gpu_block * node_len
                    end = start + node_len
                    dc = cpu_block._cache(n)
                    dc.copy_(sc[start:end], non_blocking=True)

    def host_cache_ready(self) -> torch.cuda.Event | None:
        """
        Return an event that must be waited on before
        the on-device caches are successfully synced
        to the host memory.
        """
        if self.host_cache.disabled:
            return None

        ev = torch.cuda.Event()
        ev.record(self.host_cache.d2h_stream)
        return ev

    def cache_kv(self, n: int) -> tuple[torch.Tensor, ...]:
        if not self.host_cache.disabled and (self.mode & 1):
            assert (
                self.h2d_events and self.h2d_events[0][1] == n
            ), f"sync event is not available for layer {n}"
            ev, _ = self.h2d_events.pop(0)
            ev.wait()  # ensure page-in is done

        return self.gpu_cache.cache_kv(n)


@dataclass
class CacheNode:
    toks_h: int
    toks: list[int]
    cache: RawCache
    clock: int
    node_id: int
    next: list["CacheNode"] = field(default_factory=list)

    def __lt__(self, other: "CacheNode") -> bool:
        # An arbitrary order to break ties in the cache
        # eviction logic. Note that we do not use Python
        # `id()` here because we need the logic to give
        # the same answer on all model-parallel ranks.
        return self.node_id < other.node_id

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class Cache:
    """Host cache for repeated prompts"""

    limit_toks: int
    node_len: int  # cache block length, in tokens
    page_len: int  # granularity of the trie, in tokens
    nodes: list[CacheNode] = field(default_factory=list)
    frozen: set[int] = field(default_factory=set)
    num_nodes: int = 0
    clock: int = 0
    next_id: int = 0
    downsize: float = 0.75  # downsizing factor in maintain()
    limit: int = 0  # in number of nodes
    disabled: bool = False

    free_caches: list[RawCache] = field(default_factory=list)
    h2d_stream: torch.cuda.Stream = None  # type: ignore
    d2h_stream: torch.cuda.Stream = None  # type: ignore

    evicted: set[int] = field(default_factory=set)

    def __post_init__(self):
        self.limit = self.limit_toks // self.page_len
        self.disabled = self.limit == 0
        self._create_streams()

    def _create_streams(self) -> None:
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    def tick(self) -> None:
        "Tick the cache clock."
        self.frozen.clear()
        self.clock += 1

    def maintain(self) -> None:
        """
        Ensure the cache stays within its token limit.
        """
        if self.num_nodes <= self.limit:
            return

        target = int(self.downsize * self.limit)
        to_evict = self.num_nodes - target

        nodes: list[tuple[int, int, CacheNode, list[CacheNode]]] = []
        stk = [(n, self.nodes, 0) for n in self.nodes]
        while stk:
            n, p, d = stk.pop()
            nodes.append((n.clock, d, n, p))
            stk.extend((c, n.next, d - 1) for c in n.next)

        assert self.num_nodes == len(nodes)
        self.num_nodes -= to_evict
        logger.info("Cache will evict %s nodes", to_evict)

        heapq.heapify(nodes)
        while to_evict > 0:
            n, p = heapq.heappop(nodes)[2:]
            if metrics.recording():
                self.evicted.add(n.toks_h)
            assert len(n.next) == 0
            p.remove(n)
            self.free_caches.append(n.cache)
            to_evict -= 1

    def prepare_lane(
        self,
        gpu_cache: RawCache,
        blocks: list[int],
        tokens: list[int],
        no_insert: bool = False,  # for testing purposes
    ) -> DynCacheLane:
        """
        Construct a cache lane object to be used by a
        prefill call. The lane will contain references
        to cache blocks on host memory that can either
        be used to save computation or that must be
        populated during prefilling.

        The lane produced should be used to construct
        a ``DynCache`` object to pass to the prefill
        function.
        """
        total_tokens = len(tokens)

        if self.disabled:
            metrics.dump(
                "cache",
                tokens=total_tokens,
                cached=0,
                evict_misses=0,
            )
            return DynCacheLane(blocks, [], 0)

        ready = 0
        evict_misses = 0
        caches: list[RawCache] = []
        inserting = False
        nodes = self.nodes
        while len(tokens) >= self.page_len:
            toks = tokens[: self.page_len]
            toks_h = hashints(toks)
            tokens = tokens[self.page_len :]

            if not inserting:
                for n in nodes:
                    if id(n) in self.frozen:
                        continue
                    if n.toks_h == toks_h and n.toks == toks:
                        n.clock = self.clock
                        caches.append(n.cache)
                        nodes = n.next
                        break
                else:
                    ready = len(caches)
                    inserting = True
                    if no_insert:
                        break

            if inserting:
                if self.free_caches:
                    cache = self.free_caches.pop()
                else:
                    cache = RawCache.build_like(
                        gpu_cache,
                        length=self.node_len,
                        device="cpu",
                        pin_memory=True,
                    )

                n = CacheNode(
                    toks_h=toks_h,
                    toks=toks,
                    cache=cache,
                    clock=self.clock,
                    node_id=self.next_id,
                )
                if toks_h in self.evicted:
                    evict_misses += 1
                nodes.append(n)
                self.frozen.add(id(n))
                caches.append(n.cache)
                nodes = n.next
                self.num_nodes += 1
                self.next_id += 1

        ready = ready if inserting else len(caches)

        metrics.dump(
            "cache",
            tokens=total_tokens,
            cached=ready * self.page_len,
            evict_misses=evict_misses * self.page_len,
        )

        return DynCacheLane(blocks, caches, ready)

    def preallocate(self, gpu_cache: RawCache) -> None:
        """
        Preallocate the cache in host memory. Note
        that this is a surprisingly costly operation
        for large caches (e.g., 6s for 10G).
        """
        if self.disabled:
            return

        rc = RawCache.build_like(
            gpu_cache,
            length=self.node_len * self.limit,
            device="cpu",
            pin_memory=True,  # to enable async transfers
        )
        for i in range(self.limit):
            self.free_caches.append(
                rc.view(
                    start=i * self.node_len,
                    length=self.node_len,
                )
            )
