# Copyright (c) Meta Platforms, Inc. and affiliates.

import threading
from queue import Queue
from threading import Condition, Lock
from typing import Generic, TypeVar

import moodist
import torch

from cwm.fastgen.api import GenAPI, Packet
from cwm.text.tokenizers import Tokenizer

from .api import ImpGenAPI

T = TypeVar("T")


class Future(Generic[T]):
    """
    A Future represents the result of a computation that may become available in the future.
    We create a custom class for this because the docs say not to instantiate a concurrent.futures.Future by yourself.
    """

    def __init__(self):
        self._x: T | None = None
        self._c: Condition = Condition(Lock())
        self._ready: bool = False

    def set(self, val: T) -> None:
        with self._c:
            if self._ready:
                raise RuntimeError("Future has already been set")
            self._x = val
            self._ready = True
            self._c.notify_all()

    def get(self) -> T:
        with self._c:
            self._c.wait_for(self.ready)
            assert self._x is not None
            return self._x

    def ready(self) -> bool:
        return self._ready


class ImpGen(ImpGenAPI):
    """
    Simple imperative interface to FastGen
    Generate tokens by ```result = ig.generate(prompt_tokens)```

    This class handles all the multiprocessing / broadcasting / synchronization / queueing boilerplate,
    making user code easier to read and write.

    Intended use:
    - Repeatedly call ig.work() on the main thread on all model parallel ranks
    - Call ig.generate() in dedicated threads on any rank
    """

    def __init__(
        self,
        g: GenAPI,
        tp_rank: int,
        tp_group: torch.distributed.ProcessGroup,
    ):
        self.g = g
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.tp_size = tp_group.size()

        self.nseqs: int = 0  # Number of sequences generated
        self.ntoks: int = 0  # Number of tokens generated

        self.i = 0
        self.done = False
        self.lock = threading.Lock()

        # _q is the FastGen input queue
        # _tp_queue is used to broadcast packets across ranks in a tp_group
        # _sync_queue synchronizes _q across ranks in the tp_group
        # this enables calling ig.generate() on all ranks in a tp_group
        # without having to synchronize the prompts
        self._q = Queue[Packet | None]()
        self._tp_queue = moodist.Queue(tp_group, location=range(self.tp_size))
        self._sync_queue = moodist.Queue(tp_group, location=range(self.tp_size))
        self._stop_queue = moodist.Queue(tp_group, location=range(self.tp_size))
        self._generator = g.generate(self._q)
        self._in_flight: int = 0
        self._futures: dict[int, Future[Packet]] = {}

    def update_model(self) -> None:
        """
        Callback to invoke when the underlying weight tensors have
        been updated in-place.
        """
        self.g.update_model()

    def generate(
        self,
        tokens: list[int],
        max_gen: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_str: str | None = None,
    ) -> Packet:
        """
        Generate output tokens given input tokens using imperative syntax:
        ```result = ig.generate(prompt_tokens)```

        Note:
        - This method will block, so should be called from a dedicated thread.
        """
        fut = self.generate_future(tokens, max_gen, temperature, top_p, stop_str)
        return fut.get()

    def generate_future(
        self,
        tokens: list[int],
        max_gen: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_str: str | None = None,
    ) -> Future[Packet]:
        with self.lock:
            thread_id = self.i * self.tp_size + self.tp_rank
            self.i += 1
        assert isinstance(thread_id, int)

        # Enqueue generation task
        fut = Future[Packet]()
        self._futures[thread_id] = fut
        packet = Packet(
            thread_id,
            tokens,
            temperature=temperature,
            top_p=top_p,
            max_gen=max_gen,
            stop_str=stop_str,
        )
        self._tp_queue.put_object(packet)

        return fut

    def stop(self) -> None:
        """
        Stop FastGen.
        This will eventually make work() return done=True,
        signalling that we're done generating all requests and no further work calls are needed.
        """
        if not self.done:
            self._stop_queue.put_object(self.tp_rank)
            self.done = True
        if self._stop_queue.qsize() == self.tp_size and self.tp_rank == 0:
            self._tp_queue.put_object(None)

    def work(self) -> bool:
        """
        Check for new generation requests, and work on pending requests.
        Must be called from the main thread until the returned done flag is True.

        This methods returns as soon as a generation is finished,
        or when there are no pending requests at the moment.
        It will only block to do GPU work, never to wait for new generation requests.
        """
        # Sync the fastgen queues
        self._sync_q(block=self._in_flight == 0)

        if self._in_flight == 0:
            return False

        try:
            # This should never block to wait for requests, since _in_flight > 0
            pkt = next(self._generator)

        except StopIteration:
            assert self._in_flight == 1, f"{self._in_flight=} {len(self._futures)=}"
            assert len(self._futures) == 0, f"{self._in_flight=} {len(self._futures)=}"
            return True
        if pkt is not None:
            self._in_flight -= 1
        # pkt.thread_id will be in self._futures only on the rank that called generate.
        # Other ranks will ignore this packet even though they participated in the fastgen call that generated it.
        if pkt is not None and pkt.thread_id in self._futures:
            fut = self._futures.pop(pkt.thread_id)
            fut.set(pkt)
            self.nseqs += 1
            self.ntoks += len(pkt.tokens)

        return False

    def _sync_q(self, block: bool = False):
        # Get packets from _q, synchronize, then add them to the fastgen queue _q
        # After calling this function, tp ranks will have the same _q, as required by fastgen
        # This method is only called from work() on the main thread

        packets: list[Packet] = []
        while True:
            if self.tp_rank == 0:
                if block and len(packets) == 0 and self._tp_queue.empty():
                    self._tp_queue.wait(timeout=1)

                self._sync_queue.put_object(not self._tp_queue.empty())

            if self._sync_queue.get_object():
                packets.append(self._tp_queue.get_object())
            else:
                break

        for p in packets:
            self._q.put(p)
            # if p is not None:
            self._in_flight += 1

    @property
    def tokenizer(self) -> Tokenizer:
        return self.g.tokenizer
