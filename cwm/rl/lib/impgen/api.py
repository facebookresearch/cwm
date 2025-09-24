# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC, abstractmethod

from cwm.fastgen import api
from cwm.text.tokenizers import Tokenizer


class ImpGenAPI(ABC):
    """
    Abstract imperative‐style generation engine.
    All backends should implement this interface.
    """

    @abstractmethod
    def generate(
        self,
        tokens: list[int],
        max_gen: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_str: str | None = None,
    ) -> api.Packet:
        """
        Synchronous generate: block until this request is done and return its Packet.
        """

    @abstractmethod
    def work(self) -> bool:
        """
        Progress any in‐flight asynchronous work.
        Return True if “done” (no more work ever), False otherwise.

        Use this function if your generation engine needs to run in the main thread, i.e. when the backend is FastGen.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Signal “no more generate() calls” and cleanly wind down.

        Use this function if your generation engine needs to run in the main thread.
        """

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        """
        The Tokenizer used by this engine.
        """
