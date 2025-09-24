# Copyright (c) Meta Platforms, Inc. and affiliates.

import dataclasses
from collections.abc import Generator
from queue import Queue
from typing import Any, Generic, NoReturn, TypeVar

from cwm.text import tokenizers as cwm_tokenizers


@dataclasses.dataclass
class Packet:
    thread_id: Any
    "An arbitrary identifier to link inputs to outputs."
    tokens: list[int] = dataclasses.field(default_factory=list)
    "The prompt if used as input, or else the generation output."
    text: str | None = None
    "The prompt given as string, used if ``tokens`` is empty."
    temperature: float | None = None
    "Optional temperature setting."
    top_p: float | None = None
    "Optional top-p nucleus sampling setting."
    stop_str: str | None = None
    "If not None, the generation stops after the first occurrence of the stop_str."
    logprobs: list[float] | None = None
    """
    Logprobs for the tokens list. In output packets only,
    and only when requested.
    """
    sample_logprobs: list[float] | None = None
    """
    Similar to `logprobs`, but `sample_logprobs` are adjusted by
    temperature and top-p sampling. In output packets only,
    and only when requested.
    """
    max_gen: int | None = None
    "The maximum number of tokens to generate."


# An empty type to be used as type parameter of GenAPI
# when the generator never returns errors.
NoErr = NoReturn


# Parametric error type for GenAPI.
TErr = TypeVar("TErr")


class GenAPI(Generic[TErr]):
    """
    A one-shot request response API to a language model.

    The API defined by this class enables processing multiple
    one-shot conversation threads concurrently.

    Override 'generate' in child classes.
    """

    tokenizer: cwm_tokenizers.Tokenizer
    max_batch: int

    def update_model(self) -> None:
        """
        Callback to invoke when the underlying weight tensors have
        been updated in-place.
        """

    def generate(
        self, q: Queue[Packet | None]
    ) -> Generator[Packet | TErr | None, None, None]:
        """
        Read from the queue and generate a completion or an error
        for each prompt. The specific error type returned depends
        on the implementation of the generator.

        The outputs may come in a different order from the inputs
        so that latency can be kept minimal. We use a queue to pass
        requests in order to keep generation running even when the
        input lags. To terminate the generation, send None in the
        queue; in-flights generations will be completed before
        termination.

        The generator may output None at any time to hand over
        control to the calling code.
        """
        raise NotImplementedError
