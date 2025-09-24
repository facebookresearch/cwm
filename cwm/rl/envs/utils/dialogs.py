# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
For the purpose of environment design, multiturn dialogs are purely lists of
messages.

Here we provide some utilities for manipulating and creating messages.
"""

from typing import Literal

from cwm.text.datatypes import MessageBase
from cwm.text.tokenizers import InstructTokenizer


def maybe_truncate_message(
    message: MessageBase,
    tokenizer: InstructTokenizer,
    *,
    max_tokens: int,
    abbrev_tokens: list[int] | None = None,
    add_end_token: bool = True,
    where: Literal["left", "right"] = "left",
) -> MessageBase:
    """
    Left or right truncation of a message body to fit a maximum number of
    tokens.
    """
    if abbrev_tokens is None:
        abbrev_tokens = []

    tokens = tokenizer.encode_message(message, add_end_token=add_end_token)  # type: ignore
    if len(tokens) > max_tokens:
        msg_body_tokens = tokenizer.encode(message.body, bos=False, eos=False)
        non_body_len = len(tokens) - len(msg_body_tokens)
        new_body_len = max_tokens - non_body_len - len(abbrev_tokens)
        new_tokens = (
            abbrev_tokens + msg_body_tokens[-new_body_len:]
            if where == "left"
            else msg_body_tokens[:new_body_len] + abbrev_tokens
        )
        message.body = tokenizer.decode(new_tokens)

    return message
