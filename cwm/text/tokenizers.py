# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
import functools
import logging
import operator
from abc import abstractmethod
from copy import copy
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from cwm.text.datatypes import (
    CWMChatMessage,
    MessageBase,
    Role,
)

logger = logging.getLogger()


class Tokenizer(abc.ABC):
    version: str

    @property
    @abstractmethod
    def n_words(self) -> int:
        """Number of tokens in the tokenizer's vocabulary"""
        ...

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """Beginning of text token id"""
        ...

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """End of text token id"""
        ...

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Padding token id"""
        ...

    @property
    def has_bos_id(self) -> bool:
        """Whether the tokenizer has a BOS token."""
        return True

    @abstractmethod
    def id_to_piece(self, token_id: int) -> str:
        """Convert a token id to the token str"""

    @abstractmethod
    def piece_to_id(self, piece: str) -> int:
        """Convert a token str to the token id"""

    @property
    def stop_tokens(self) -> list[int]:
        """Set of stop tokens to cut the decoding"""
        return [self.eos_id]

    @abstractmethod
    def _encode(self, text: str) -> list[int]:
        """String to token ids"""

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        assert isinstance(text, str)
        t = self._encode(text)
        if bos and self.has_bos_id:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    @abstractmethod
    def _decode(self, tokens: list[int]) -> str:
        """Token ids to string"""

    def decode(self, tokens: list[int], *, cut_at_stop_tokens: bool = True) -> str:
        """Token ids to string"""
        if cut_at_stop_tokens:
            tokens = self.cut_at_stop_tokens(tokens, keep_stop_token=False)
        return self._decode(tokens)

    def cut_at_stop_tokens(
        self, tokens: list[int], *, keep_stop_token: bool = True
    ) -> list[int]:
        """Trim the suffix of ``tokens`` that follows the first stop token.
        Whether the stop token is included in the output or not is
        controlled by the ``keep_stop_token`` flag.
        """
        if self.stop_tokens is None:
            return tokens

        for k, t in enumerate(tokens):
            if t in self.stop_tokens:
                return tokens[: k + (1 if keep_stop_token else 0)]
        return tokens


class InstructTokenizer(Tokenizer):
    message_cls: type[MessageBase]

    @property
    @abstractmethod
    def eom_id(self) -> int:
        """End of message token id"""

    @property
    @abstractmethod
    def eot_id(self) -> int:
        """End of turn token id"""

    @abstractmethod
    def encode_message_header(self, message: MessageBase) -> list[int]:
        """Encode a message header to a sequence of tokens"""

    @abstractmethod
    def encode_message(
        self, message: MessageBase, add_end_token: bool = True
    ) -> list[int]:
        """Encode a message to a sequence of tokens"""

    @property
    def has_think_token_ids(self) -> bool:
        return False

    @property
    def think_token_ids(self) -> list[int]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement think_token_ids"
        )

    def encode_prompt_dialog(
        self,
        dialog: list[MessageBase],
        *,
        bos: bool = True,
        think: bool = False,
    ) -> list[int]:
        tokens: list[int] = []

        if bos and self.has_bos_id:
            tokens.append(self.bos_id)
        if not dialog:
            return tokens

        for msg in dialog[:-1]:
            tokens.extend(self.encode_message(msg))

        last_msg = dialog[-1]
        if last_msg.source == Role.assistant:
            assert not think, "think=True not supported for last message from assistant"
            # Last message from assistant is treated as a generation prefix
            tokens.extend(self.encode_message(last_msg, add_end_token=False))
        else:
            tokens.extend(self.encode_message(last_msg))

            # Append assistant header as generation prefix
            header_tokens = self.encode_message_header(
                self.message_cls(
                    source=Role.assistant,
                    body="",
                    version=self.message_cls.version,
                )
            )
            tokens.extend(header_tokens)

            if think:
                if not self.has_think_token_ids:
                    raise ValueError(
                        f"think=True not supported for {self.__class__.__name__}"
                    )
                tokens.extend(self.think_token_ids)

        return tokens


class TikTokenTokenizer(Tokenizer):
    """Tiktoken Tokenizer"""

    version: str = "tiktoken"

    BEGIN_OF_TEXT = "<|begin_of_text|>"
    END_OF_TEXT = "<|end_of_text|>"
    PAD = "<|pad|>"

    NUM_RESERVED_TOKENS = 256
    DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    DEFAULT_TIKTOKEN_SPECIAL_TOKENS: ClassVar = {
        BEGIN_OF_TEXT: 0,
        END_OF_TEXT: 1,
        PAD: 4,
    }
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000

    def __init__(self, model_path: str) -> None:
        super().__init__()
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = self.get_all_special_tokens_with_ids(
            mergeable_ranks
        )
        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=self.DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )
        self._n_words = self.tkt_model.n_vocab
        self._bos_id = self.piece_to_id(self.BEGIN_OF_TEXT)
        self._eos_id = self.piece_to_id(self.END_OF_TEXT)
        self._pad_id = self.piece_to_id(self.PAD)
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id if self.has_bos_id else 'No bos_id'} - EOS ID: {self.eos_id}",
        )

    @property
    def n_words(self) -> int:
        return self._n_words

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_all_special_tokens_with_ids(
        self, mergeable_ranks: dict[bytes, int]
    ) -> dict:
        all_special_tokens_with_ids = copy(self.DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(self.NUM_RESERVED_TOKENS)) - set(
            all_special_tokens_with_ids.values()
        )
        for id_ in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id_}|>"] = id_
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)
        return all_special_tokens_with_ids

    def id_to_piece(self, token_id: int) -> str:
        return self.tkt_model.decode_single_token_bytes(token_id).decode()

    def piece_to_id(self, piece: str) -> int:
        return self.tkt_model.encode_single_token(piece)

    def _encode(self, text: str) -> list[int]:
        if len(text) < self.TIKTOKEN_MAX_ENCODE_CHARS:
            return self.tkt_model.encode_ordinary(text)
        subs = [
            text[i : i + self.TIKTOKEN_MAX_ENCODE_CHARS]
            for i in range(0, len(text), self.TIKTOKEN_MAX_ENCODE_CHARS)
        ]
        return functools.reduce(
            operator.iadd,
            self.tkt_model.encode_ordinary_batch(subs),
            [],
        )

    def _decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)


class CWMInstructTokenizer(TikTokenTokenizer, InstructTokenizer):
    """
    Instruct tokenizer with special tokens for chat dialogs and
    execution traces.
    """

    message_cls: type[MessageBase] = CWMChatMessage
    version = "cwm_instruct"

    BEGIN_OF_TEXT = "<|begin_of_text|>"
    END_OF_TEXT = "<|end_of_text|>"
    PAD = "<|pad|>"

    START_HEADER_ID = "<|start_header_id|>"
    END_HEADER_ID = "<|end_header_id|>"
    EOM_ID = "<|eom_id|>"  # end of message
    EOT_ID = "<|eot_id|>"  # end of turn

    FRAME_SEP_ID = "<|frame_sep|>"
    ACTION_SEP_ID = "<|action_sep|>"
    RETURN_SEP_ID = "<|return_sep|>"
    CALL_SEP_ID = "<|call_sep|>"
    LINE_SEP_ID = "<|line_sep|>"
    EXCEPTION_SEP_ID = "<|exception_sep|>"
    ARG_SEP_ID = "<|arg_sep|>"
    TRACE_CONTEXT_START_ID = "<|trace_context_start|>"
    THINKING_START_ID = "<|reasoning_thinking_start|>"
    THINKING_END_ID = "<|reasoning_thinking_end|>"
    THINK_STR_START = "<think>\n"
    THINK_STR_END = "</think>"

    DEFAULT_TIKTOKEN_SPECIAL_TOKENS: ClassVar = {
        **TikTokenTokenizer.DEFAULT_TIKTOKEN_SPECIAL_TOKENS,
        START_HEADER_ID: 6,
        END_HEADER_ID: 7,
        EOM_ID: 8,
        EOT_ID: 9,
        FRAME_SEP_ID: 100,
        ACTION_SEP_ID: 101,
        RETURN_SEP_ID: 102,
        CALL_SEP_ID: 103,
        LINE_SEP_ID: 104,
        EXCEPTION_SEP_ID: 105,
        ARG_SEP_ID: 106,
        TRACE_CONTEXT_START_ID: 107,
        THINKING_START_ID: 120,
        THINKING_END_ID: 121,
    }

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path=model_path)
        self._eom_id = self.piece_to_id(self.EOM_ID)
        self._eot_id = self.piece_to_id(self.EOT_ID)
        self.header_start_id = self.piece_to_id(self.START_HEADER_ID)
        self.header_end_id = self.piece_to_id(self.END_HEADER_ID)
        self.frame_sep_id: int = self.piece_to_id(self.FRAME_SEP_ID)
        self.action_sep_id: int = self.piece_to_id(self.ACTION_SEP_ID)
        self.return_sep_id: int = self.piece_to_id(self.RETURN_SEP_ID)
        self.call_sep_id: int = self.piece_to_id(self.CALL_SEP_ID)
        self.line_sep_id: int = self.piece_to_id(self.LINE_SEP_ID)
        self.exception_sep_id: int = self.piece_to_id(self.EXCEPTION_SEP_ID)
        self.arg_sep_id: int = self.piece_to_id(self.ARG_SEP_ID)
        self.trace_context_start_id: int = self.piece_to_id(self.TRACE_CONTEXT_START_ID)
        self.thinking_start_id: int = self.piece_to_id(self.THINKING_START_ID)
        self.thinking_end_id: int = self.piece_to_id(self.THINKING_END_ID)
        assert self.has_bos_id

    @property
    def eom_id(self) -> int:
        return self._eom_id

    @property
    def eot_id(self) -> int:
        return self._eot_id

    @property
    def stop_tokens(self) -> list[int]:
        return [self.eom_id, self.eot_id, self.eos_id]

    def encode_message_header(self, message: MessageBase) -> list[int]:
        return [
            self.header_start_id,
            *self.encode(message.source_str.strip(), bos=False, eos=False),
            self.header_end_id,
            *self.encode("\n\n", bos=False, eos=False),
        ]

    def encode_thinking(self, text: str) -> list[int]:
        """
        similar to TikTokenTokenizer._encode, but with special handling
        for the thinking start and end tags
        """
        subs = [
            text[i : i + self.TIKTOKEN_MAX_ENCODE_CHARS]
            for i in range(0, len(text), self.TIKTOKEN_MAX_ENCODE_CHARS)
        ]
        return functools.reduce(
            operator.iadd,
            self.tkt_model.encode_batch(
                subs,
                allowed_special={self.THINKING_START_ID, self.THINKING_END_ID},
                disallowed_special=(),
            ),
            [],
        )

    def encode_message(
        self, message: MessageBase, add_end_token: bool = True
    ) -> list[int]:
        """
        identical to InstructTiktokenTokenizer.encode_message, for non-reasoning data
        only when there's either `THINKING_START_ID` or `THINKING_END_ID` in the message body,
        we'll use `encode_thinking` to encode the message body
        """
        if message.body is None:
            body_tokens = []
        else:
            # Note: if message body is empty, i.e. "", then no token is output by `encode`,
            # we'll get [] like above, so tokenization is identical with body="" and body=None
            body = message.body

            # We don't strip tool replies (for completeness) or assistant
            # prompts (to avoid tokenization issues)
            if message.source != Role.tool and not (
                message.source == Role.assistant and not add_end_token
            ):
                body = body.strip()

            if message.source == Role.assistant:
                # used for SFT on reasoning data
                body_tokens = self.encode_thinking(body)
            else:
                body_tokens = self.encode(body, bos=False, eos=False)

        header_tokens = self.encode_message_header(message)
        if add_end_token:
            end_token = self.eot_id if message.eot else self.eom_id
            return header_tokens + body_tokens + [end_token]
        return header_tokens + body_tokens

    @property
    def has_think_token_ids(self) -> bool:
        return True

    @cached_property
    def think_token_ids(self) -> list[int]:
        return self.encode(self.THINK_STR_START, bos=False, eos=False)


def build_tokenizer(name: str, path: str | None = None) -> Tokenizer:
    if name == TikTokenTokenizer.version:
        assert path is not None
        return TikTokenTokenizer(path)

    if name == CWMInstructTokenizer.version:
        assert path is not None
        return CWMInstructTokenizer(path)

    raise NotImplementedError(f"{name} tokenizer type is not implemented")
