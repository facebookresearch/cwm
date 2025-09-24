# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass
class MessageBase:
    # Role. e.g. user/assistant/system/tool
    source: Role

    # Primary content of the message
    body: str

    # Message versioning
    version: str

    # Whether to use an EOT token when ending a message or EOM (legacy)
    eot: bool = False

    # Use metadata for experimental information
    metadata: Any | None = None

    @property
    def source_str(self) -> str:
        return self.source.strip()

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CWMChatMessage(MessageBase):
    version: str = "cwm_message_v1"

    # Always use EOT token
    eot: bool = True

    # Tool name for source==Role.tool
    tool: str = ""

    @property
    def source_str(self) -> str:
        if self.source == Role.tool:
            return f"{self.source.strip()}: {self.tool}"
        else:
            return self.source.strip()

    @classmethod
    def system(cls, body: str) -> "CWMChatMessage":
        return CWMChatMessage(source=Role.system, body=body, eot=True)

    @classmethod
    def user(cls, body: str) -> "CWMChatMessage":
        return CWMChatMessage(source=Role.user, body=body, eot=True)

    @classmethod
    def assistant(cls, body: str = "") -> "CWMChatMessage":
        return CWMChatMessage(source=Role.assistant, body=body, eot=True)

    @classmethod
    def assistant_eot(cls, body: str = "") -> "CWMChatMessage":
        return cls.assistant(body)

    @classmethod
    def tool_result(cls, body: str, *, tool: str) -> "CWMChatMessage":
        return CWMChatMessage(source=Role.tool, tool=tool, body=body, eot=True)

    def __str__(self) -> str:
        body = repr(self.body)
        ending = "eot" if self.eot else "eom"
        if self.source == Role.tool:
            source = f"{self.source}: {self.tool}"
        else:
            source = self.source
        return f"[{source},{ending}] {body}"

    def assert_valid(self) -> None:
        self.check_version()
        self.check_source()
        self.check_body()
        self.check_eot()

    def check_source(self) -> None:
        assert self.source in [Role.user, Role.assistant, Role.system, Role.tool]
        if self.source == Role.tool:
            assert self.tool != ""  # always want a tool name

    def check_version(self) -> None:
        assert self.version == "cwm_message_v2"

    def check_body(self) -> None:
        if self.source in [Role.system, Role.tool]:
            return
        assert self.body is not None
        assert self.body.strip() != ""

    def check_eot(self) -> None:
        # CWMChatMessages always use EOT
        assert self.eot

    @classmethod
    def from_dict(cls, repr_dict: dict) -> "CWMChatMessage":
        return CWMChatMessage(
            source=repr_dict["source"],
            eot=repr_dict["eot"],
            body=repr_dict["body"],
            metadata=repr_dict.get("metadata"),  # ok if not present
            tool=repr_dict.get("tool", ""),  # ok if not present
        )


@dataclass
class BaseTextDatum(Generic[T]):
    @dataclass
    class Source:
        path: Path
        line_no: int
        pos: int

    val: T
    _: KW_ONLY
    src: Source | Sequence[Source] | Any = None

    def __len__(self):
        return len(self.val)


@dataclass
class StrDatum(BaseTextDatum[str]):
    pass


@dataclass
class DictDatum(BaseTextDatum[dict]):
    pass
