# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import time
from typing import Literal
from uuid import uuid4

from openai.types.chat import (
    ChatCompletionMessageToolCallParam as ToolCall,
)
from pydantic import BaseModel, Field


class BaseSchema(BaseModel):
    type: str
    enum: list | None = None
    description: str | None = None


class ArraySchema(BaseModel):
    type: Literal["array"]
    description: str
    items: "JsonSchema | None" = None
    minItems: int = 0


class ObjectSchema(BaseModel):
    type: Literal["object"]
    properties: dict[str, "JsonSchema"]
    required: list[str]
    additionalProperties: bool


JsonSchema = ArraySchema | ObjectSchema | BaseSchema


class FunctionDefinition(BaseModel):
    name: str
    description: str | None
    parameters: dict


class ToolInfo(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class CompletionsRequest(BaseModel):
    prompt: list[int] | str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 8192
    min_tokens: int = 0  # ignored
    model: str | None = None  # ignored
    n: int = 1
    stop: str | None = None


class ContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def content_str(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(c.text for c in self.content)


class ReasoningConfig(BaseModel):
    enabled: bool = False
    effort: str | None = None
    summary: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    tools: list[ToolInfo] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 0
    min_tokens: int = 0  # ignored
    model: str | None = None  # ignored
    n: int = 1
    stop: str | None = None
    reasoning: ReasoningConfig | None = None
    reasoning_effort: str | None = None
    stream: bool = False  # ignored


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage


class OpenAIBase(BaseModel):
    object: str
    id: str = Field(default_factory=lambda: uuid4().hex)
    created: int = Field(default_factory=lambda: int(time.time()))


class ChatCompletionResponse(OpenAIBase):
    object: str = "chat.completion"
    system_fingerprint: str | None = None
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: CompletionUsage | None = None
    model: str


class ChatDeltaMessage(BaseModel):
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @staticmethod
    def from_chat(m: ChatMessage) -> "ChatDeltaMessage":
        tool_calls = copy.deepcopy(m.tool_calls)
        for i, tc in enumerate(tool_calls):
            tc["index"] = i  # type: ignore
        return ChatDeltaMessage(
            role=m.role,
            content=m.content_str,
            reasoning_content=m.reasoning_content,
            tool_calls=tool_calls,
        )


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatDeltaMessage
    logprobs: None = None
    finish_reason: str | None = None

    @staticmethod
    def from_chat(
        c: ChatCompletionChoice,
    ) -> "ChatCompletionStreamChoice":
        return ChatCompletionStreamChoice(
            index=c.index,
            delta=ChatDeltaMessage.from_chat(c.message),
            finish_reason="stop",
        )


class ChatCompletionStreamResponse(OpenAIBase):
    object: str = "chat.completion.chunk"
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: CompletionUsage | None = None

    @staticmethod
    def from_chat(
        r: ChatCompletionResponse,
    ) -> "ChatCompletionStreamResponse":
        Choice = ChatCompletionStreamChoice
        choices = [Choice.from_chat(c) for c in r.choices]
        return ChatCompletionStreamResponse(
            model=r.model, choices=choices, usage=r.usage
        )
