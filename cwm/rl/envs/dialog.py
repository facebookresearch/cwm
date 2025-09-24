# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from cwm.rl.envs.api import Env, State, Transition
from cwm.rl.envs.tool_executor import ToolExecutor
from cwm.rl.envs.utils.dialogs import maybe_truncate_message
from cwm.text.datatypes import MessageBase, Role
from cwm.text.tokenizers import InstructTokenizer, Tokenizer


@dataclass
class DialogState(State):
    # Episode step counter
    step_id: int
    # Episode data, e.g., time-constant information from `episode_args`
    episode: Any
    # Tool executor (tools may keep per-episode state)
    tool_executor: ToolExecutor | None = None


class DialogEnv(Env):
    """
    Abstract Env subclass that provides convenience functions for prompting
    with an instruction model message format.
    """

    tokenizer: InstructTokenizer
    message_cls: type[MessageBase]

    def __init__(
        self,
        tokenizer: Tokenizer,
        abbrev_str: str = "[...]",
    ) -> None:
        assert isinstance(tokenizer, InstructTokenizer)
        self.tokenizer = cast(InstructTokenizer, tokenizer)
        self.message_cls = self.tokenizer.message_cls
        self.abbrev_tokens = self.tokenizer.encode(abbrev_str, bos=False, eos=False)

    def maybe_truncate(
        self,
        message: MessageBase,
        *,
        max_tokens: int,
        where: Literal["left", "right"] = "left",
    ) -> MessageBase:
        return maybe_truncate_message(
            message=message,
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            abbrev_tokens=self.abbrev_tokens,
            where=where,
        )

    def transition(
        self,
        *,
        messages: Sequence[MessageBase] | None = None,
        action: list[int] | None = None,
        action_str: str | None = None,
        initial: bool = False,
        terminal: bool = False,
        outcomes: dict[str, Any] | None = None,
        info: dict | None = None,
        think: bool = False,
    ) -> Transition:
        """
        Create a Transition with the given messages as observation.

        Assistant prompts are not automatically added. This method takes care
        of adding the trailing EOT token of the previous action if it is
        missing, e.g., due truncation.

        If `think` is True, the `thinking_start_id` token is added to the observation.
        """

        observation: list[int] = []
        observation_str: str | None = None
        if messages:
            if (
                action is not None
                and self.tokenizer.eot_id is not None
                and action[-1] != self.tokenizer.eot_id
            ):
                observation.append(self.tokenizer.eot_id)

            observation += self.tokenizer.encode_prompt_dialog(
                list(messages), bos=initial
            )

            if think:
                assert messages[-1].source == Role.assistant
                observation.append(self.tokenizer.thinking_start_id)
            observation_str = self.tokenizer.decode(
                observation, cut_at_stop_tokens=False
            )

        return Transition(
            action=action,
            action_str=action_str,
            observation=observation,
            observation_str=observation_str,
            terminal=terminal,
            outcomes=outcomes,
            info=info,
        )
