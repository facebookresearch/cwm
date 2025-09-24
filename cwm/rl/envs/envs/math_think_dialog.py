# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass
from typing import Any, Literal

import re2

from cwm.exec.math.compare import (
    CompareResult,
    exec_compare_values,
    exec_math_verify_compare_values,
)
from cwm.rl.envs import api as env_api
from cwm.rl.envs import config, dialog, prompts
from cwm.rl.envs.outcomes import thought_info
from cwm.rl.envs.tool_executor import ToolExecutor, python_tool_limits
from cwm.text.datatypes import MessageBase
from cwm.text.tokenizers import Tokenizer

logger = logging.getLogger()


@dataclass
class MathThinkDialogEpisode:
    task_id: str
    problem: str
    answer: str


@dataclass
class MathThinkDialogCWMEpisode(MathThinkDialogEpisode):
    past_actions: list[Any] | None = None
    prediction_type: Literal["full", "answer_only"] = "full"
    multiturn_dialogs: list[tuple[str, str]] | None = None


class MathThinkDialogEnv(dialog.DialogEnv):
    """
    Single-turn math reasoning environment. It imposes reasoning within the "<think> </think>" tag and answer within "<answer> </answer>" tag.
    """

    # Formatting templates
    prompt_template = "Solve the following math problem: {problem}. Provide the final answer in latex boxed like so: <think> YOUR REASONING HERE </think> <answer> $\\boxed{{YOUR ANSWER HERE}}$ </answer>."

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_attempts: int = 3,
        max_prompt_len: int = 2048,
        max_action_len: int = 1024,
        answer_only_action_len: int = 512,
        stop_str: str | None = None,
        stop_think: bool = False,
    ) -> None:
        super().__init__(tokenizer)

        self.max_attempts = max_attempts
        self.max_prompt_len = max_prompt_len
        self._max_action_len = max_action_len
        self.answer_only_action_len = answer_only_action_len
        self.stop_str = stop_str
        self.stop_think = stop_think

    def initial_messages(
        self, episode: MathThinkDialogEpisode, tool_executor: ToolExecutor | None
    ) -> list[MessageBase]:
        assert tool_executor is None, "tools not supported"
        prompt = self.prompt_template.format(problem=episode.problem)
        return [
            self.maybe_truncate(
                self.message_cls.user(prompt), max_tokens=self.max_prompt_len
            ),
            self.message_cls.assistant("<think>"),
        ]

    def max_action_len(self, state: env_api.State) -> int:
        return self._max_action_len

    @property
    def default_outcomes(self) -> dict[str, Any]:
        return {
            "pass": False,
            "ours_pass": False,
            "math_verify_pass": False,
            "reasoning_found": False,
            "n_reasoning_tokens": 0,
            "answer": None,
            "action_len": 0,
        }

    def start(
        self, episode_args: dict | None = None
    ) -> tuple[env_api.State, env_api.Transition]:
        assert episode_args is not None, "episode_args must be provided."
        episode = MathThinkDialogCWMEpisode(
            task_id=episode_args.get("task_id", "N/A"),
            problem=episode_args["problem"],
            answer=episode_args["answer"],
            past_actions=[],
            prediction_type="full",
        )

        return (
            dialog.DialogState(step_id=0, episode=episode),
            self.transition(
                initial=True,
                messages=self.initial_messages(episode, None),
                # The think is forced to be False as it's added to the assistant message
                # We do it as we assume the "<think>" is not a single token
                think=False,
            ),
        )

    def _extract_reasoning_and_answer(self, text: str) -> tuple[str | None, str | None]:
        # Pattern to match <think> and <answer> blocks
        pattern = r"(.*?)</think>\s*<answer>(.*?)</answer>"
        opt = re2.Options()
        opt.dot_nl = True

        if match := re2.search(pattern, text, opt):
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

        return None, None

    def _extract_answer(self, text: str) -> str | None:
        pattern = r"<answer>(.*?)</answer>"
        opt = re2.Options()
        opt.dot_nl = True

        if match := re2.search(pattern, text, opt):
            answer = match.group(1).strip()
            return answer

        return None

    def _eval(
        self, prediction: str, ground_truth: str, timeout: int
    ) -> tuple[CompareResult, CompareResult]:
        """
        Evaluates the prediction against the ground truth.
        """

        cwm_comp_res = exec_compare_values(prediction, ground_truth, timeout=timeout)

        math_verify_comp_res = exec_math_verify_compare_values(
            prediction, ground_truth, timeout=timeout
        )

        return cwm_comp_res, math_verify_comp_res

    def _eval_and_compute_metrics(
        self,
        state: env_api.State,
        action: list[int],
        reasoning: str | None,
        answer: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Evaluates the reasoning and answer, computes metrics, and returns the outcomes and info.
        """
        outcomes = self.default_outcomes

        th_info = thought_info(reasoning, self.tokenizer)

        outcomes["action_len"] = len(action)
        outcomes["reasoning_found"] = reasoning is not None
        outcomes["n_reasoning_tokens"] = (
            th_info["n_tokens"] if reasoning is not None else 0
        )

        info = {
            "reasoning": reasoning,
            "answer": answer,
            "answer_extracted": None,
            "ground_truth": str(state.episode.answer),
            "ground_truth_extracted": None,
            **th_info,
        }

        if answer is not None:
            cwm_comp_res, math_verify_comp_res = self._eval(
                answer, str(state.episode.answer), timeout=60
            )
            outcomes["ours_pass"] = cwm_comp_res.result
            info["answer_extracted"] = cwm_comp_res.normalized_expr_1
            info["ground_truth_extracted"] = cwm_comp_res.normalized_expr_2
            info["compare_info"] = cwm_comp_res.info
            info["compare_duration"] = str(cwm_comp_res.duration)

            outcomes["math_verify_pass"] = math_verify_comp_res.result
            info["math_verify_answer_extracted"] = (
                math_verify_comp_res.normalized_expr_1
            )
            info["math_verify_ground_truth_extracted"] = (
                math_verify_comp_res.normalized_expr_2
            )

            outcomes["pass"] = outcomes["ours_pass"] or outcomes["math_verify_pass"]

        if info["answer_extracted"] is not None:
            outcomes["answer"] = str(info["answer_extracted"])

        return outcomes, info

    def _stop_think_condition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> bool:
        return (
            self.stop_think
            and "</think>" not in action_str
            and state.episode.prediction_type == "full"
            and action[-1] not in self.tokenizer.stop_tokens
        )

    def _stop_think_transition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> env_api.Transition:
        state.episode.prediction_type = "answer_only"
        observation_str = "...</think>"
        observation = self.tokenizer.encode(observation_str, bos=False, eos=False)

        return env_api.Transition(
            action=action,
            action_str=action_str,
            observation=observation,
            observation_str=observation_str,
            terminal=False,
            outcomes=self.default_outcomes | {"action_len": len(action)},
        )

    def step(self, state: env_api.State, action: list[int]) -> env_api.Transition:
        # We assume any truncation based on maximum length has already been
        # applied to the input; double-check anyway.
        assert len(action) <= self.max_action_len(state)

        state.step_id += 1

        action_str = self.tokenizer.decode(self.tokenizer.cut_at_stop_tokens(action))
        state.episode.past_actions.append(action_str)

        # Insert <\think> if it's not presented in action and action doesn't end with eot
        if self._stop_think_condition(state, action, action_str):
            return self._stop_think_transition(state, action, action_str)

        # We assume here the answer is split into multiple actions
        match state.episode.prediction_type:
            case "full":
                reasoning, answer = self._extract_reasoning_and_answer(action_str)
            case "answer_only":
                reasoning = state.episode.past_actions[state.step_id - 1]
                answer = self._extract_answer(action_str)

        outcomes, info = self._eval_and_compute_metrics(
            state, action, reasoning, answer
        )

        # Check if we're done
        if state.step_id >= self.max_attempts or outcomes["pass"]:
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes,
                info=info,
            )

        feedback = "Your answer is incorrect. Please try again."
        return self.transition(
            messages=[
                self.message_cls.user(feedback),
                self.message_cls.assistant(),
            ],
            action=action,
            action_str=action_str,
            outcomes=outcomes,
            info=info,
            think=False,
        )


@dataclass
class MathThinkCWMDialogState(dialog.DialogState):
    n_step_tokens: int = 0


class MathThinkCWMDialogEnv(MathThinkDialogEnv):
    """
    Use Qwen prompt format https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py#L134-L140 for math reasoning.
    If `think` is set to True, it will add "self.tokenizer.THINKING_START_ID" to the observation, and requires the response to be in the format:
    "reasoning + self.tokenizer.THINKING_END_ID + \\boxed{answer}".
    Otherwise, it only captures the answer in the "\\boxed{answer}" format.

    stop_think option: Insert the thinking_end_id token to the action, so that the model can stop generating the answer.
    """

    prompt_template_cot = "{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$."
    prompt_template_direct = "{problem}\n\nWrap your answer in $\\boxed{{}}$."

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_attempts: int = 3,
        max_prompt_len: int = 2048,
        max_action_len: int = 1024,
        answer_only_action_len: int = 512,
        stop_str: str | None = None,
        think: bool = False,
        stop_think: bool = False,
        use_think_tag: bool = False,
        strict_parsing: bool = True,
        use_direct_prompt: bool = False,
        tools: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_attempts=max_attempts,
            max_prompt_len=max_prompt_len,
            max_action_len=max_action_len,
            answer_only_action_len=answer_only_action_len,
            stop_str=stop_str,
            stop_think=stop_think,
        )
        self.think = think
        self.use_think_tag = self.think and use_think_tag
        self.strict_parsing = strict_parsing
        self.tools = tools

        if use_direct_prompt:
            self.prompt_template = self.prompt_template_direct
        else:
            self.prompt_template = self.prompt_template_cot

        if self.stop_think:
            assert (
                self.think and not self.use_think_tag
            ), "The stop thinking option relies on the THINKING_END_ID token, you must set think = True"

    @property
    def use_think_token(self) -> bool:
        return self.think and not self.use_think_tag

    @property
    def assistant_prefix(self) -> str:
        if self.think and self.use_think_tag:
            return prompts.think_tag_prompt()
        return ""

    @property
    def thinking_end_text(self) -> str:
        if self.use_think_tag:
            return prompts.THINK_TAG_END
        return self.tokenizer.THINKING_END_ID

    def _create_tool_executor(self) -> ToolExecutor | None:
        """Create tool executor if tools are enabled."""
        if self.tools:
            return ToolExecutor(python_tool_limits(max_executions=4))
        return None

    def initial_messages(
        self, episode: MathThinkDialogEpisode, tool_executor: ToolExecutor | None
    ) -> list[MessageBase]:
        system_prompt: str = ""

        prompt = self.prompt_template.format(problem=episode.problem)
        messages = list[MessageBase]()
        # Only add the system prompt for thinktag envs for backward compatibility
        if self.use_think_tag:
            system_prompt = prompts.get_cwm_sys_prompt(
                think=self.think, use_think_tag=self.use_think_tag
            )

        if tool_executor is not None:
            messages.append(
                tool_executor.system_prompt(
                    system_prompt + "\n\nYou may invoke tools during reasoning."
                )
            )
        elif system_prompt:
            messages.append(self.message_cls.system(system_prompt))

        if isinstance(episode, MathThinkDialogCWMEpisode):
            multiturn_dialogs = episode.multiturn_dialogs or []
        else:
            multiturn_dialogs = []
        for user, assistant in multiturn_dialogs:
            messages.append(self.message_cls.user(user))
            messages.append(self.message_cls.assistant(assistant))
        messages.extend(
            [
                self.maybe_truncate(
                    self.message_cls.user(prompt), max_tokens=self.max_prompt_len
                ),
                self.message_cls.assistant(self.assistant_prefix),
            ]
        )
        return messages

    def max_action_len(self, state: env_api.State) -> int:
        budget = self._max_action_len - state.n_step_tokens
        if self.stop_think:
            match state.episode.prediction_type:
                case "full":
                    # the -2 is reserved to account for the "...<|reasoning_thinking_end|>"
                    return budget - self.answer_only_action_len - 2
                case "answer_only":
                    return self.answer_only_action_len
            raise ValueError(
                f"Unknown prediction_type: {state.episode.prediction_type}"
            )
        else:
            return budget

    def start(
        self, episode_args: dict | None = None
    ) -> tuple[env_api.State, env_api.Transition]:
        assert episode_args is not None, "episode_args must be provided."
        episode = MathThinkDialogCWMEpisode(
            task_id=episode_args.get("task_id", "N/A"),
            problem=episode_args["problem"],
            answer=episode_args["answer"],
            past_actions=[],
            prediction_type="full",
            multiturn_dialogs=episode_args.get("multiturn_dialogs", None),
        )
        tool_executor = self._create_tool_executor()

        return (
            MathThinkCWMDialogState(
                step_id=0, episode=episode, tool_executor=tool_executor
            ),
            self.transition(
                initial=True,
                messages=self.initial_messages(episode, tool_executor),
                think=self.use_think_token,
            ),
        )

    def _extract_reasoning_and_answer(self, text: str) -> tuple[str | None, str | None]:
        opt = re2.Options()
        opt.dot_nl = True

        reasoning, answer = None, None

        if self.think:
            pattern = "(.*?)" + re2.escape(self.thinking_end_text) + "(.*)"
            if match := re2.search(pattern, text, opt):
                reasoning = match.group(1).strip()
                candidate_answer = match.group(2).strip()
            else:
                return None, None
        else:
            candidate_answer = text

        # Format check: answer section should not start with format string from
        # system prompt
        if self.use_think_tag and candidate_answer.strip().startswith(
            "[your external response]"
        ):
            return None, None

        pattern = (
            r".*?\$(\\boxed{.*?})\$\.?$"
            if self.strict_parsing
            else r".*?\$(\\boxed{.*?})\$"
        )

        candidate_answers = re2.findall(pattern, candidate_answer, opt)
        if len(candidate_answers) == 1:  #  or len(candidate_answers) > 1:
            answer = candidate_answers[0].strip()
            return reasoning, answer

        return None, None

    def _extract_answer(self, text: str) -> str | None:
        pattern = (
            r"\$(\\boxed{.*?})\$\.?$" if self.strict_parsing else r"\$(\\boxed{.*?})\$"
        )
        opt = re2.Options()
        opt.dot_nl = True
        if match := re2.search(pattern, text, opt):
            answer = match.group(1).strip()
            return answer
        return None

    def _stop_think_condition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> bool:
        return (
            self.stop_think
            and action[-1] != self.tokenizer.eot_id
            and self.tokenizer.thinking_end_id not in action
            and state.episode.prediction_type == "full"
        )

    def _stop_think_transition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> env_api.Transition:
        state.episode.prediction_type = "answer_only"

        # Now add back the thinking_end_id
        observation = self.tokenizer.encode("...", bos=False, eos=False) + [
            self.tokenizer.thinking_end_id
        ]

        observation_str = self.tokenizer.decode(observation, cut_at_stop_tokens=False)

        return env_api.Transition(
            action=action,
            action_str=action_str,
            observation=observation,
            observation_str=observation_str,
            terminal=False,
            outcomes=self.default_outcomes | {"action_len": len(action)},
        )

    def _tool_execution_transition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> env_api.Transition | None:
        """Handle tool execution if applicable. Returns transition if tools were executed, None otherwise."""
        if state.tool_executor is None:
            return None

        tool_feedback = state.tool_executor.maybe_exec_tools(action_str)
        if not tool_feedback:
            return None

        tool_feedback = [
            self.maybe_truncate(
                # XXX hard-coded feedback length
                feedback,
                max_tokens=1024,
                where="left",
            )
            for feedback in tool_feedback
        ]

        # Thinking section ended? Prompt again with thinking prefix
        gen_prefix = ""
        think = False
        if action_str.count(self.thinking_end_text) > 0:
            gen_prefix = self.assistant_prefix
            think = self.use_think_token

        tr = self.transition(
            messages=tool_feedback + [self.message_cls.assistant(gen_prefix)],
            action=action,
            action_str=action_str,
            outcomes=self.default_outcomes,
            think=think,
        )
        state.n_step_tokens += len(tr.observation)
        return tr

    def step(self, state: env_api.State, action: list[int]) -> env_api.Transition:
        # We assume any truncation based on maximum length has already been
        # applied to the input; double-check anyway.
        assert len(action) <= self.max_action_len(state)

        action_str = self.tokenizer.decode(self.tokenizer.cut_at_stop_tokens(action))
        state.episode.past_actions.append(action_str)
        state.n_step_tokens += len(action)

        # Handle tool execution
        tool_transition = self._tool_execution_transition(state, action, action_str)
        if tool_transition:
            return tool_transition

        state.step_id += 1

        if self.use_think_tag and action_str.count(self.thinking_end_text) != 1:
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=self.default_outcomes | {"action_len": state.n_step_tokens},
                info=dict(error=f"Occurrence of {self.thinking_end_text} is not 1"),
            )

        # Insert the thinking_end_id if it's not presented in action and action doesn't end with eot
        if self._stop_think_condition(state, action, action_str):
            trans = self._stop_think_transition(state, action, action_str)
            trans.outcomes["action_len"] = state.n_step_tokens
            return trans

        # We assume here the answer is split into multiple actions
        match state.episode.prediction_type:
            case "full":
                reasoning, answer = self._extract_reasoning_and_answer(action_str)
            case "answer_only":
                reasoning = state.episode.past_actions[state.step_id - 1]
                answer = self._extract_answer(action_str)

        outcomes, info = self._eval_and_compute_metrics(
            state, action, reasoning, answer
        )
        outcomes["action_len"] = state.n_step_tokens
        info["action_len"] = state.n_step_tokens

        # Check if we're done
        if state.step_id >= self.max_attempts or outcomes["pass"]:
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes,
                info=info,
            )

        feedback = "Your answer is incorrect. Please try again."
        return self.transition(
            messages=[
                self.message_cls.user(feedback),
                self.message_cls.assistant(self.assistant_prefix),
            ],
            action=action,
            action_str=action_str,
            outcomes=outcomes,
            info=info,
            think=self.use_think_token,
        )


config.register_env(
    config.EnvConfig(
        name="math_cwmthinktag_env:64k",
        cls=MathThinkCWMDialogEnv,
        init_kwargs={
            "max_attempts": 1,
            "max_action_len": 65536,
            "think": True,
            "stop_think": False,
            "use_think_tag": True,
            "strict_parsing": False,
            "use_direct_prompt": True,
            "tools": False,
        },
    )
)
