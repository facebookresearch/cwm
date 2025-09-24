# Copyright (c) Meta Platforms, Inc. and affiliates.

import base64
import json
import pickle
import sys
import zlib
from dataclasses import dataclass
from typing import Any, Literal, cast

import re2

from cwm.exec.code.eval import ExecResult, ExecStatus
from cwm.exec.code.eval.python import exec_lcbcodegen_tests
from cwm.rl.envs import api as env_api
from cwm.rl.envs import config, dialog, prompts
from cwm.rl.envs import outcomes as outcomes_ut
from cwm.rl.envs.outcomes import thought_info
from cwm.rl.envs.tool_executor import ToolExecutor, python_tool
from cwm.rl.envs.utils import code as code_ut
from cwm.text.datatypes import MessageBase
from cwm.text.tokenizers import Tokenizer

# Prompts according to the LCB paper, https://arxiv.org/abs/2403.07974
OFFICIAL_SYSTEM_MESSAGE = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program."


@dataclass
class LCBEpisode:
    id: str
    title: str
    content: str
    date: str
    difficulty: str

    starter_code: str
    public_test_cases: list[dict[str, str]]
    private_test_cases: list[dict[str, str]]

    metadata: dict


@dataclass
class LCBCWMEpisode(LCBEpisode):
    prediction_type: str
    past_actions: list[str]


def _official_question_template_answer(lcb_episode: LCBEpisode):
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."

    question_content = lcb_episode.content
    starter_code = lcb_episode.starter_code
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code != "":
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt


def _ours_question_template_answer(
    format_guide: str, outline_guide: str, lcb_episode: LCBEpisode
):
    question_content = lcb_episode.content
    starter_code = lcb_episode.starter_code

    prompt = (
        f"Provide a Python solution for the following competitive programming question: {question_content}.\n"
        + outline_guide
        + format_guide
    )
    if starter_code != "":
        prompt += (
            f" Use the provided function signature:\n```python\n{starter_code}\n```"
        )
    else:
        prompt += " Your code should read from and write to standard io."
    return prompt


class LCBCodeGenEnv(dialog.DialogEnv):
    """
    Single- or multi-turn LiveCodeBench code-generation env.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt_format: Literal["official", "ours", "tools"],
        max_attempts: int = 1,
        feedback_format: str | None = None,
        max_action_len: int = 4096,
        think: bool = False,
        use_think_tag: bool = False,
    ) -> None:
        assert not (think and not use_think_tag) or hasattr(
            tokenizer, "thinking_start_id"
        )
        super().__init__(tokenizer)

        if prompt_format == "official" and feedback_format is not None:
            raise ValueError(f"{prompt_format=} incompatible with {feedback_format=}")

        self.max_prompt_len = 2048
        self.max_feedback_len = 1024
        self._max_action_len = max_action_len
        self.max_attempts = max_attempts

        self.think = think
        self.use_think_tag = use_think_tag
        self.prompt_format = prompt_format
        self.feedback_format = feedback_format
        self.retry_guide = "Give it another try.\n"
        self.outline_guide = ""

        self.succeeded_test_format: str = (
            "- input `{test_input}` passed with expected output `{test_output}`"
        )
        self.failed_test_format: str = "- input `{test}` failed:\n{info}\n"
        self.parsing_error_feedback = (
            "Could not parse your code.\n" + self.retry_guide + self.format_guide
        )
        self.feedback_format_code = (
            "Your code failed the following tests:\n\n{failed_tests}\n"
            + self.retry_guide
            + self.format_guide
        )
        self.all_tests_feedback_format_code = (
            "Your code passed the following tests:\n\n{succeeded_tests}\n\n"
            + "Your code failed the following tests:\n\n{failed_tests}\n\n"
            + self.retry_guide
            + self.format_guide
        )
        self.no_execution_feedback_code = (
            "This solution was wrong.\n" + self.retry_guide + self.format_guide
        )
        self.no_exec_feedback = (
            "Consider if the previous solution is correct and provide a new one if it is not.\n"
            + self.format_guide
        )
        self.parsing_feedback_code = (
            "Could not parse your code.\n" + self.retry_guide + self.format_guide
        )

    @property
    def use_think_token(self) -> bool:
        return self.think and not self.use_think_tag

    @property
    def format_guide(self) -> str:
        return "Your code should be enclosed in triple backticks like so: ```python YOUR CODE HERE ```. Use the backticks for your code only."

    def max_action_len(self, state: env_api.State) -> int:
        return self._max_action_len

    def _create_episode(self, episode_args: dict) -> LCBEpisode:
        """Create a LCBEpisode from episode arguments."""
        try:
            private_test_cases = json.loads(episode_args["private_test_cases"])
        except json.JSONDecodeError:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(episode_args["private_test_cases"])
                    )
                )
            )

        return LCBEpisode(
            id=episode_args["question_id"],
            title=episode_args["question_title"],
            content=episode_args["question_content"],
            date=episode_args["contest_date"],
            difficulty=episode_args["difficulty"],
            starter_code=episode_args["starter_code"],
            public_test_cases=json.loads(episode_args["public_test_cases"]),
            private_test_cases=private_test_cases,
            metadata=json.loads(episode_args["metadata"]),
        )

    def _validate_tests(self, episode: LCBEpisode) -> None:
        """Validate that episode has required test cases."""
        assert (
            len(episode.public_test_cases) >= 1
        ), "Public test cases should be provided."

    def _create_tool_executor(self) -> ToolExecutor | None:
        """Create tool executor if tools are enabled."""
        if self.prompt_format == "tools":
            return ToolExecutor(python_tool())
        return None

    def initial_prompt_messages(
        self, episode: LCBEpisode, tool_executor: ToolExecutor | None = None
    ) -> list[MessageBase]:
        messages: list[MessageBase] = []

        match self.prompt_format:
            case "official":
                messages.append(self.message_cls.system(OFFICIAL_SYSTEM_MESSAGE))
                prompt = _official_question_template_answer(lcb_episode=episode)
            case "ours":
                prompt = _ours_question_template_answer(
                    self.format_guide, self.outline_guide, lcb_episode=episode
                )
                if self.use_think_tag:
                    system_prompt = prompts.get_cwm_sys_prompt(
                        think=self.think,
                        use_think_tag=self.use_think_tag,
                    )
                    messages.append(self.message_cls.system(system_prompt))
            case "tools":
                assert tool_executor is not None
                messages.append(
                    tool_executor.system_prompt(
                        'You are an expert Python programmer and tool user. Before attempting to solve the full problem, you may verify PARTS of your solution with the "python" tool.'
                    )
                )
                prompt = _ours_question_template_answer(
                    self.format_guide, self.outline_guide, lcb_episode=episode
                )

        messages += [
            self.maybe_truncate(
                self.message_cls.user(prompt),
                max_tokens=self.max_prompt_len,
                where="left",
            ),
            self.message_cls.assistant(self.assistant_prefix),
        ]

        return messages

    def start(
        self, episode_args: dict | None = None
    ) -> tuple[env_api.State, env_api.Transition]:
        assert episode_args is not None

        episode = self._create_episode(episode_args)
        self._validate_tests(episode)
        tool_executor = self._create_tool_executor()
        prompt_messages = self.initial_prompt_messages(
            episode=episode, tool_executor=tool_executor
        )

        return (
            dialog.DialogState(step_id=0, episode=episode, tool_executor=tool_executor),
            self.transition(
                initial=True,
                messages=prompt_messages,
                think=self.use_think_token,
            ),
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

        return self.transition(
            messages=tool_feedback + [self.message_cls.assistant()],
            action=action,
            action_str=action_str,
        )

    def _parsing_error_transition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> env_api.Transition:
        """Handle code parsing errors and return a transition."""
        outcomes = code_ut.failed_code_exec_outcomes
        terminal = state.step_id >= self.max_attempts
        return self.transition(
            terminal=terminal,
            action=action,
            action_str=action_str,
            messages=[
                self.maybe_truncate(
                    self.message_cls.user(self.parsing_error_feedback),
                    max_tokens=self.max_feedback_len,
                    where="right",
                ),
                self.message_cls.assistant(),
            ]
            if not terminal
            else None,
            outcomes=outcomes,
            think=self.use_think_token,
        )

    def _exec_code_and_outcomes(
        self, state: env_api.State, code: str
    ) -> tuple[list[ExecResult], dict[str, Any]]:
        """Execute code and return results with outcomes."""
        io_sample = {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t["input"]
                        for t in state.episode.public_test_cases
                        + state.episode.private_test_cases
                    ],
                    "outputs": [
                        t["output"]
                        for t in state.episode.public_test_cases
                        + state.episode.private_test_cases
                    ],
                    "fn_name": state.episode.metadata.get("func_name", None),
                }
            ),
        }

        # Evaluate on all tests with early stopping; if public tests are
        # passing we'll have a terminal state, otherwise we won't check private
        # tests anyway.
        # Some samples don't have public tests and we'll not do multiple steps
        # for these.
        results = exec_lcbcodegen_tests(
            code=code,
            sample=io_sample,
            early_stopping=True,
        )
        status = [r.status for r in results]
        public_results = results[: len(state.episode.public_test_cases)]
        public_status = [r.status for r in public_results]

        outcomes = code_ut.code_exec_outcomes(status, public_status)

        return results, outcomes

    def _terminal_condition(self, state: env_api.State, outcomes: dict) -> bool:
        """Check if episode should terminate."""
        return state.step_id >= self.max_attempts or outcomes["public_pass"]

    def _extract_reasoning(self, text: str) -> str | None:
        pattern = "(.*?)" + re2.escape(self.thinking_end_text)
        opt = re2.Options()
        opt.dot_nl = True
        if match := re2.search(pattern, text, opt):
            reasoning = match.group(1).strip()
            return reasoning
        return None

    @property
    def assistant_prefix(self) -> str:
        if self.think and self.use_think_tag:
            return prompts.THINK_TAG_START
        return ""

    @property
    def thinking_end_text(self) -> str:
        if self.use_think_tag:
            return prompts.THINK_TAG_END
        return self.tokenizer.THINKING_END_ID

    def step(self, state: env_api.State, action: list[int]) -> env_api.Transition:
        assert isinstance(state, dialog.DialogState)
        state = cast(dialog.DialogState, state)
        assert len(action) <= self.max_action_len(state)

        action_str = self.tokenizer.decode(self.tokenizer.cut_at_stop_tokens(action))

        tool_transition = self._tool_execution_transition(state, action, action_str)
        if tool_transition:
            return tool_transition

        state.step_id += 1

        reasoning_str = self._extract_reasoning(action_str) if self.think else None
        n_reasoning_tokens = (
            len(self.tokenizer.encode(reasoning_str)) if reasoning_str else 0
        )
        outcomes: dict = outcomes_ut.reasoning_found(
            reasoning_str is not None
        ) | outcomes_ut.n_reasoning_tokens(n_reasoning_tokens)

        code = code_ut.extract_first_code(action_str, language="python")
        if code is None:  # parsing error
            tr = self._parsing_error_transition(state, action, action_str)
            tr.outcomes |= outcomes
            return tr

        results, exec_outcomes = self._exec_code_and_outcomes(state, code)
        outcomes |= exec_outcomes

        if self._terminal_condition(state, outcomes):
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes,
            )

        # Re-prompt with exec feedback
        public_results = results[: len(state.episode.public_test_cases)]
        feedback = self.code_exec_feedback(state.episode, public_results)
        feedback_message = self.maybe_truncate(
            self.message_cls.user(feedback),
            max_tokens=self.max_feedback_len,
            where="right",
        )

        return self.transition(
            messages=[
                feedback_message,
                self.message_cls.assistant(self.assistant_prefix),
            ],
            action=action,
            action_str=action_str,
            outcomes=outcomes,
            think=self.use_think_token,
        )

    def code_exec_feedback(
        self,
        episode: LCBEpisode,
        public_results: list[ExecResult],
    ) -> str:
        public_test_cases = episode.public_test_cases
        assert len(public_results) <= len(public_test_cases)

        succeeded_tests = ""
        failed_tests = ""
        for test, result in zip(public_test_cases, public_results, strict=False):
            match result.status:
                case (
                    ExecStatus.EXCEPTION
                    | ExecStatus.SYNTAX_ERROR
                    | ExecStatus.FAILURE
                ):
                    failed_tests += self.failed_test_format.format(
                        test=test["input"],
                        info=code_ut.shorten_exec_info(
                            _shorten_backtrace_filenames, result.info
                        ),
                    )
                case ExecStatus.TIMEOUT:
                    failed_tests += self.failed_test_format.format(
                        test=test["input"],
                        info="Execution took too long",
                    )
                case ExecStatus.SUCCESS:
                    succeeded_tests += self.succeeded_test_format.format(
                        test_input=test["input"],
                        test_output=test["output"],
                    )
                case _:
                    failed_tests += self.failed_test_format.format(
                        test=["input"], info="NO INFO"
                    )

        match self.feedback_format:
            case "no_exec_feedback":
                feedback = self.no_execution_feedback_code
            case "no_exec" | "no_exec2":
                feedback = self.no_exec_feedback
            case "all_tests" | "all_tests_code_only":
                if len(succeeded_tests) == 0:
                    feedback = self.feedback_format_code.format(
                        failed_tests=failed_tests
                    )
                else:
                    feedback = self.all_tests_feedback_format_code.format(
                        failed_tests=failed_tests, succeeded_tests=succeeded_tests
                    )
            case _:
                feedback = self.feedback_format_code.format(failed_tests=failed_tests)

        return feedback


class LCBCWMThinkEnv(LCBCodeGenEnv):
    answer_only_action_len = 1024
    think: bool = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt_format: Literal["official", "ours", "tools"],
        max_attempts: int = 1,
        max_action_len: int = 4096,
        stop_think: bool = False,
        use_think_tag: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            max_attempts=max_attempts,
            max_action_len=max_action_len,
            think=self.think,
            use_think_tag=self.think and use_think_tag,
        )
        self.stop_think = stop_think
        if self.stop_think:
            assert (
                not self.use_think_tag
            ), "stop_think is not compatible with use_think_tag"

    def max_action_len(self, state: env_api.State) -> int:
        match state.episode.prediction_type:
            case "full":
                return self._max_action_len - self.answer_only_action_len - 2
            case "answer_only":
                return self.answer_only_action_len
            case _:
                raise ValueError(
                    f"Unknown prediction type: {state.episode.prediction_type}"
                )

    def _create_episode(self, episode_args: dict) -> LCBCWMEpisode:
        """Create a LCBCWMEpisode from episode arguments."""
        base_episode = super()._create_episode(episode_args)

        return LCBCWMEpisode(
            id=base_episode.id,
            title=base_episode.title,
            content=base_episode.content,
            date=base_episode.date,
            difficulty=base_episode.difficulty,
            starter_code=base_episode.starter_code,
            public_test_cases=base_episode.public_test_cases,
            private_test_cases=base_episode.private_test_cases,
            metadata=base_episode.metadata,
            prediction_type="full",
            past_actions=[],
        )

    def _extract_reasoning_and_answer(self, text: str) -> tuple[str | None, str | None]:
        # extract reasoning before and after
        pattern = "(.*?)" + re2.escape(self.thinking_end_text) + "(.*?$)"
        opt = re2.Options()
        opt.dot_nl = True
        if match := re2.search(pattern, text, opt):
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

        return None, None

    def _stop_think_condition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> bool:
        """Check if we should stop thinking and switch to answer-only mode."""
        return (
            self.stop_think
            and self.tokenizer.thinking_end_id not in action
            and state.episode.prediction_type == "full"
            and action[-1] not in self.tokenizer.stop_tokens
        )

    def _stop_think_transition(
        self, state: env_api.State, action: list[int], action_str: str
    ) -> env_api.Transition:
        state.episode.prediction_type = "answer_only"
        observation = self.tokenizer.encode("...", bos=False, eos=False) + [
            self.tokenizer.thinking_end_id
        ]
        observation_str = self.tokenizer.decode(observation, cut_at_stop_tokens=False)
        outcomes: dict = (
            outcomes_ut.reasoning_found(False)
            | outcomes_ut.n_reasoning_tokens(0)
            | code_ut.failed_code_exec_outcomes
        )

        return env_api.Transition(
            action=action,
            action_str=action_str,
            observation=observation,
            observation_str=observation_str,
            terminal=False,
            outcomes=outcomes,
        )

    def step(self, state: env_api.State, action: list[int]) -> env_api.Transition:
        assert isinstance(state, dialog.DialogState)
        state = cast(dialog.DialogState, state)
        assert len(action) <= self.max_action_len(state)

        state.step_id += 1
        action_str = self.tokenizer.decode(self.tokenizer.cut_at_stop_tokens(action))
        state.episode.past_actions.append(action_str)

        if self._stop_think_condition(state, action, action_str):
            return self._stop_think_transition(state, action, action_str)

        match state.episode.prediction_type:
            case "full":
                reasoning, answer = self._extract_reasoning_and_answer(action_str)
            case "answer_only":
                reasoning, answer = (
                    state.episode.past_actions[state.step_id - 1],
                    action_str,
                )
            case _:
                raise ValueError(
                    f"Unknown prediction type: {state.episode.prediction_type}"
                )

        th_info = thought_info(reasoning, self.tokenizer)

        code = code_ut.extract_single_code(answer or "", language="python")

        info = {
            "reasoning": reasoning,
            "answer": answer,
            "answer_extracted": code,
            **th_info,
        }

        n_reasoning_tokens = len(self.tokenizer.encode(reasoning)) if reasoning else 0
        outcomes: dict = outcomes_ut.reasoning_found(
            reasoning is not None
        ) | outcomes_ut.n_reasoning_tokens(n_reasoning_tokens)

        if code is None:
            tr = self._parsing_error_transition(state, action, action_str)
            tr.outcomes |= outcomes
            return tr

        exec_results, exec_outcomes = self._exec_code_and_outcomes(state, code)
        outcomes |= exec_outcomes

        if self._terminal_condition(state, outcomes):
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes,
                info=info,
            )

        # Re-prompt with exec feedback
        public_results = exec_results[: len(state.episode.public_test_cases)]
        feedback = self.code_exec_feedback(state.episode, public_results)
        feedback_message = self.maybe_truncate(
            self.message_cls.user(feedback),
            max_tokens=self.max_feedback_len,
            where="right",
        )

        return self.transition(
            messages=[
                feedback_message,
                self.message_cls.assistant(self.assistant_prefix),
            ],
            action=action,
            action_str=action_str,
            outcomes=outcomes,
            think=self.use_think_token,
        )


def _shorten_backtrace_filenames(backtrace: str) -> str:
    # Truncate backtrace filename for both brevity and portability
    # Remove frame of test runner
    frames = backtrace.splitlines()
    i = 0
    while i < len(frames) - 1:
        if re2.match(r" *File \".*/python_lcbcodegen.py\".*", frames[i]):
            frames.pop(i)
            frames.pop(i)
            if i < len(frames) and set(frames[i]) == {" ", "^"}:
                # python 3.11 error markers
                frames.pop(i)
        else:
            i += 1
    backtrace = "\n".join(frames)
    # Actual file becomes "solution.py"
    backtrace = re2.sub(r"File \"/tmp/.*/code.py\"", 'File "solution.py"', backtrace)
    backtrace = re2.sub(r"File \"<source>\"", 'File "solution.py"', backtrace)
    # Truncate path to python site-packages
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    backtrace = re2.sub(f'File ".*/lib/{py_ver}', 'File "...', backtrace)

    return backtrace


config.register_env(
    config.EnvConfig(
        name="lcb_codegen_cwmthinktag:64k",
        cls=LCBCWMThinkEnv,
        init_kwargs={
            "prompt_format": "ours",
            "max_attempts": 1,
            "max_action_len": 65536,
            "stop_think": False,
            "use_think_tag": True,
        },
    )
)
