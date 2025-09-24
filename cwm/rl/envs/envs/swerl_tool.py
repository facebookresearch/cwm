# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import TypedDict, cast

import cwm.rl.swerl as swerl
from cwm.rl.envs import api as env_api
from cwm.rl.envs import config, dialog, outcomes, prompts, rewards
from cwm.rl.swerl.default_backends import get_default_modal_backend
from cwm.rl.swerl.default_tools import parse_tool_calls
from cwm.rl.swerl.eval_backend.eval import EvalResult, eval_instance_default
from cwm.rl.swerl.modal_backend import ModalBackend
from cwm.text.datatypes import CWMChatMessage
from cwm.text.tokenizers import CWMInstructTokenizer, Tokenizer

SYSTEM_UBUNTU = """
You are working in a linux container. The container session is persistent but will restart when encountering errors or after a {session_timeout} session timeout in each turn. You are also constrained by a {context_size} total context window limit. You can interact with the container through the following tool interface for at most {max_turns} turns. Here are the specifications:

<tool: bash>
[command(s)]
</tool>
Executes bash command(s) [command(s)] in the current session. [command(s)] can be any non-interactive bash command(s), either single or multi-line.

<tool: edit>
[path]
<<<<<<< SEARCH
[search_lines]
=======
[replacement_lines]
>>>>>>> REPLACE
</tool>
Replaces [search_lines] from [path] with [replacement_lines], where [path] must exist and [search_lines] must uniquely and exactly match one or more consecutive lines from the original file, including indentations and whitespaces.

<tool: create>
[path]
[content]
</tool>
Creates a new file at [path] with [content], where [path] must not exist, but its parent directory must exist.

<tool: submit>
[result]
</tool>
Marks [result] as your final submission. The meaning and format of [result] depends on the specific user request.

REQUIREMENTS:
* You cannot invoke more than one tool in each step.
* You must invoke exactly one tool in each step before any <tool: submit> call.
* After successful submission, provide a final response if the request is resolved, otherwise continue iterating. You cannot provide a final response before invoking <tool: submit>.
""".strip()

SYSTEM_UBUNTU_NO_PLUGINS = """
You are working in a linux container. The container session is persistent but will restart when encountering errors or after a {session_timeout} session timeout in each turn. You are also constrained by a {context_size} total context window limit. You can interact with the container through the following tool interface for at most {max_turns} turns. Here are the specifications:

<tool: bash>
[command(s)]
</tool>
Executes bash command(s) [command(s)] in the current session. [command(s)] can be any non-interactive bash command(s), either single or multi-line.

<tool: submit>
[result]
</tool>
Marks [result] as your final submission. The meaning and format of [result] depends on the specific user request.

REQUIREMENTS:
* You cannot invoke more than one tool in each step.
* You must invoke exactly one tool in each step before any <tool: submit> call.
* After successful submission, provide a final response if the request is resolved, otherwise continue iterating. You cannot provide a final response before invoking <tool: submit>.
""".strip()

USER_ISSUE_SOLVING = """
Solve the following issue by implementing the necessary code changes and submitting a patch file:

<issue_description>
{issue}
</issue_description>

The [result] argument of <tool: submit> should be the path to a patch file that resolves the issue. This file must be accessible from the current working directory and should contain the end-to-end code changes in git diff format. You can refine and submit your patch multiple times as needed to ensure correctness.

Once you've submitted at least once and are confident in your solution, provide a final response summarizing your work using the following markdown template as a guide. Feel free to adjust the sections as necessary to fit the specific issue and solution:

```markdown
## Issue Summary

Brief description of the issue (bug/feature/enhancement), affected component, and expected outcome.

## Investigation and Reproduction

Steps taken to understand and reproduce the issue: commands executed, test cases run, error messages, debugging approach, and root cause analysis. For features, describe requirement analysis and codebase exploration.

## Solution and Code Changes

High-level approach and rationale for the chosen solution. Specific modifications by file/component, including new functions, classes, or architectural decisions. Explain how changes work together if multiple files are involved.

## Testing and Verification

Description of how you ensure the patch fully resolves the issue, including new tests added (or test changes made) to verify correctness, how you ran existing tests to prevent regressions, and how comprehensive your testing process is (e.g., what edge cases you covered, how you validated correctness across different scenarios).
```

Your primary objective is to ensure patch correctness above all else - thoroughly explore the codebase, think hard, and leverage significant execution to verify correctness by writing comprehensive tests to validate your solution and running existing tests to prevent regressions. Only submit when you are genuinely confident in your patch's correctness.

I've uploaded the corresponding code repository at {repo_root} and installed all the necessary dependencies. Now, the bash session has started, with the current working directory set to the repo root.
""".strip()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class SWERLConfig:
    context_size: int = 131072
    session_timeout: float = 120.0
    sandbox_timeout: int = 2400

    eval_root: str = ""

    max_prompt_len: int = 8192
    max_user_turn_len: int = 32768
    max_action_len: int = 65536
    max_turns: int = 128
    eval_timeout: float = 300.0

    # threshold for SWE-RL edit distance reward
    swerl_reward_threshold: float | None = None
    use_execution_reward: bool = True
    block_network: bool = False

    plugins: bool = True
    strict_format: bool = True
    think: bool = False
    test_patch_hint: bool = False

    use_think_tag: bool = False
    backend: str = "modal"  # 'modal' or other custom backends

    @property
    def use_think_token(self) -> bool:
        return self.think and not self.use_think_tag

    @property
    def assistant_prefix(self) -> str:
        if self.think and self.use_think_tag:
            return prompts.think_tag_prompt()
        return ""

    def thinking_end_text(self, tokenizer: CWMInstructTokenizer) -> str:
        if self.think and self.use_think_tag:
            return prompts.THINK_TAG_END
        return tokenizer.THINKING_END_ID

    @property
    def use_swerl_reward(self) -> bool:
        return self.swerl_reward_threshold is not None


@dataclass
class SWERLDialogEpisode:
    class Args(TypedDict):
        instance_id: str
        issue: str
        patch: str
        docker_url: str
        repo_root_path: str

    episode_args: Args
    tool_backend: swerl.tools.ToolBackend | None


@dataclass
class SWERLState(env_api.State):
    step_id: int
    episode: SWERLDialogEpisode
    cleaned_up: bool = False
    num_tokens: int = 0
    pred_patch: str | None = None

    def close(self) -> None:
        if self.cleaned_up:
            return
        try:
            if self.episode.tool_backend is not None:
                self.episode.tool_backend.destroy()
        # silent any cleanup exceptions
        except Exception:
            logger.exception("Unexpected cleanup exception")
        finally:
            self.cleaned_up = True


# 100KB
MAX_MESSAGE_LEN = 102400


def eval_instance(
    eval_root: str,
    eval_timeout: float,
    episode: SWERLDialogEpisode,
    pred_patch: str,
    backend: str = "modal",
) -> EvalResult:
    assert episode.tool_backend is not None

    if pred_patch.strip() == "":
        return EvalResult(outcome="fail", message="Empty patch")

    # Some optimizations
    if "diff" not in pred_patch:
        return EvalResult(outcome="fail", message="'diff' not in patch")

    if not eval_root:
        tmpdir = os.getenv("TMPDIR", tempfile.gettempdir())
        eval_root = os.path.join(tmpdir, "swerl-evals")
        logger.info(f"Setting eval_root to {eval_root}")

    assert isinstance(episode.tool_backend, ModalBackend)
    eval_dir = os.path.join(eval_root, str(uuid.uuid4()))
    workdir = episode.episode_args["repo_root_path"]
    eval_result = eval_instance_default(
        spec=episode.episode_args,  # type: ignore
        pred_patch=pred_patch,
        eval_dir=eval_dir,
        timeout=eval_timeout,
        workdir=workdir,
        rm_dir_after_eval=True,
        backend=backend,
    )

    return eval_result


BUDGET_NOTICE_TEMPLATE = """
<budget>
* Remaining turns: {remaining_turns}
* Remaining tokens: {remaining_tokens}
</budget>
""".strip()

SUBMISSION_FEEDBACK_PROMPT = """
The following patch content is marked as your final submission:

<submission>
{pred_patch}
</submission>

Review the patch content and ensure it correctly resolves the issue. If necessary, you can continue refining the patch and submit an updated version.
""".strip()

HINT_PROMPT = """
Here is a test patch that demonstrates the expected behavior. Your submitted patch needs to pass these tests to be considered correct:

```diff
{test_patch}
```
""".strip()

FEEDBACK_OUTPUT_PROMPT = """
<output>
{feedback}
</output>
""".strip()


def budget_notice(
    remaining_turns: int,
    remaining_tokens: int,
) -> str:
    notice = BUDGET_NOTICE_TEMPLATE.format(
        remaining_turns=remaining_turns,
        remaining_tokens=remaining_tokens,
    )
    return notice


class SWERLToolEnv(dialog.DialogEnv):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: SWERLConfig,
    ) -> None:
        self.config = config
        super().__init__(tokenizer)
        assert self.config.use_execution_reward or self.config.use_swerl_reward

    def max_action_len(self, state: env_api.State) -> int:
        return self.config.max_action_len

    @staticmethod
    def make_info_dict(
        episode_args: SWERLDialogEpisode.Args,
        message: str = "",
        eval_duration: float = -1.0,
        pred_patch: str | None = None,
    ) -> dict:
        return dict(
            source=episode_args.get("source", "N/A"),
            instance_id=episode_args.get("instance_id", "N/A"),
            message=message,
            eval_duration=eval_duration,
            pred_patch=pred_patch,
        )

    def start(
        self, episode_args: dict | None = None
    ) -> tuple[env_api.State, env_api.Transition]:
        assert episode_args is not None, "episode_args must be provided."

        env_prompt_template = (
            SYSTEM_UBUNTU if self.config.plugins else SYSTEM_UBUNTU_NO_PLUGINS
        )
        env_prompt = env_prompt_template.format(
            context_size=self.config.context_size,
            max_turns=self.config.max_turns,
            session_timeout=f"{int(self.config.session_timeout)}-second",
        )
        sys_prompt = prompts.get_cwm_sys_prompt(
            think=self.config.think,
            use_think_tag=self.config.use_think_tag,
            environment=env_prompt,
        )

        # We do a lazy initialization
        tool_backend: swerl.tools.ToolBackend | None = None

        episode = SWERLDialogEpisode(
            episode_args=cast(SWERLDialogEpisode.Args, episode_args),
            tool_backend=tool_backend,
        )
        for key in SWERLDialogEpisode.Args.__annotations__.keys():
            assert key in episode_args, f"Missing key {key} in episode_args"

        issue = episode.episode_args["issue"]
        if self.config.test_patch_hint and "test_patch" in episode.episode_args:
            # Some hard problems / ambiguous issues may require a hint
            hint = HINT_PROMPT.format(test_patch=episode.episode_args["test_patch"])  # type: ignore
            issue = issue.rstrip() + "\n\n" + hint
        user_prompt = USER_ISSUE_SOLVING.format(
            issue=issue,
            repo_root=episode.episode_args["repo_root_path"],
        )

        # We force the use of CWMChatMessage for SWE-RL (tool)
        prompt_messages = [
            CWMChatMessage.system(sys_prompt),
            self.maybe_truncate(
                CWMChatMessage.user(user_prompt),
                max_tokens=self.config.max_prompt_len,
                where="left",
            ),
            CWMChatMessage.assistant(self.config.assistant_prefix),
        ]
        transition = self.transition(
            initial=True,
            messages=prompt_messages,
            info=self.make_info_dict(episode.episode_args),
            think=self.config.use_think_token,
        )
        return (
            SWERLState(
                step_id=0, episode=episode, num_tokens=len(transition.observation)
            ),
            transition,
        )

    def step(self, state: env_api.State, action: list[int]) -> env_api.Transition:
        start_time = time.perf_counter()
        state = cast(SWERLState, state)
        assert isinstance(state, SWERLState)
        # We assume any truncation based on maximum length has already been
        # applied to the input; double-check anyway.
        assert len(action) <= self.max_action_len(state)
        state.step_id += 1
        state.num_tokens += len(action)

        tokenizer: CWMInstructTokenizer = self.tokenizer  # type: ignore
        episode = state.episode
        action_str = tokenizer.decode(tokenizer.cut_at_stop_tokens(action))

        if state.num_tokens > self.config.context_size:
            info = self.make_info_dict(
                episode.episode_args,
                f"Context size exceeded: {state.num_tokens} > {self.config.context_size}",
            )
            state.close()
            outcomes_ = outcomes.failed()
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes_,
                info=info,
            )

        if state.step_id >= self.config.max_turns:
            info = self.make_info_dict(episode.episode_args, "Max turns reached")

            # Cleanup the backend before the return to maximally save costs
            state.close()
            outcomes_ = outcomes.failed()
            return self.transition(
                terminal=True,
                action=action,
                action_str=action_str,
                outcomes=outcomes_,
                info=info,
            )

        answer = action_str
        if self.config.think:
            thinking_end_text = self.config.thinking_end_text(tokenizer)
            num_ending_tokens = action_str.count(thinking_end_text)
            answer = action_str.split(thinking_end_text, 1)[-1]

        try:
            if self.config.think and num_ending_tokens != 1:
                raise swerl.errors.FormatError(
                    f"Expected exactly one thinking end token, got {num_ending_tokens}"
                )
            tool_name, tool_input = parse_tool_calls(answer)[0]

            if episode.tool_backend is None:
                episode_args = episode.episode_args
                startup_commands = swerl.common.restore_env(episode_args)  # type: ignore
                work_dir = episode_args["repo_root_path"]
                startup_commands += (
                    "\n"
                    + swerl.common.retain_only_current_branch_ancestor_commits(work_dir)
                )

                backend_args = {
                    "image_url": episode_args["docker_url"],
                    "session_timeout": self.config.session_timeout,
                    "work_dir": work_dir,
                    "startup_commands": startup_commands,
                    "server_python_path": swerl.common.get_server_python_path(
                        episode_args  # type: ignore
                    ),
                    "sandbox_timeout": self.config.sandbox_timeout,
                    "plugins": self.config.plugins,
                }

                if self.config.backend == "modal":
                    backend_args["block_network"] = self.config.block_network
                    episode.tool_backend = get_default_modal_backend(**backend_args)  # type: ignore
                else:
                    raise ValueError(
                        f"Unsupported backend: {self.config.backend}. Supported: 'modal'"
                    )
            tool_result = episode.tool_backend.apply_tool(tool_name, tool_input)
        except (
            swerl.errors.FormatError,
            swerl.errors.NoSuchToolError,
        ) as e:
            # Two possible situations:
            # 1. the assistant has finished the task and is presenting a summary
            if (
                state.pred_patch is not None
                and (not self.config.think or num_ending_tokens == 1)
                # If the format is strict, the markdown "##" is enforced
                and (not self.config.strict_format or "##" in answer)
            ):
                state.close()
                outcomes_, info = self.eval(episode, state.pred_patch)
                return self.transition(
                    terminal=True,
                    action=action,
                    action_str=action_str,
                    outcomes=outcomes_,
                    info=info,
                )
            # If strict, any format error causes a failure. Otherwise, the error is
            # treated as a user message.
            if self.config.strict_format:
                state.close()
                return self.transition(
                    terminal=True,
                    action=action,
                    action_str=action_str,
                    outcomes=outcomes.failed(),
                    info=self.make_info_dict(
                        episode.episode_args,
                        f"Tool call format error: {type(e).__name__}: {str(e)}",
                    ),
                )
            # 2. the assistant has made a format error
            # we return a "user" message instead of a tool result
            # Not returning `.tool_result` because it's a user-level constraint
            # that the agent should make one and only one tool call before its submission,
            # different from the tool calling semantics where the last dialog without
            # tool calls indicates the end of the turn.
            feedback = f"{type(e).__name__}: {str(e)}"
            feedback_message = self.maybe_truncate(
                CWMChatMessage.user(feedback),
                max_tokens=self.config.max_user_turn_len,
                where="right",
            )
            transition = self.transition(
                messages=[
                    feedback_message,
                    CWMChatMessage.assistant(self.config.assistant_prefix),
                ],
                action=action,
                action_str=action_str,
                outcomes=outcomes.failed(),
                info=dict(duration=time.perf_counter() - start_time),
                think=self.config.use_think_token,
            )
            state.num_tokens += len(transition.observation)
            return transition

        except Exception as e:
            state.close()
            # Retrieve modal backend logs for better debugging
            message = ""
            if isinstance(episode.tool_backend, ModalBackend):
                message = episode.tool_backend.get_debugging_info()
                # If it's a timeout, we return a failing transition
                if episode.tool_backend.timed_out():
                    info = self.make_info_dict(
                        episode.episode_args, f"Timeout: {message}"
                    )
                    return self.transition(
                        terminal=True,
                        action=action,
                        action_str=action_str,
                        outcomes=outcomes.failed(),
                        info=info,
                    )
            info = self.make_info_dict(episode.episode_args, message)
            raise RuntimeError(action_str, info) from e
        else:
            if tool_name == "submit" and tool_result.success:
                state.pred_patch = tool_result.output
                feedback = SUBMISSION_FEEDBACK_PROMPT.format(
                    pred_patch=state.pred_patch
                )
            else:
                feedback = tool_result.output

        # Budget estimation
        # * how many tokens are left
        # * how much time is left
        # * how many turns are left
        feedback = FEEDBACK_OUTPUT_PROMPT.format(feedback=feedback)
        num_estimated_feedback_tokens = len(
            self.tokenizer.encode_message(
                CWMChatMessage.tool_result(
                    feedback + "\n\n" + BUDGET_NOTICE_TEMPLATE, tool=tool_name
                )
            )
        )
        num_estimated_feedback_tokens = min(
            num_estimated_feedback_tokens, self.config.max_user_turn_len
        )
        remaining_tokens = (
            self.config.context_size - state.num_tokens - num_estimated_feedback_tokens
        )
        remaining_turns = self.config.max_turns - state.step_id
        budget_notice_prompt = budget_notice(
            remaining_turns=remaining_turns,
            remaining_tokens=remaining_tokens,
        )
        feedback = f"{feedback}\n\n{budget_notice_prompt}"
        feedback_message = self.maybe_truncate(
            CWMChatMessage.tool_result(feedback, tool=tool_name),
            max_tokens=self.config.max_user_turn_len,
            where="right",
        )
        # No intermediate reward
        transition = self.transition(
            messages=[
                feedback_message,
                CWMChatMessage.assistant(self.config.assistant_prefix),
            ],
            action=action,
            action_str=action_str,
            outcomes=outcomes.failed(),
            info=dict(duration=time.perf_counter() - start_time),
            think=self.config.use_think_token,
        )
        state.num_tokens += len(transition.observation)
        return transition

    def eval(self, episode: SWERLDialogEpisode, pred_patch: str) -> tuple[dict, dict]:
        # Evaluation
        if self.config.use_execution_reward and not self.config.use_swerl_reward:
            outcomes_, info = self.get_execution_outcomes(episode, pred_patch)
        elif self.config.use_swerl_reward and not self.config.use_execution_reward:
            assert self.config.swerl_reward_threshold is not None
            outcomes_, info = self.get_swerl_outcomes(
                self.config.swerl_reward_threshold, episode, pred_patch
            )
        elif self.config.use_execution_reward and self.config.use_swerl_reward:
            assert self.config.swerl_reward_threshold is not None
            # Use the execution reward
            outcomes_, info = self.get_execution_outcomes(episode, pred_patch)
            # Add the swerl reward info
            if not outcomes.successful_pass(outcomes_):
                swerl_outcomes, swerl_info = self.get_swerl_outcomes(
                    self.config.swerl_reward_threshold, episode, pred_patch
                )
                outcomes_ |= swerl_outcomes
        else:
            raise AssertionError
        return outcomes_, info

    def get_execution_outcomes(
        self, episode: SWERLDialogEpisode, pred_patch: str
    ) -> tuple[dict, dict]:
        # Evaluation
        eval_start_time = time.perf_counter()
        eval_result = eval_instance(
            self.config.eval_root,
            self.config.eval_timeout,
            episode,
            pred_patch,
            self.config.backend,
        )
        eval_duration = time.perf_counter() - eval_start_time

        # Info
        if eval_result.outcome == "pass":
            outcome = outcomes.passed()
        else:
            outcome = outcomes.failed()
        info = self.make_info_dict(
            episode_args=episode.episode_args,
            message=eval_result.message[:MAX_MESSAGE_LEN],
            pred_patch=pred_patch,
            eval_duration=eval_duration,
        )
        return outcome, info

    def get_swerl_outcomes(
        self,
        threshold: float,
        episode: SWERLDialogEpisode,
        pred_patch: str,
    ) -> tuple[dict, dict]:
        # Compute the auxiliary outcomes based on edit distance
        try:
            outcomes_ = swerl.similarities.calculate_similarities_unidiff(
                oracle_patches=[episode.episode_args["patch"]],
                pred_patches=[pred_patch],
                ignore_non_oracle_files=True,
            )
        except Exception:
            logger.exception("Error calculating swe-rl outcomes")
            outcomes_ = outcomes.failed()
        info = self.make_info_dict(
            episode_args=episode.episode_args,
            pred_patch=pred_patch,
        )
        outcomes_ = outcomes_ | outcomes.outcome("threshold", threshold)
        return outcomes_, info


def _similarities_reward(tr: env_api.Transition, scale: float = 1.0) -> float:
    if not tr.terminal:
        return 0.0

    if outcomes.successful_pass(tr.outcomes):
        return -1.0 + scale

    if "similarities" in tr.outcomes and not tr.outcomes["similarities"]:
        assert (
            len(tr.outcomes["pred_patch_dict"]) == 0
            and len(tr.outcomes["oracle_patch_dict"]) == 0
        )
        return -1.0

    def similarities_avg(
        similarities: list[swerl.similarities.ChangeSimilarity],
    ) -> float:
        return sum(map(lambda x: x["similarity"], similarities)) / len(similarities)

    if (
        "similarities" in tr.outcomes
        and similarities_avg(tr.outcomes["similarities"]) >= tr.outcomes["threshold"]
    ):
        return -1.0 + tr.outcomes["threshold"] * scale

    return -1.0


class SimilaritiesRewardFn(env_api.RewardFn):
    def __init__(self, scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    @property
    def range(self) -> tuple[float, float]:
        return (-1.0, -1.0 + self.scale)

    def __call__(self, transition: env_api.Transition) -> list[float]:
        return rewards.reward_unroll(
            transition.action, _similarities_reward(transition, scale=self.scale)
        )


config.register_reward_fn("similarities", SimilaritiesRewardFn)
config.register_reward_fn("similarities_2", partial(SimilaritiesRewardFn, scale=2.0))


config.register_env(
    config.EnvConfig(
        name="swerl_tool_env:modal:eval",
        cls=SWERLToolEnv,
        init_kwargs={
            "config": SWERLConfig(
                block_network=True,
                strict_format=False,
                sandbox_timeout=7200,  # 2 hours
                eval_timeout=600.0,
            )
        },
    )
)


config.register_env(
    config.EnvConfig(
        name="swerl_tool_env:think_tag:modal:eval",
        cls=SWERLToolEnv,
        init_kwargs={
            "config": SWERLConfig(
                block_network=True,
                strict_format=False,
                think=True,
                use_think_tag=True,
                sandbox_timeout=7200,  # 2 hours
                eval_timeout=600.0,
            )
        },
    )
)

config.register_env(
    config.EnvConfig(
        name="swerl_tool_env:noplugins:think_tag:modal:eval",
        cls=SWERLToolEnv,
        init_kwargs={
            "config": SWERLConfig(
                block_network=True,
                strict_format=False,
                plugins=False,
                think=True,
                use_think_tag=True,
                sandbox_timeout=7200,  # 2 hours
                eval_timeout=600.0,
            )
        },
    )
)
