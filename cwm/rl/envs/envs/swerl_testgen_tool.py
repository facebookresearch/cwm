# Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import TypedDict, cast

import cwm.rl.swerl as swerl
from cwm.rl.envs import api as env_api
from cwm.rl.envs import config, dialog, outcomes
from cwm.rl.envs.prompts import get_cwm_sys_prompt, think_tag_prompt
from cwm.rl.swerl.default_backends import get_default_modal_backend
from cwm.rl.swerl.default_tools import DEFAULT_TOOLS, parse_tool_calls
from cwm.rl.swerl.eval_backend.eval import (
    EvalResult,
    eval_instance_general_modal,
)
from cwm.rl.swerl.modal_backend import ModalBackend
from cwm.text.datatypes import CWMChatMessage
from cwm.text.tokenizers import CWMInstructTokenizer, Tokenizer

from .swerl_tool import (
    BUDGET_NOTICE_TEMPLATE,
    FEEDBACK_OUTPUT_PROMPT,
    SYSTEM_UBUNTU,
    budget_notice,
)

SUBMISSION_FEEDBACK_PROMPT = """
Review the following test which is marked as a submission:

<submission>
{test}
</submission>
""".strip()


USER_TESTGEN = """
Here is a GitHub issue and you are tasked to generate a test (`test_issue.py`) that can be used to reproduce and verify the fix for the issue:

<issue_description>
{issue}
</issue_description>

This test file should be completely self-contained and executable directly with `python test_issue.py`, without requiring any testing frameworks like pytest or unittest.

IMPORTANT GUIDELINES:

Create a standalone Python script (test_issue.py) that:
   - Imports only the necessary modules from the repository
   - Sets up the minimum environment needed to reproduce the issue
   - Contains all logic within the script itself (no external test dependencies)
   - Runs quickly and terminates itself (no background servers or long-running processes)

CRITICAL: For each of the test cases: your test script MUST use these EXACT print statements to indicate test results for each test case:
   - Print "Issue reproduced" when the code confirms the bug exists
   - Print "Issue resolved" when the code runs without the issue
   - Print "Other issues" if there are any unexpected problems

Example format for the test script:

```python
from sqlfluff import lint

def test__rules__std_L060_raised() -> None:
    try:
        sql = "SELECT   IFNULL(NULL, 100),
            NVL(NULL,100);"
        result = lint(sql, rules=["L060"])
        assert len(result) == 2
    except:
        print("Other issues")
        return

    try:
        assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
        assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."
        print("Issue resolved")
    except AssertionError:
        print("Issue reproduced")
        return

    return

test__rules__std_L060_raised()
```

The [result] argument of <tool: submit> should be the path to the generated test file (`test_issue.py`), like this:

<tool: submit>
test_issue.py
</tool>

Your final response should be a summary of how the test works. I've uploaded the corresponding code repository at {repo_root} and installed all the necessary dependencies. Now, the bash session has started, with the current working directory set to the repo root.
""".strip()

logger = logging.getLogger()


@dataclass
class SWERLConfig:
    context_size: int = 131072
    session_timeout: float = 120.0
    sandbox_timeout: int = 7200

    # # Must be an NFS path for enroot, because workers & servers may not be in the same node.
    # # We need eval_root visible to the enroot server
    # # Not required for modal as we can mount the local path remotely
    # eval_root: str = ""

    max_prompt_len: int = 8192
    max_user_turn_len: int = 32768
    max_action_len: int = 65536
    max_turns: int = 128
    eval_timeout: float = 120.0
    block_network: bool = True

    think: bool = False
    use_think_tag: bool = False

    @property
    def use_think_token(self) -> bool:
        return self.think and not self.use_think_tag

    @property
    def assistant_prefix(self) -> str:
        if self.think and self.use_think_tag:
            return think_tag_prompt()
        return ""

    def thinking_end_text(self, tokenizer: CWMInstructTokenizer) -> str:
        if self.think and self.use_think_tag:
            return "</think>"
        return tokenizer.THINKING_END_ID


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
    test: str | None = None

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
MAGIC_STR = "SWERL_TESTGEN_928630291888"


def create_patch_from_code(python_code: str) -> str:
    patch_header = """diff --git a/test_issue.py b/test_issue.py
new file mode 100644
index 0000000..e69de29
"""
    patch_body = []
    patch_body.append("--- /dev/null")
    patch_body.append("+++ b/test_issue.py")

    code_lines = python_code.split("\n")
    patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")

    for line in code_lines:
        patch_body.append(f"+{line}")

    return patch_header + "\n".join(patch_body) + "\n"


def eval_instance(
    eval_timeout: float,
    episode: SWERLDialogEpisode,
    test: str,
) -> EvalResult:
    assert episode.tool_backend is not None

    if test.strip() == "":
        return EvalResult(outcome="fail", message="Empty code")

    if "Issue reproduced" not in test or "Issue resolved" not in test:
        return EvalResult(
            outcome="fail",
            message="'Issue reproduced' or 'Issue resolved' not in test",
        )

    # check syntax
    try:
        ast.parse(test)
    except SyntaxError as e:
        return EvalResult(
            outcome="fail",
            message=f"Syntax error in test: {e.msg} at line {e.lineno}, column {e.offset}",
        )

    workdir = episode.episode_args["repo_root_path"]

    startup_commands = swerl.common.restore_env(episode.episode_args)  # type: ignore
    git = f"git -C {workdir}"
    startup_commands += "\n" + "\n".join(
        [
            # remove all remotes
            f"{git} remote | xargs -n1 {git} remote remove",
            # delete all branches except the current one
            textwrap.dedent(
                f"""\
            cur=$({git} symbolic-ref --quiet --short HEAD);
            {git} for-each-ref --format='%(refname)' refs \\
                | awk -v cur="$cur" '$0 != "refs/heads/" cur' \\
                | xargs -n1 -I{{}} {git} update-ref -d {{}}
        """
            ),
            # unset upstream for current branch
            f"{git} branch --unset-upstream",
            # purge reflog
            f"{git} reflog expire --expire=now --all",
            f"{git} gc --prune=now",
        ]
    )
    # XXX(yuxiang): source ~/.bashrc is a temporary hack
    setup_script = f"source ~/.bashrc\n{startup_commands}\necho {MAGIC_STR}"
    default_args = dict(
        image_url=episode.episode_args["docker_url"],
        timeout=eval_timeout,
        workdir=workdir,
        setup_script=setup_script,
    )
    test_patch = create_patch_from_code(test)

    # 1. should print "Issue reproduced"
    eval_result = eval_instance_general_modal(
        test_script="python test_issue.py",
        test_patch=test_patch,
        **default_args,  # type: ignore
    )
    if (
        "Issue reproduced" not in eval_result.message
        or "Issue resolved" in eval_result.message
        or "Other issues" in eval_result.message
    ):
        return EvalResult(
            outcome="fail",
            message=f"Test script did not print 'Issue reproduced' or printed 'Issue resolved' or 'Other issues'\n\n{eval_result.message}",
        )

    reproduction_message = eval_result.message
    assert "Issue reproduced" in reproduction_message
    # pass only means it can reproduce the original issue, not that the test patch can verify the fix
    return EvalResult(
        outcome="pass",
        message=f">>> Reproduction message\n{reproduction_message}",
        duration=eval_result.duration,
    )


class SWERLToolEnv(dialog.DialogEnv):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: SWERLConfig,
    ) -> None:
        self.config = config
        super().__init__(tokenizer)

    def max_action_len(self, state: env_api.State) -> int:
        return self.config.max_action_len

    @staticmethod
    def make_info_dict(
        episode_args: SWERLDialogEpisode.Args,
        message: str = "",
        eval_duration: float = -1.0,
        test: str | None = None,
    ) -> dict:
        return dict(
            source=episode_args.get("source", "N/A"),
            instance_id=episode_args.get("instance_id", "N/A"),
            message=message,
            eval_duration=eval_duration,
            test=test,
        )

    def start(
        self, episode_args: dict | None = None
    ) -> tuple[env_api.State, env_api.Transition]:
        assert episode_args is not None, "episode_args must be provided."

        env_prompt = SYSTEM_UBUNTU.format(
            context_size=self.config.context_size,
            max_turns=self.config.max_turns,
            session_timeout=f"{int(self.config.session_timeout)}-second",
        )

        sys_prompt = get_cwm_sys_prompt(
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

        user_prompt = USER_TESTGEN.format(
            issue=episode.episode_args["issue"],
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
                tools = {
                    **DEFAULT_TOOLS,
                    "submit": swerl.tools.submit_file,
                }
                episode.tool_backend = get_default_modal_backend(
                    tools=tools,
                    image_url=episode_args["docker_url"],
                    session_timeout=self.config.session_timeout,
                    work_dir=episode_args["repo_root_path"],
                    startup_commands=swerl.common.restore_env(episode_args),  # type: ignore
                    server_python_path=swerl.common.get_server_python_path(
                        episode_args  # type: ignore
                    ),
                    sandbox_timeout=self.config.sandbox_timeout,
                    block_network=self.config.block_network,
                )
            tool_result = episode.tool_backend.apply_tool(tool_name, tool_input)
        except (
            swerl.errors.FormatError,
            swerl.errors.NoSuchToolError,
        ) as e:
            # Two possible situations:
            # 1. the assistant has finished the task and is presenting a summary
            if state.test is not None:
                state.close()
                outcomes_, info = self.eval(episode, state.test)
                return self.transition(
                    terminal=True,
                    action=action,
                    action_str=action_str,
                    outcomes=outcomes_,
                    info=info,
                )
            else:
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
                state.test = tool_result.output
                feedback = SUBMISSION_FEEDBACK_PROMPT.format(test=state.test)
            else:
                feedback = tool_result.output

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

    def eval(self, episode: SWERLDialogEpisode, test: str) -> tuple[dict, dict]:
        # Evaluation
        eval_start_time = time.perf_counter()
        eval_result = eval_instance(
            self.config.eval_timeout,
            episode,
            test,
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
            test=test,
            eval_duration=eval_duration,
        )
        return outcome, info


config.register_env(
    config.EnvConfig(
        name="swerl_testgen_tool_env:think_tag:modal:eval",
        cls=SWERLToolEnv,
        init_kwargs={
            "config": SWERLConfig(
                think=True,
                use_think_tag=True,
            )
        },
    )
)
