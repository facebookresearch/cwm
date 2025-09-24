# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

from .errors import NoSuchToolError
from .remote.session import SessionOutput


# API
class BashResult(SessionOutput):
    exit_code: int


@dataclass(frozen=True)
class ToolCallResult:
    output: str
    success: bool
    metadata: Any = None


ToolType: TypeAlias = Callable[["ToolBackend", str], ToolCallResult]
T = TypeVar("T")


class ToolBackend(Protocol):
    """
    Small wrapper around a container that provides a defined set of
    LLM tools for interaction.

    During initialization, it is supposed to raise `BackendInitError`
    if the container cannot be created or started.
    """

    @property
    def tools(self) -> dict[str, ToolType]: ...

    def run_bash(self, command: str) -> BashResult: ...

    # NOTE(yuxiang): stateless command execution is tricky because the tool calling
    # commands implicitly depends on the current context. For example, LLM can edit a file
    # using relative paths, while a stateless run_bash function doesn't know the working dir.
    # TODO: we should also add a `cwd` parameter to `client.run_command`
    # def run_bash_stateless(self, command: str) -> BashResult:
    #     """
    #     This is a stateless version of `run_bash` that always runs the command
    #     in the same environment. Why do we need this? For some tools like "edit",
    #     we want them to be run in a consistent environment. Otherwise, for example,
    #     an agent can switch the env to break the tool.
    #     """
    #     return self.run_bash(command)

    def destroy(self) -> None: ...

    def __enter__(self: T) -> T:
        return self

    def __exit__(self, exc_type: type[T], exc_value: T, traceback: T) -> None:
        self.destroy()

    def apply_tool(
        self,
        tool_name: str,
        tool_input: str,
    ) -> ToolCallResult:
        """
        Apply a tool to the given command and return the result.
        Raises NoSuchToolError if the tool is not found.
        """
        if tool_name not in self.tools:
            raise NoSuchToolError(f"Tool '{tool_name}' not found.")
        tool = self.tools[tool_name]
        return tool(self, tool_input)


# Tool builders


def bash_feedback_default(output: BashResult) -> ToolCallResult:
    if output["status"] == "success":
        return ToolCallResult(output["output"], True)
    if output["error_type"] in ["exit", "broken_pipe", "other"]:
        # buffer = output["output"]
        feedback = "The current session has terminated unexpectedly."
    elif output["error_type"] == "timeout":
        feedback = "The current session has terminated due to timeout."
    elif output["error_type"] == "too_long":
        feedback = "The current session has terminated because your previous command produced too much output."
    else:
        raise AssertionError
    feedback += " A fresh session has started with the current working directory reset to default."
    return ToolCallResult(feedback, False)


EOF = "EOF_938176592"
BashFeedbackFnType = Callable[[BashResult], ToolCallResult]


def make_bash(
    backend: ToolBackend,
    command: str,
    bash_feedback_fn: BashFeedbackFnType = bash_feedback_default,
) -> ToolCallResult:
    result = backend.run_bash(command)
    return bash_feedback_fn(result)


def make_python_plugin(
    backend: ToolBackend,
    command: str,
    script_path: str,
) -> ToolCallResult:
    # We make the plugin path configurable, so the user can set a fixed python path
    # for running plugins. This is helpful when the images use an old python that makes
    # the plugin scripts fail (e.g., python 3.5 in django)
    command = (
        f"${{PYPLUGIN_PYTHON_PATH:-python3}} {script_path} <<'{EOF}'\n{command}\n{EOF}"
    )
    bash_result = backend.run_bash(command)
    success = bash_result["status"] == "success" and bash_result["exit_code"] == 0
    return ToolCallResult(bash_result["output"], success)


def make_python_plugins_from_dir(
    plugin_root: str,
    bind_target: str,
    plugin_names: list[str] | None = None,
) -> dict[str, ToolType]:
    plugins = dict[str, ToolType]()
    for path in Path(plugin_root).rglob("*.py"):
        name = path.stem
        if plugin_names is not None and name not in plugin_names:
            continue
        target_path = Path(bind_target) / path.relative_to(plugin_root)
        target_path_str = target_path.as_posix()
        plugins[name] = partial(make_python_plugin, script_path=target_path_str)

    # Make sure the specified plugins are available
    if plugin_names is not None:
        for name in plugin_names:
            assert name in plugins, f"Plugin '{name}' not found in {plugin_names}"

    return plugins


def submit_file(backend: ToolBackend, command: str) -> ToolCallResult:
    path = command.strip()
    bash_result = backend.run_bash(f"cat {path}")
    if bash_result["status"] == "error" or bash_result["exit_code"] != 0:
        error = f"Failed to submit {path}. Make sure it exists and is readable."
        return ToolCallResult(error, False)
    return ToolCallResult(bash_result["output"], True)


def submit_diff_file(backend: ToolBackend, command: str) -> ToolCallResult:
    path = command.strip()
    bash_result = backend.run_bash(f"cat {path}")
    if bash_result["status"] == "error" or bash_result["exit_code"] != 0:
        error = f"Failed to submit {path}. Make sure it exists and is readable."
        return ToolCallResult(error, False)
    if bash_result["output"].strip() == "":
        error = f"Failed to submit {path}. It is an empty file."
        return ToolCallResult(error, False)
    if not bash_result["output"].startswith("diff"):
        error = f"Failed to submit {path}. It is not a valid diff file."
        return ToolCallResult(error, False)
    return ToolCallResult(bash_result["output"], True)
