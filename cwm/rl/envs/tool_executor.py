# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from os import path as osp

import re2

from cwm.exec.code.eval.python import runners_dir
from cwm.exec.code.lib.server import ForkServer
from cwm.text.datatypes import CWMChatMessage

logger = logging.getLogger()


ToolCallback = Callable[[str], str]


@dataclass
class ToolSpec:
    help: str
    func: ToolCallback


def _python_tool_forkserver(source: str) -> str:
    timeout = 10

    fork_server = ForkServer.global_instance()
    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, osp.join(runners_dir(), "python_tool.py")],
    )

    try:
        input_w.send({"source": source})
        if not output_r.poll(timeout=timeout):
            return "error: Timeout"
        res = output_r.recv()

        if "done" in res:
            logger.warning(
                "ForkServer execution failed, return code {res['returncode']}"
            )
            result = "error"
        elif "error" in res:
            result = f"error: {res['error']}"
        else:
            result = "completed."
            if stdout := res.get("stdout"):
                result = result + f" [stdout]{stdout}[/stdout]"
            if stderr := res.get("stderr"):
                result = result + f" [stderr]{stderr}[/stderr]"
        return result
    except Exception:
        logger.exception("ForkServer execution failed")
        return "error"
    finally:
        input_w.close()
        output_r.close()
        fork_server.kill(vpid)


class PythonToolWithLimits:
    def __init__(self, max_executions: int):
        self.n_executions = 0
        self.max_executions = max_executions

    def __call__(self, source: str) -> str:
        if self.n_executions >= self.max_executions:
            return f"error: maximum execution count ({self.max_executions}) reached"

        self.n_executions += 1
        return _python_tool_forkserver(source)


def python_tool() -> dict[str, ToolSpec]:
    return {
        "python": ToolSpec(
            help="Run Python code in an isolated sandbox.",
            func=_python_tool_forkserver,
        ),
    }


def python_tool_limits(*, max_executions: int) -> dict[str, ToolSpec]:
    return {
        "python": ToolSpec(
            help=f"Run Python code in an isolated sandbox up to {max_executions} times.",
            func=PythonToolWithLimits(max_executions=max_executions),
        ),
    }


class ToolExecutor:
    """
    Utility class for tool calling support.

    Tools are registered in `__init__()` as `str` -> `str` callbacks.
    `maybe_exec_tools()` will invoke all tool calls for a given generated
    assistant text and return a CWMChatMessage for each call (or an empty list
    if no calls were requested).
    """

    # TODO: more robust parsing logic so that we can have at least "<tool>..</tool>" in the call?
    tool_pattern = r"<tool: ([^>]*?)>\n(.*?)</tool>"

    def __init__(self, tools: dict[str, ToolSpec]) -> None:
        self.tools = tools

    def system_prompt(self, user_prompt: str) -> CWMChatMessage:
        if not self.tools:
            return CWMChatMessage.system(user_prompt)

        tool_help = "\n\nThe following tools are available:\n\n"
        for tool, spec in self.tools.items():
            tool_help += f"- {tool}: {spec.help}\n"

        return CWMChatMessage.system(user_prompt + tool_help)

    def maybe_exec_tools(self, assistant_text: str) -> list[CWMChatMessage]:
        messages: list[CWMChatMessage] = []
        opt = re2.Options()
        opt.dot_nl = True

        for m in re2.finditer(self.tool_pattern, assistant_text, opt):
            tool, content = m.group(1), m.group(2)
            if tool not in self.tools:
                messages.append(self.no_such_tool_message(tool))
                continue

            # Exceptions from tool calls are not caught -- tools need to manage
            # them and produce appropriate error messages
            result = self.tools[tool].func(content)
            messages.append(CWMChatMessage.tool_result(result, tool=tool))

        return messages

    def no_such_tool_message(self, tool: str) -> CWMChatMessage:
        return CWMChatMessage.tool_result("error: No such tool", tool=tool)
