# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from typing import cast

from .errors import FormatError
from .tools import ToolType, make_bash, submit_diff_file

# Default tools
DEFAULT_TOOLS = {
    "bash": cast(ToolType, make_bash),
    "submit": cast(ToolType, submit_diff_file),
}


TOOL_CALL_REGEX = r"<tool:\s([a-zA-Z_]+)>\n(.*?)\n</tool>"


def parse_tool_calls(
    raw_output: str,
    ensure_nonempty_args: bool = True,
    one_and_only: bool = True,
) -> list[tuple[str, str]]:
    """Parse tool calls from the raw output."""
    tool_calls: list[tuple[str, str]] = re.findall(
        TOOL_CALL_REGEX, raw_output, re.DOTALL
    )

    if one_and_only and len(tool_calls) > 1:
        tool_names = [tc[0] for tc in tool_calls]
        raise FormatError(
            f"Expected one tool call, but found {len(tool_calls)}: {', '.join(tool_names)}. "
            "Ensure only one <tool:...></tool> block is present in the response."
        )

    num_opening_tags = raw_output.count("<tool:")
    num_closing_tags = raw_output.count("</tool>")
    if len(tool_calls) != num_opening_tags or len(tool_calls) != num_closing_tags:
        raise FormatError(
            f"Mismatch in tool tag counts: found {len(tool_calls)} tool calls via regex, "
            f"but counted {num_opening_tags} opening tags and "
            f"{num_closing_tags} closing tags."
        )

    if len(tool_calls) == 0:
        raise FormatError("No tool calls found")

    outputs = list[tuple[str, str]]()
    for tool_name, tool_input in tool_calls:
        tool_name = tool_name.strip()
        if ensure_nonempty_args and tool_input.strip() == "":
            raise FormatError(
                f"Tool '{tool_name}' requires non-empty arguments, but received empty or whitespace-only input."
            )
        outputs.append((tool_name, tool_input))
    return outputs
