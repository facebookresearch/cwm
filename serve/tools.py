# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import re
from dataclasses import dataclass
from uuid import uuid4

from serve.openai_api import (
    ArraySchema,
    BaseSchema,
    ChatMessage,
    FunctionDefinition,
    JsonSchema,
    ObjectSchema,
    ToolCall,
    ToolInfo,
)

logger = logging.getLogger(__name__)


def tool_from_call_id(id: str) -> str:
    """
    Given an tool call id as created by a ``Tool.parse``
    method, return the corresponding tool name.
    """
    return id[:-33]


class Tool:
    """
    Subclasses of this class define prompting, parsing, and
    formatting for tools that the model has been trained to
    use.
    """

    name: str  # name used by the model
    api_name: str  # name used in the API

    @staticmethod
    def all_tools() -> list[type["Tool"]]:
        # Depending on the model capabilities we may make this
        # list customizable through fgserve configs.
        # Keep FunctionTool last so that it acts as a default
        return [BashTool, EditTool, CreateTool, FunctionTool]

    @staticmethod
    def from_fd(fd: FunctionDefinition) -> "Tool | None":
        for cls in Tool.all_tools():
            tool = cls.from_fd(fd)
            if tool is not None:
                return tool
        return None

    def parse(self, body: str) -> ToolCall | None:
        fn_args = self._parse(body)
        if fn_args is None:
            return None
        return ToolCall(
            {
                "type": "function",
                "id": f"{fn_args[0]}-{uuid4().hex}",
                "function": {
                    "name": fn_args[0],
                    "arguments": fn_args[1],
                },
            }
        )

    def _parse(self, body: str) -> tuple[str, str] | None:
        raise NotImplementedError

    def format(self, tc: ToolCall) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


def _example_from_schema(s: JsonSchema, n: int = 1) -> str:
    if isinstance(s, BaseSchema):
        if s.type == "boolean":
            return str(n % 2 == 0).lower()
        if s.type == "number":
            return str(n)
        if s.type == "string":
            if s.enum:
                return repr(s.enum[n % len(s.enum)])
            return f'"text_{"abcdefgh"[n % 7]}"'
        return f"<a {s.type}>"
    items: list[str] = []
    if isinstance(s, ArraySchema):
        if not s.items:
            return '["a", "b", "c"]'
        for _ in range(s.minItems or 3):
            items.append(_example_from_schema(s.items, n))
            n += 1
        return f"[{', '.join(items)}]"
    if isinstance(s, ObjectSchema):
        for pn, p in s.properties.items():
            example = _example_from_schema(p, n)
            items.append(f"{json.dumps(pn)}: {example}")
            n += 1
        return f"{{{', '.join(items)}}}"
    raise NotImplementedError(f"unknown schema: {s}")


BASH_DESCRIPTION = """
<tool: bash>
[command(s)]
</tool>
Executes bash command(s) [command(s)] in the current session. [command(s)] \
can be any non-interactive bash command(s), either single or multi-line.
""".strip()


@dataclass
class BashTool(Tool):
    name = "bash"
    command_parameter: str
    api_name: str

    @staticmethod
    def from_fd(fd: FunctionDefinition) -> "BashTool | None":
        # Claude Code Bash
        if fd.name != "Bash":
            return None
        try:
            params = ObjectSchema(**fd.parameters)
        except Exception:
            return None
        if "command" not in params.properties:
            return None
        return BashTool(
            api_name=fd.name,
            command_parameter="command",
        )

    def describe(self) -> str:
        return BASH_DESCRIPTION

    def _parse(self, body: str) -> tuple[str, str] | None:
        params = {self.command_parameter: body.strip()}
        return self.api_name, json.dumps(params)

    def format(self, tc: ToolCall) -> str:
        assert tc["function"]["name"] == self.api_name
        params = json.loads(tc["function"]["arguments"])
        command = params[self.command_parameter]
        return f"<tool: bash>\n{command}\n</tool>"


SEARCH_REPLACE_REGEX = re.compile(
    r"(.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE"
)

EDIT_DESCRIPTION = """
<tool: edit>
[path]
<<<<<<< SEARCH
[search_lines]
=======
[replacement_lines]
>>>>>>> REPLACE
</tool>
Replaces [search_lines] from [path] with [replacement_lines], where [path] \
must exist and [search_lines] must uniquely and exactly match one or more \
consecutive lines from the original file, including indentations and \
whitespaces.
""".strip()


@dataclass
class EditTool(Tool):
    name = "edit"
    path_parameter: str
    old_parameter: str
    new_parameter: str
    api_name: str

    @staticmethod
    def from_fd(fd: FunctionDefinition) -> "EditTool | None":
        # Claude Code Edit
        if fd.name != "Edit":
            return None
        try:
            params = ObjectSchema(**fd.parameters)
        except Exception:
            return None
        for p in ("file_path", "old_string", "new_string"):
            if p not in params.properties:
                return None
        return EditTool(
            api_name=fd.name,
            path_parameter="file_path",
            old_parameter="old_string",
            new_parameter="new_string",
        )

    def describe(self) -> str:
        return EDIT_DESCRIPTION

    def _parse(self, body: str) -> tuple[str, str] | None:
        matches = SEARCH_REPLACE_REGEX.findall(body.strip())
        if not matches:
            return None
        path, old, new = matches[0]
        params = {
            self.path_parameter: path,
            self.old_parameter: old,
            self.new_parameter: new,
        }
        return self.api_name, json.dumps(params)

    def format(self, tc: ToolCall) -> str:
        assert tc["function"]["name"] == self.api_name
        params = json.loads(tc["function"]["arguments"])
        return "\n".join(
            [
                "<tool: edit>",
                params[self.path_parameter],
                "<<<<<<< SEARCH",
                params[self.old_parameter],
                "=======",
                params[self.new_parameter],
                ">>>>>>> REPLACE",
                "</tool>",
            ]
        )


CREATE_DESCRIPTION = """
<tool: create>
[path]
[content]
</tool>
Creates a new file at [path] with [content], where [path] must not exist, \
but its parent directory must exist.
""".strip()


@dataclass
class CreateTool(Tool):
    name = "create"
    path_parameter: str
    content_parameter: str
    api_name: str

    @staticmethod
    def from_fd(fd: FunctionDefinition) -> "CreateTool | None":
        # Claude Code Write -- The semantics are not exactly matched
        # with CWM `create` because `create` requires the file to not
        # exist. Thankfully, since `create` has stricter semantics,
        # we are not at risk to make Claude Code fail.
        if fd.name != "Write":
            return None
        try:
            params = ObjectSchema(**fd.parameters)
        except Exception:
            return None
        for p in ("file_path", "content"):
            if p not in params.properties:
                return None
        return CreateTool(
            api_name=fd.name,
            path_parameter="file_path",
            content_parameter="content",
        )

    def describe(self) -> str:
        return CREATE_DESCRIPTION

    def _parse(self, body: str) -> tuple[str, str] | None:
        lines = body.strip().split("\n", 1)
        if not lines:
            return None
        if len(lines) == 1:
            path, content = lines[0], ""
        else:
            path, content = lines
        params = {
            self.path_parameter: path,
            self.content_parameter: content,
        }
        return self.api_name, json.dumps(params)

    def format(self, tc: ToolCall) -> str:
        assert tc["function"]["name"] == self.api_name
        params = json.loads(tc["function"]["arguments"])
        return "\n".join(
            [
                "<tool: create>",
                params[self.path_parameter],
                params[self.content_parameter],
                "</tool>",
            ]
        )


@dataclass
class FunctionTool(Tool):
    name: str
    api_name: str
    description: str | None
    parameters: ObjectSchema

    @staticmethod
    def from_fd(fd: FunctionDefinition) -> "FunctionTool | None":
        try:
            return FunctionTool(
                name=fd.name,
                api_name=fd.name,
                description=fd.description,
                parameters=ObjectSchema(**fd.parameters),
            )
        except Exception:
            return None

    def describe(self) -> str:
        d = f"<tool: {self.name}>\n[json params]\n</tool>"
        if self.description:
            d += "\n" + self.description + "\n"
        d += "\nParameters must respect the following schema: "
        d += self.parameters.model_dump_json(indent=2)
        if self.parameters.required:
            reqs = ", ".join(self.parameters.required)
            d += f"\nRequired parameters: {reqs}\n"
        d += "\nExample:\n"
        example = _example_from_schema(self.parameters)
        d += f"<tool: {self.name}>\n{example}\n</tool>"
        return d

    def _parse(self, body: str) -> tuple[str, str] | None:
        return self.api_name, body.strip()

    def format(self, tc: ToolCall) -> str:
        assert tc["function"]["name"] == self.api_name
        args = tc["function"]["arguments"]
        return f"<tool: {self.name}>\n{args}\n</tool>"


def prepare_tools(
    tool_infos: list[ToolInfo],
) -> tuple[ChatMessage | None, list[Tool]]:
    """
    Given a list of available tools as obtained from an HTTP
    chat request, return a system prompt to describe the
    available tools and a list of `Tool` objects to use for
    parsing the model answer.
    """
    tools: list[Tool] = []
    for ti in tool_infos:
        tool = Tool.from_fd(ti.function)
        if tool is not None:
            tools.append(tool)
    if not tools:
        return None, []
    message = "You have access to the following tools:\n"
    for tool in tools:
        message += tool.describe() + "\n\n"
    return ChatMessage(role="system", content=message.strip()), tools


TOOL_RE = re.compile(r"(?:^|\n)<tool: (\w+)>((?s:.)*?)</tool>")


def parse_calls(msg: str, tools: list[Tool]) -> tuple[str, list[ToolCall]]:
    """
    Parse the model answer for tool calls and return the
    stripped answer with the tools calls extracted as
    structured data ready to be sent to the API user.
    """
    tool_map = {tool.name: tool for tool in tools}
    tool_calls: list[ToolCall] = []
    segments: list[tuple[int, int]] = []
    for match in TOOL_RE.finditer(msg):
        segments.append((match.start(), match.end()))
        name, body = match.groups()
        if name in tool_map:
            result = tool_map[name].parse(body)
            if result is not None:
                tool_calls.append(result)
            else:
                logger.warning(f"Parsing error for tool: {name}\n{body}")
    for beg, end in reversed(segments):
        msg = msg[:beg] + msg[end:]
    return msg.strip(), tool_calls


def format_calls(tool_calls: list[ToolCall], tools: list[Tool]) -> str:
    """
    Format tools calls as provided by the API as a string
    that the model understands.
    """
    tool_map = {tool.api_name: tool for tool in tools}
    calls: list[str] = []
    for tc in tool_calls:
        func = tc["function"]["name"]
        if func not in tool_map:
            logger.warning(f"Tool call for unknown tool: {func}\n{tc}")
            continue
        calls.append(tool_map[func].format(tc))
    return "\n".join(calls)


def _run_tests() -> None:
    global uuid4

    uuid = uuid4()
    uuid4 = lambda: uuid  # noqa: E731

    def unwrap(s: str) -> str:
        return re.sub(r"<tool: [^>]*>\n((?s:.)*?)\n</tool>", r"\1", s)

    bt = BashTool(
        command_parameter="cmd",
        api_name="Bash",
    )
    _, tcs = parse_calls("<tool: bash>\necho hello world\n</tool>", [bt])
    assert len(tcs) == 1
    tc = tcs[0]
    assert tool_from_call_id(tc["id"]) == "Bash"
    assert tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"cmd": "echo hello world"}', tc
    assert bt.format(tc) == "<tool: bash>\necho hello world\n</tool>"
    assert tc == bt.parse(unwrap(bt.format(tc)))

    ct = CreateTool(
        path_parameter="path",
        content_parameter="content",
        api_name="Create",
    )
    tc = ct.parse("abc\ndef\nghi\n\n")
    assert tc is not None
    assert tc["function"]["name"] == "Create"
    assert tc["function"]["arguments"] == json.dumps(
        {"path": "abc", "content": "def\nghi"}
    ), tc
    assert ct.format(tc) == "<tool: create>\nabc\ndef\nghi\n</tool>"
    assert tc == ct.parse(unwrap(ct.format(tc)))

    et = EditTool(
        path_parameter="path",
        old_parameter="old",
        new_parameter="new",
        api_name="Edit",
    )
    body = "foo.py\n<<<<<<< SEARCH\nabc\ndef\n=======\nghi\n>>>>>>> REPLACE"
    tc = et.parse(body)
    assert tc is not None
    assert tc["function"]["name"] == "Edit"
    assert tc["function"]["arguments"] == json.dumps(
        {
            "path": "foo.py",
            "old": "abc\ndef",
            "new": "ghi",
        }
    )
    assert et.format(tc) == f"<tool: edit>\n{body}\n</tool>"
    assert tc == et.parse(unwrap(et.format(tc)))

    ft = FunctionTool(
        name="get_weather",
        api_name="GetWeather",
        description=None,
        parameters=ObjectSchema(
            type="object",
            properties={},
            required=[],
            additionalProperties=False,
        ),
    )
    body = '{ "some": "json", "data": 10 }'
    _, tcs = parse_calls(f"<tool: get_weather>\n{body}\n</tool>", [ft])
    assert len(tcs) == 1
    tc = tcs[0]
    assert tc["function"]["name"] == "GetWeather"
    assert tc["function"]["arguments"] == body
    assert ft.format(tc) == f"<tool: get_weather>\n{body}\n</tool>"
    assert tc == ft.parse(unwrap(ft.format(tc)))

    print("all tests passed")


if __name__ == "__main__":
    _run_tests()
