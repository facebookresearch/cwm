# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
# type: ignore
# NOTE: please do not simplify the type definitions.
# This script is intended for an older version of Python.
import sys
import os
from typing import Tuple


def main(command: str) -> Tuple[bool, str]:
    lines = command.split("\n", 1)
    assert len(lines) > 0
    if len(lines) == 1:
        path, new_content = lines[0], ""
    else:
        path, new_content = lines

    if os.path.exists(path):
        return (
            False,
            f"{path} already exists. Cannot create a new file at an existing path.",
        )

    if path == "":
        return False, "Path is empty. Please provide a valid path."

    parent_dir = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(parent_dir):
        return (
            False,
            f"File creation failed. The parent directory {parent_dir} does not exist.",
        )

    with open(path, "w") as f:
        f.write(new_content)

    num_lines = new_content.count("\n") + 1
    return (
        True,
        f"File created successfully at: {path} ({num_lines} lines).",
    )


if __name__ == "__main__":
    command = sys.stdin.read()
    if command.endswith("\n"):
        # NOTE: we expect a heredoc string, which always ends with a newline.
        # So we trim the last newline character.
        command = command[:-1]
    success, feedback = main(command)
    print(feedback)
    exit(0 if success else 1)
