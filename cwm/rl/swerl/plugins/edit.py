# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
# type: ignore
# NOTE: please do not simplify the type definitions.
# This script is intended for an older version of Python.
import sys
import re
import os
from typing import Tuple, List

SEARCH_REPLACE_REGEX = re.compile(
    r"(.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE"
)


PARSE_ERROR = """
Failed to parse the edit tool call. Make sure it has the following format:

<tool: edit>
[path]
<<<<<<< SEARCH
[search_lines]
=======
[replacement_lines]
>>>>>>> REPLACE
</tool>
""".strip()

SNIPPET_LINES = 8


def find_all_occurrences(text: str, substr: str):
    indices: List[int] = []
    start = 0
    while True:
        index = text.find(substr, start)
        if index == -1:
            break
        indices.append(index)
        # Move start index forward to search for next occurrence.
        start = index + 1
    return indices


def main(command: str) -> Tuple[bool, str]:
    """Parse the edit command, apply the changes to the code state in place, and return a feedback message"""
    predicted_path_search_replaces: List[Tuple[str, str, str]] = (
        SEARCH_REPLACE_REGEX.findall(command)
    )
    if len(predicted_path_search_replaces) == 0:
        return False, PARSE_ERROR
    if len(predicted_path_search_replaces) > 1:
        return (
            False,
            f"Only one search/replace block is allowed. Got: {len(predicted_path_search_replaces)}.",
        )
    path, search_text, replacement_text = predicted_path_search_replaces[0]
    if not os.path.exists(path):
        return (False, f"The path {path} does not exist. Please provide a valid path.")
    with open(path, "r") as f:
        file_content = f.read()
    # "\n" to make sure indentation is preserved
    file_content = "\n" + file_content
    search_text = "\n" + search_text

    occurrences = find_all_occurrences(file_content, search_text)
    if len(occurrences) == 0:
        message = f"No replacement was performed, the search lines did not appear verbatim in {path}."
        first_newline = search_text.find("\n", 1)
        # [1:] as we prepended "\n" to search_text
        first_search_line = (
            search_text[1:first_newline] if first_newline != -1 else search_text[1:]
        )
        if first_search_line.lstrip() in file_content:
            message += f" The search lines must exactly match one or more consecutive lines from the file, with indentations preserved."
        return False, message
    elif len(occurrences) > 1:
        newline_occurrences = find_all_occurrences(file_content, "\n")
        linenums = [
            linenum + 1
            for linenum, idx in enumerate(newline_occurrences)
            if file_content.startswith(search_text, idx)
        ]
        return (
            False,
            f"No replacement was performed. Multiple occurrences of the search text in lines {linenums} of {path}. Please ensure it is unique.",
        )

    assert len(occurrences) == 1
    replacement_text = "\n" + replacement_text
    new_file_content = file_content.replace(search_text, replacement_text)[1:]
    with open(path, "w") as f:
        f.write(new_file_content)

    # Prepare the success message
    occurrence = occurrences[0]
    line = file_content.count("\n", 0, occurrence) + 1
    num_lines = new_file_content.count("\n") + 1
    # Display the surrounding lines after the replacement
    # https://github.com/anthropics/anthropic-quickstarts/blob/81c4085944abb1734db411f05290b538fdc46dcd/computer-use-demo/computer_use_demo/tools/edit.py#L187-L198
    start_line = max(1, line - SNIPPET_LINES)
    # -1 because we prepended "\n" to replacement_text
    end_line = line + SNIPPET_LINES + replacement_text.count("\n") - 1
    snippet_lines = new_file_content.split("\n")[start_line - 1 : end_line]
    snippet_content = "\n".join(
        [f"{i:6}\t{line}" for i, line in enumerate(snippet_lines, start=start_line)]
    )

    message = (
        f"File {path} has been successfully updated at line {line} and now contains {num_lines} lines. "
        f"The surrounding lines after the edit are shown below:\n{snippet_content}\n"
        f"Review the changes and make sure they are as expected."
    )
    return True, message


if __name__ == "__main__":
    command = sys.stdin.read()
    success, feedback = main(command)
    print(feedback)
    exit(0 if success else 1)
