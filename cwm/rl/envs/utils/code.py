# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Code manipulation utilities.
"""

from collections.abc import Callable
from typing import Literal

import re2

from cwm.exec.code.eval import ExecStatus
from cwm.rl.envs import outcomes

LanguageType = Literal["python"]


def extract_first_code(text: str, language: LanguageType = "python") -> str | None:
    # Try ```python <code> ``` first since we specifically asked for it;
    # match any triple backticks otherwise.
    pattern = {
        "python": r"``` *(python|py)(.*?)```",
        "cpp": r"``` *(cpp)(.*?)```",
        "c": r"``` *(c)(.*?)```",
        "lean": r"``` *(lean|lean4)(.*?)```",
    }[language]
    opt = re2.Options()
    opt.dot_nl = True
    if match := re2.search(pattern, text, opt):
        return match.group(2)
    pattern = r"``` *(.*?)```"
    if match := re2.search(pattern, text, opt):
        return match.group(1)
    return None


def extract_single_code(text: str, language: LanguageType = "python") -> str | None:
    # Same as extract_first_code but returns None if multiple code blocks are found of the specified language.
    pattern = {
        "python": r"``` *(python|py)(.*?)```",
        "cpp": r"``` *(cpp)(.*?)```",
        "c": r"``` *(c)(.*?)```",
        "lean": r"``` *(lean|lean4)(.*?)```",
    }[language]
    opt = re2.Options()
    opt.dot_nl = True
    candidates = re2.findall(pattern, text, opt)
    candidates = [code for lang, code in candidates]

    if not candidates:  # If no candidates found, try a more generic pattern
        pattern = r"``` *(.*?)```"
        candidates = re2.findall(pattern, text, opt)

    if len(candidates) == 1:
        return candidates[0]

    return None


def shorten_exec_info(
    shorten_backtrace_filenames: Callable[[str], str], info: str
) -> str:
    info = shorten_backtrace_filenames(info)

    # Do character-based truncation here; token limits will be applied in
    # append_user_message().
    max_info = 512
    if len(info) > max_info:
        # often in stack traces, the last lines are more informative,
        # so we're applying left-truncation at the line-level.
        if "Traceback" in info:
            info_lines = info.splitlines(True)
            shortened_info = info_lines[0]
            info_lines = info_lines[1:]
            traceback_info_len = len(shortened_info)
            kept_lines: list[str] = []
            for line in info_lines[::-1]:
                if (traceback_info_len + sum(len(kl) for kl in kept_lines)) > max_info:
                    break
                kept_lines.append(line)

            shortened_info += "... <<truncated>>\n" + "".join(reversed(kept_lines))
            info = shortened_info
        else:
            info = info[:max_info] + "... <<truncated>>"
    return info


failed_code_exec_outcomes = (
    outcomes.parsing(False)
    | outcomes.compiling(False)
    | outcomes.failed()
    | outcomes.public_passing(False)
)


def code_exec_outcomes(
    status: list[ExecStatus], public_status: list[ExecStatus]
) -> dict[str, bool]:
    return (
        outcomes.parsing(True)
        | outcomes.compiling(all(s != ExecStatus.SYNTAX_ERROR for s in status))
        | outcomes.passing(all(s == ExecStatus.SUCCESS for s in status))
        | outcomes.public_passing(all(s == ExecStatus.SUCCESS for s in public_status))
    )
