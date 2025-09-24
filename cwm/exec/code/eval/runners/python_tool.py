# Copyright (c) Meta Platforms, Inc. and affiliates.

import builtins
import json
import linecache
import re
import sys
import traceback
from io import StringIO
from multiprocessing.connection import Connection


class EarlyReturnException(Exception):
    """Custom exception to handle early returns in user code."""

    pass


def custom_exit(*args, **kwargs):
    """Function to replace exit() and quit(), raises EarlyReturnException."""
    raise EarlyReturnException()


def execute_code(code):
    """Execute wrapper to handle EarlyReturnException and if __name__ == '__main__'"""
    try:
        # Execute user code with __name__ set to '__main__'
        exec(code, {"__name__": "__main__"})
    except EarlyReturnException:
        # Early return from user code
        pass


def format_tb(tb):
    """Strip the traceback to only include the user code."""
    frames = tb.splitlines()
    i = 0
    while i < len(frames) - 1:
        if re.match(r" *File \".*/python_tool.py\".*", frames[i]):
            frames.pop(i)
            frames.pop(i)
            if i < len(frames) and set(frames[i]) == {" ", "^"}:
                # python 3.11 error markers
                frames.pop(i)
        else:
            i += 1
    return "\n".join(frames)


def main() -> None:
    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    data = input_r.recv()
    source: str = data["source"]

    try:
        compiled = compile(source, "<source>", "exec")
        linecache.cache["<source>"] = (
            len(source),
            None,
            source.splitlines(True),
            "<source>",
        )
    except BaseException:
        tb = format_tb(traceback.format_exc())
        output_w.send_bytes(json.dumps({"error": tb}).encode("utf8"))
        return

    sys.stdin = StringIO()
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    builtins.exit = custom_exit
    builtins.quit = custom_exit
    sys.exit = custom_exit

    try:
        execute_code(compiled)
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    except BaseException:
        tb = format_tb(traceback.format_exc())
        output_w.send_bytes(json.dumps({"error": tb}).encode("utf8"))
    else:
        output_w.send_bytes(json.dumps({"stdout": out, "stderr": err}).encode("utf8"))


if __name__ == "__main__":
    main()
