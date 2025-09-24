# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
A generic runner that takes as input serialized code
and arguments and sends back the result of executing
this code in the sandbox. Arguments and return values
must be pickle-able.

The serialized code can import modules relative to
the directory containing the lib/ folder.
"""

import argparse
import json
import marshal
import multiprocessing as mp
import os
import sys
import traceback
import types
from multiprocessing.connection import Connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--fork_before_work", action="store_true")
    parser.add_argument("input_fd", type=int)
    parser.add_argument("output_fd", type=int)
    args = parser.parse_args()

    sys.path.insert(0, args.root_dir)

    input_r = Connection(args.input_fd, writable=False)
    output_w = Connection(args.output_fd, readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    while (data := input_r.recv()) is not None:
        if args.fork_before_work:
            conn_r, conn_w = mp.Pipe(duplex=False)
            if (pid := os.fork()) == 0:
                try:
                    conn_r.close()
                    conn_w.send(run(data))
                finally:
                    # don't run the parent atexit hooks
                    os._exit(0)
            conn_w.close()
            out = conn_r.recv()
            os.waitpid(pid, 0)
        else:
            out = run(data)
        del data

        output_w.send_bytes(json.dumps(out).encode("utf8"))


def run(data: dict) -> dict:
    fn_data = data["code"]
    fn_code = marshal.loads(fn_data)
    fn = types.FunctionType(fn_code, globals())
    fargs = data["args"]
    fkwargs = data["kwargs"]
    try:
        ret = fn(*fargs, **fkwargs)
        return {"return": ret}
    except BaseException as e:
        tb = traceback.format_exc(chain=True)
        return {"except": repr(e), "traceback": tb}


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
