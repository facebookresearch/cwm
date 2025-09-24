# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
# A command-line interface (CLI) for the SWERL remote client. It accepts a serialized JSON string from stdin.
#
# Testing:
#    python -m remote.server --host 0.0.0.0 --port 8888
#    echo '{"method": "get_new_id"}' | python -m remote.client_cli --host localhost --port 8888

import argparse
import json
import sys

from .client import PersistentClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()
    payload: dict = json.loads(sys.stdin.read())
    with PersistentClient(host=args.host, port=args.port) as client:
        method = payload["method"]
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        func = getattr(client, method, None)
        result = func(*args, **kwargs)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
