# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from os import path as osp
from pathlib import Path

import cwm
from cwm.exec.code.lib.server import (
    ForkServer,
    JSONConnection,
)

_runners_dir = osp.dirname(osp.abspath(__file__))


def runners_dir() -> str:
    return _runners_dir


logger = logging.getLogger()


@dataclass
class CompareResult:
    result: bool
    normalized_expr_1: str = ""  # Normalized expression for expr_1
    normalized_expr_2: str = ""  # Normalized expression for expr_2
    info: str = ""
    duration: float = -1  # Execution time in seconds


def _get_compare_results(
    output_r: JSONConnection,
    timeout: float,
) -> CompareResult:
    start_time = time.perf_counter()
    try:
        while True:
            if not output_r.poll(timeout=start_time + timeout - time.perf_counter()):
                # Timeout occurred
                return CompareResult(result=False, info="Timeout", duration=timeout)

            res = output_r.recv()

            if "exception" in res:
                return CompareResult(
                    result=False,
                    info=res["exception"],
                    duration=time.perf_counter() - start_time,
                )

            return CompareResult(
                result=res["result"],
                normalized_expr_1=res["normalized_expr_1"],
                normalized_expr_2=res["normalized_expr_2"],
                info="",
                duration=time.perf_counter() - start_time,
            )

    except BaseException as e:
        logger.exception(
            f"Error parsing execution results: {e}\n{traceback.format_exc()}"
        )
        return CompareResult(result=False, info=str(e), duration=-1)


def exec_compare_values(
    expr_1: str,
    expr_2: str,
    expr_1_regex: str = ".*",
    expr_2_regex: str = ".*",
    exact_bracket: bool = True,
    expr_1_expect_boxed: bool = True,
    timeout: float = 30.0,
    fork_server: ForkServer | None = None,
    **spawn_args,
) -> CompareResult:
    if fork_server is None:
        fork_server = ForkServer.global_instance()

    python_path = Path(cwm.__path__[0]).parent
    os_env = dict(os.environ)
    os_env["PYTHONPATH"] = str(python_path)

    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, osp.join(runners_dir(), "compare_runner.py")],
        env=os_env,
        **spawn_args,
    )

    try:
        input_w.send(
            {
                "expr_1": expr_1,
                "expr_2": expr_2,
                "expr_1_regex": expr_1_regex,
                "expr_2_regex": expr_2_regex,
                "exact_bracket": exact_bracket,
                "expr_1_expect_boxed": expr_1_expect_boxed,
            }
        )
        return _get_compare_results(output_r, timeout)
    finally:
        input_w.close()
        output_r.close()
        fork_server.kill(vpid)


def exec_math_verify_compare_values(
    expr_1: str,
    expr_2: str,
    timeout: float = 30.0,
    fork_server: ForkServer | None = None,
    **spawn_args,
) -> CompareResult:
    if fork_server is None:
        fork_server = ForkServer.global_instance()

    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, osp.join(runners_dir(), "math_verify_runner.py")],
        **spawn_args,
    )

    try:
        input_w.send(
            {
                "expr_1": expr_1,
                "expr_2": expr_2,
            }
        )
        return _get_compare_results(output_r, timeout)
    finally:
        input_w.close()
        output_r.close()
        fork_server.kill(vpid)
