# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import sys
import time
from os import path as osp

from ..lib.server import (
    ForkServer,
)
from .types import ExecResult, ExecStatus

_runners_dir = osp.join(osp.dirname(osp.abspath(__file__)), "runners")


def runners_dir() -> str:
    return _runners_dir


logger = logging.getLogger()


def exec_lcbcodegen_tests(
    code: str,
    sample: dict,
    timeout: int = 6,  # per test timeout
    fork_server: ForkServer | None = None,
    early_stopping: bool = True,
    **spawn_args,
) -> list[ExecResult]:
    assert early_stopping, "early_stopping = False not supported for LCB Codegen"

    if fork_server is None:
        fork_server = ForkServer.global_instance()

    num_tests = len(json.loads(sample["input_output"])["inputs"])

    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, osp.join(runners_dir(), "python_lcbcodegen.py")],
        **spawn_args,
    )
    try:
        input_w.send({"code": code, "sample": sample, "timeout": timeout})

        test_results = []  # [ExecResult()] * num_tests
        start_time = time.perf_counter()

        if not output_r.poll(
            timeout=start_time + timeout * num_tests - time.perf_counter()
        ):
            test_results.append(ExecResult(status=ExecStatus.TIMEOUT))
        else:
            res = output_r.recv()
            if "stderr" in res:
                test_results.append(
                    ExecResult(status=ExecStatus.FAILURE, info=res["stderr"])
                )
                return test_results
            results = res["results"]
            metadata = res["metadata"]
            for res_idx, result in enumerate(results):
                if result == True:  # noqa: E712
                    test_results.append(ExecResult(status=ExecStatus.SUCCESS))
                else:
                    assert res_idx == len(results) - 1
                    assert "error_code" in metadata
                    if metadata["error_code"] == -1:
                        test_results.append(
                            ExecResult(
                                status=ExecStatus.SYNTAX_ERROR, info=metadata["error"]
                            )
                        )
                    elif metadata["error_code"] == -2:
                        test_results.append(
                            ExecResult(
                                status=ExecStatus.FAILURE,
                                info=f"Expected output `{metadata['expected']}` but got `{metadata['prediction']}`",  # " for input `{metadata['input']}`",
                            )
                        )
                    elif metadata["error_code"] == -3:
                        test_results.append(
                            ExecResult(
                                status=ExecStatus.TIMEOUT,
                                info=f"Timeout error on `{metadata['input']}`",  # f", expected `{metadata['expected']}`"
                            )
                        )
                    elif metadata["error_code"] == -4:
                        test_results.append(
                            ExecResult(
                                status=ExecStatus.EXCEPTION,
                                info=f"Runtime error on `{metadata['input']}`",  # f", expected `{metadata['expected']}`"
                            )
                        )
                    else:
                        raise NotImplementedError("")

        return test_results
    except Exception as e:
        logger.exception(f"Code execution failed: {repr(e)}")
        return [ExecResult()]
    finally:
        input_w.close()
        output_r.close()
        fork_server.kill(vpid)
