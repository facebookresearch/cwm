# Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import faulthandler
import json
import signal
import sys
from decimal import Decimal
from io import StringIO
from multiprocessing.connection import Connection
from unittest.mock import mock_open, patch

from runners_utils.pyext import RuntimeModule  # type: ignore

sys.setrecursionlimit(10000)


## timeout utilities
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    raise TimeoutException


## stdio capturing utilities
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    #

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


# general utilities

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"


def send_return_object(
    output_w: Connection,
    results: list[int],
    error: str,
    error_message: str,
    error_code: int,
    input: str | None = None,
    expected: str | None = None,
    prediction: str | None = None,
):
    """
    Builds the full return object in a json and returns it
    """
    metadata = {
        "error": error,
        "error_message": error_message,
        "error_code": error_code,
    }
    if input is not None:
        metadata["input"] = input.strip()
    if expected is not None:
        metadata["expected"] = expected.strip()
    if prediction is not None:
        metadata["prediction"] = prediction.strip()
    return_obj = json.dumps({"results": results, "metadata": metadata}).encode("utf8")

    output_w.send_bytes(return_obj)
    return


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def compile_code(code: str, timeout: int, output_w: Connection):
    signal.alarm(timeout)
    try:
        tmp_sol = RuntimeModule.from_string("tmp_sol", "", code)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accessible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    except Exception as e:
        send_return_object(output_w, [-1], repr(e), "Compilation Error", -1)
        return
    finally:
        signal.alarm(0)

    return compiled_sol


def get_function(compiled_sol, fn_name: str, output_w: Connection):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        send_return_object(output_w, [-1], repr(e), "Compilation Error", -1)
        return


def grade_call(
    code: str,
    all_inputs: list,
    all_outputs: list,
    fn_name: str,
    timeout: int,
    output_w: Connection,
):
    code = import_string + "\n\n" + code

    compiled_sol = compile_code(code, timeout, output_w)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name, output_w)

    if method is None:
        return

    ## inputs are stored as newline jsons for call based
    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]

    ## outputs are stored as separate jsons
    all_outputs = [json.loads(output) for output in all_outputs]

    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs, strict=False)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            prediction = method(*gt_inp)
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            # don't penalize model if it forgets added a wrapping list?
            if isinstance(gt_out, list) and gt_out:
                tmp_result = tmp_result or (prediction == gt_out[0])

            all_results.append(tmp_result)

            ## NOTE: LiveCodeBench don't perform floating point comparisons :\

            if not tmp_result:
                send_return_object(
                    output_w,
                    all_results,
                    "Wrong Answer",
                    "Wrong Answer",
                    -2,
                    repr(gt_inp),
                    repr(gt_out),
                    repr(prediction),
                )
                return
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                send_return_object(
                    output_w,
                    all_results,
                    repr(e),
                    "Time Limit Exceeded",
                    -3,
                    repr(gt_inp),
                    repr(gt_out),
                )
            else:
                all_results.append(-4)
                send_return_object(
                    output_w,
                    all_results,
                    repr(e),
                    "Runtime Error",
                    -4,
                    repr(gt_inp),
                    repr(gt_out),
                )
            return

        finally:
            signal.alarm(0)
            faulthandler.disable()

    output_w.send_bytes(
        json.dumps({"results": all_results, "metadata": {}}).encode("utf8")
    )
    return


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def compare_lines(
    stripped_prediction_lines: list[str], stripped_gt_out_lines: list[str]
) -> bool:
    if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
        return False

    for stripped_prediction_line, stripped_gt_out_line in zip(
        stripped_prediction_lines, stripped_gt_out_lines, strict=False
    ):
        if stripped_prediction_line != stripped_gt_out_line:
            return False

    return True


def send_debug_message(message: str, output_w: Connection):
    output_w.send_bytes(
        json.dumps({"results": [], "metadata": {"debug": message}}).encode("utf8")
    )


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
    output_w: Connection,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)
    # send_debug_message("cleaned code", output_w)

    ## we wrap the given code inside another function
    code = make_function(code)
    # send_debug_message("made function", output_w)

    compiled_sol = compile_code(code, timeout, output_w)
    if compiled_sol is None:
        return
    # send_debug_message("compiled code", output_w)

    method = get_function(compiled_sol, "wrapped_function", output_w)

    if method is None:
        return
    # send_debug_message("got method", output_w)

    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs, strict=False)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                call_method(method, gt_inp)
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    send_return_object(
                        output_w,
                        all_results,
                        repr(e),
                        "Time Limit Exceeded",
                        -3,
                        gt_inp,
                        gt_out,
                    )
                else:
                    all_results.append(-4)
                    send_return_object(
                        output_w,
                        all_results,
                        repr(e),
                        "Runtime Error",
                        -4,
                        gt_inp,
                        gt_out,
                    )
                return

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = (
            output_w,
            all_results + [-2],
            "Wrong Answer",
            "Wrong Answer",
            -2,
            gt_inp,
            gt_out,
            prediction,
        )

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            send_return_object(*WA_send_args)
            return

        for line_idx, (stripped_prediction_line, stripped_gt_out_line) in enumerate(
            zip(stripped_prediction_lines, stripped_gt_out_lines, strict=False)
        ):
            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: element-wise comparison if there are floating elements
            ## use `decimal` library for good floating point comparison
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## official grader uses floating point detector but `Decimal` seems cleaner!

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                send_return_object(*WA_send_args)
                return
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                send_return_object(*WA_send_args)
                return

            if decimal_prediction_line == decimal_gtout_line:
                continue

            send_return_object(*WA_send_args)
            return
        all_results.append(True)

    output_w.send_bytes(
        json.dumps({"results": all_results, "metadata": {}}).encode("utf8")
    )
    return


def main() -> None:
    """
    heavily borrowed and extended from
    https://github.com/hendrycks/apps/blob/main/eval/testing_util.py
    """

    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    data = input_r.recv()
    evaluation_sample: dict = data["sample"]
    code: str = data["code"]
    debug: bool = data.get("debug", False)
    timeout: int = data.get("timeout", 6)

    try:
        in_outs = json.loads(evaluation_sample["input_output"])
    except ValueError as e:
        raise e

    fn_name = in_outs.get("fn_name", None)

    all_inputs = in_outs["inputs"]
    all_outputs = in_outs["outputs"]

    try:
        if fn_name is None:  # io based
            grade_stdio(
                code,
                all_inputs,
                all_outputs,
                timeout,
                output_w,
            )
        else:
            grade_call(
                code,
                all_inputs,
                all_outputs,
                fn_name,
                timeout,
                output_w,
            )
    except Exception as e:
        send_return_object(output_w, [-1], repr(e), "TestRunner Error", -1)


if __name__ == "__main__":
    main()
