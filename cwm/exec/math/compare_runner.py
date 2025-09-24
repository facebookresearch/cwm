# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import sys
import traceback
from multiprocessing.connection import Connection

from cwm.exec.math.math_process import (
    compare_expressions_complex,
    normalize_final_answer,
    try_evaluate_frac,
    try_evaluate_latex,
)


def eval_answer(answer: str, expect_boxed: bool) -> str:
    # use empty string regex -> fall back to extract_result_from_boxed
    return try_evaluate_frac(
        try_evaluate_latex(
            normalize_final_answer(answer, "^$", expect_boxed=expect_boxed)
        )
    )


def compare(
    expr_1: str,
    expr_2: str,
    expr_1_expect_boxed: bool,
    expr_1_regex: str = ".*",
    expr_2_regex: str = ".*",
    exact_bracket: bool = True,
) -> tuple[str, str, bool]:
    normalized_expr_1 = eval_answer(expr_1, expect_boxed=expr_1_expect_boxed)
    normalized_expr_2 = eval_answer(expr_2, expect_boxed=False)
    result = compare_expressions_complex(
        normalized_expr_1,
        normalized_expr_2,
        expr_1_regex=expr_1_regex,
        expr_2_regex=expr_2_regex,
        exact_bracket=exact_bracket,
    )
    return normalized_expr_1, normalized_expr_2, result


def main() -> None:
    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    data = input_r.recv()

    expr_1: str = data["expr_1"]
    expr_2: str = data["expr_2"]
    expr_1_regex: str = data.get("expr_1_regex", ".*")
    expr_2_regex: str = data.get("expr_2_regex", ".*")
    exact_bracket: bool = data.get("exact_bracket", True)
    expr_1_expect_boxed: bool = data.get("expr_1_expect_boxed", True)

    try:
        normalized_expr_1, normalized_expr_2, result = compare(
            expr_1,
            expr_2,
            expr_1_expect_boxed=expr_1_expect_boxed,
            expr_1_regex=expr_1_regex,
            expr_2_regex=expr_2_regex,
            exact_bracket=exact_bracket,
        )
        output_w.send_bytes(
            json.dumps(
                {
                    "expr_1": expr_1,
                    "normalized_expr_1": normalized_expr_1,
                    "expr_2": expr_2,
                    "normalized_expr_2": normalized_expr_2,
                    "result": result,
                }
            ).encode("utf8")
        )
    except BaseException:
        output_w.send_bytes(
            json.dumps(
                {
                    "expr_1": expr_1,
                    "expr_2": expr_2,
                    "result": False,
                    "exception": traceback.format_exc(),
                }
            ).encode("utf8")
        )


def invoke_main() -> bool:
    """Invoke the main function with command line arguments."""
    expr_1 = sys.argv[1]
    expr_2 = sys.argv[2]
    normalized_expr_1, normalized_expr_2, result = compare(
        expr_1,
        expr_2,
        expr_1_expect_boxed=False,
        expr_1_regex=".*",
        expr_2_regex=".*",
        exact_bracket=True,
    )
    print(f"Expression 1: {normalized_expr_1}")
    print(f"Expression 2: {normalized_expr_2}")
    print(f"Comparison Result: {result}")
    return result


if __name__ == "__main__":
    main()
