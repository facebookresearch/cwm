# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import sys
import traceback
from itertools import product
from multiprocessing.connection import Connection

from math_verify.grader import Basic, MatrixBase, sympy_expr_eq
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)


# Adapted from https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/src/math_verify/grader.py#L743 and remove the timeout logic
def verify(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
) -> bool:
    """Verifies if the target expression matches the gold expression using multiple comparison strategies.

    This function implements a comprehensive comparison system for mathematical expressions,
    handling various types of mathematical objects (numbers, expressions, sets, matrices, etc.)
    with multiple fallback strategies.

    Note:
        - It's expected that both gold and pred has been parsed with math_verify.parse function.
        - Function is not symmetric, gold answer should be passed as gold and prediction as pred. The non-symmetric nature appears at assignment simplification and equation interval conversion.

    Args:
        gold: The reference/correct expression(s). Can be:
            - A single SymPy expression (Basic or MatrixBase)
            - A string
            - A list of any of the above
        target: The expression(s) to verify. Same types as gold.
        float_rounding: Number of decimal places to round floats to. Defaults to 6.
        numeric_precision: Number of decimal places to consider for numeric comparisons. Defaults to 15.
            - If you know the evaluated expressions will be small, you should increase this. See: https://docs.sympy.org/latest/modules/evalf.html
        strict: Whether to enforce strict comparison mode. Defaults to True.
            - In strict mode: Variables matter and sets are not comparable with tuples
            - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        timeout_seconds: Maximum time in seconds to spend on any single comparison operation.
            Defaults to 5 seconds.

    Returns:
        bool: True if target matches gold according to any of the comparison strategies,
              False otherwise.

    Comparison Strategy:
        1. String to String comparison
        2. Numeric expressions: Comparison within specified precision
        3. Symbolic equality through simplification
        4. Special handling for:
            - Relational expressions (equations/inequalities)
            - Sets and intervals
            - Matrices and vectors
            - Complex numbers
        5. Robust error handling with timeout protection

    Example:
        >>> verify(sympy.Rational(1, 3), 0.333333)  # Numeric comparison
        True
        >>> verify(sympy.Symbol('x') + 1, sympy.Symbol('y') + 1, strict=False)  # Variable matching
        True
        >>> verify(sympy.FiniteSet(1, 2), sympy.Tuple(1, 2), strict=False)  # Set-tuple comparison
        True
    """

    def compare_single_extraction(
        gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str
    ) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, Basic | MatrixBase) and isinstance(
            target, Basic | MatrixBase
        ):
            return sympy_expr_eq(
                gold, target, float_rounding, numeric_precision, strict
            )

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(
        g: Basic | MatrixBase | str, t: Basic | MatrixBase | str
    ) -> bool:
        try:
            return compare_single_extraction(g, t)
        except Exception:
            #! Do not attempt to print out the g and t during handling of exception
            # Because a) it can throw an exception itself and b) it can cause it to be stuck forever during str conversion
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(
        compare_single_extraction_wrapper(g, t) for g, t in product(gold, target)
    )


def compare(
    expr_pred: str,
    expr_gold: str,
) -> tuple[str, str, bool]:
    ground_truth_boxed = "\\boxed{" + expr_gold + "}"
    extraction_config = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ]
    target_res = get_extraction_regexes(extraction_config)
    normalized_expr_1 = extract_target_from_pred(expr_pred, target_res)
    normalized_expr_2 = extract_target_from_pred(ground_truth_boxed, target_res)
    result = verify(
        gold=normalized_expr_2,
        target=normalized_expr_1,
    )
    return normalized_expr_1, normalized_expr_2, result


def main() -> None:
    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    data = input_r.recv()

    expr_1: str = data["expr_1"]
    expr_2: str = data["expr_2"]

    try:
        normalized_expr_1, normalized_expr_2, result = compare(
            expr_1,
            expr_2,
        )
        output_w.send_bytes(
            json.dumps(
                {
                    "expr_1": expr_1,
                    "normalized_expr_1": str(normalized_expr_1),
                    "expr_2": expr_2,
                    "normalized_expr_2": str(normalized_expr_2),
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
    )
    print(f"Expression 1: {normalized_expr_1}")
    print(f"Expression 2: {normalized_expr_2}")
    print(f"Comparison Result: {result}")
    return result


if __name__ == "__main__":
    main()
