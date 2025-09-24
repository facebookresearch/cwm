# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import re
from collections.abc import Callable
from concurrent import futures
from typing import Any

import numpy as np
import sympy as sp
from sympy import Interval, Union, expand, simplify, sympify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import latex

from .unicode_to_latex import unicode_to_latex


class TimeoutException(Exception):
    pass


def run_with_timeout(func: Callable, timeout_sec: float, *args, **kwargs) -> Any:
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except futures.TimeoutError:
            raise


# from minerva
SUBSTITUTIONS = [
    ("an ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    (r"\{", "{"),
    (r"\}", "}"),
    (r"\[", "["),
    (r"\]", "]"),
]

REMOVED_EXPRESSIONS = [
    # added Feb 2025:
    # this should be done *before* removing "ft" so in particular before _normalise_result
    "\\left",
    "\\right",
    # original:
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    "°",
    "^°",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "\\displaystyle",
    "\\textstyle",
    "\\scriptstyle",
    "\\scriptscriptstyle",
]


def remove_boxed_from_solution(answer: str) -> str | None:
    box_start = "\\boxed{"
    # format is `\\boxed <value>$` or `\\boxed{<value>}`, with potential white spaces framing `<value>`
    start = answer.rfind(box_start)

    i = start + len(box_start)
    open_braces = 0
    while i < len(answer):
        if answer[i] == "{":
            open_braces += 1
        elif answer[i] == "}":
            open_braces -= 1
        if open_braces < 0:
            break
        i += 1
    new_answer = answer[:start] + answer[start + len(box_start) : i] + answer[i + 1 :]
    return new_answer


def extract_result_from_boxed(answer: str) -> str:
    box_start = "\\boxed"
    # format is `\\boxed <value>$` or `\\boxed{<value>}`, with potential white spaces framing `<value>`
    start = answer.rfind(box_start)
    if start < 0:
        return ""
    answer = answer[start + len(box_start) :].strip()
    ends_with_curly = answer.startswith("{")
    i = 0
    open_braces = 0
    while i < len(answer):
        if answer[i] == "{":
            open_braces += 1
        elif answer[i] == "}":
            open_braces -= 1
        if open_braces == 0:
            if ends_with_curly:
                answer = answer[: i + 1].strip()
                break
            elif answer[i] == "$":
                answer = answer[:i].strip()
                break
        i += 1
    else:
        return ""
    # remove extra curly braces
    while True:
        if answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1].strip()
        else:
            break
    return answer


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    # Use regex to find all simple fraction patterns like "number/number"
    # This handles cases where fractions appear within larger expressions
    def replace_fraction(match: re.Match) -> str:
        a = match.group(1)
        b = match.group(2)
        try:
            ia = int(a)
            ib = int(b)
            return f"\\frac{{{ia}}}{{{ib}}}"
        except ValueError:
            return match.group(0)  # Return original if not integers

    # Pattern matches: optional minus sign, digits, slash, optional minus sign, digits
    # Uses word boundaries to avoid matching parts of larger numbers
    pattern = r"(-?\d+)/(-?\d+)"
    result = re.sub(pattern, replace_fraction, string)
    return result


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    try:
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string
    except AssertionError:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _normalise_result(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    if "pmatrix" not in string and "bmatrix" not in string and "matrix" not in string:
        string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("cfrac", "frac")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # replace leqslant with leq
    string = string.replace("leqslant", "leq")
    string = string.replace("geqslant", "geq")

    # # remove \left and \right
    # string = string.replace("\\left", "")
    # string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    string = string.split("=")[-1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# from minerva paper + _normalise_result from xavierm
def normalize_final_answer(
    final_answer: str, regex_pattern: str, expect_boxed: bool = True
) -> str:
    """Extract and normalize a final answer to a quantitative reasoning question."""
    match = re.findall(regex_pattern, final_answer)
    extraction: str
    if len(match) > 0:
        extraction = match[0]
    else:
        extraction = extract_result_from_boxed(final_answer)

    if expect_boxed:
        if len(extraction) == 0:
            return final_answer
        else:
            final_answer = extraction

    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    # If the final answer is a single letter in parentheses, remove the parentheses
    # Example: (a) -> a (but not (ab) -> ab)
    if re.match(r"\([a-zA-Z]\)", final_answer):
        final_answer = final_answer[1]
    return _normalise_result(final_answer)


def get_list_variants_math_result(result: str) -> list:
    results_with_variants = [result]
    if result.isdigit():
        results_with_variants.append(result + ".")
        results_with_variants.append(result + ".0")
        results_with_variants.append(result + ".0.")
        results_with_variants.append(result + ".00")
        results_with_variants.append(result + ".00.")

        # add split by comma of thousands
        # NOTE: Unicode characters like 3² returns True for `isdigit`
        #       but fails to convert to int.
        try:
            value = int(result)
            results_with_variants.append(f"{value:,}")
            results_with_variants.append(f"{value:,}.")
            results_with_variants.append(f"{value:,}.0")
            results_with_variants.append(f"{value:,}.0.")
            results_with_variants.append(f"{value:,}.00")
            results_with_variants.append(f"{value:,}.00.")
        except ValueError:
            pass
    return results_with_variants


def try_evaluate_frac(expression: str, fmt: str = "0.4e") -> str:
    if isinstance(expression, float):
        return expression
    new_expression = expression
    regex = re.compile(r"\\frac{([^}]+)}{([^}]+)}")
    for match in re.finditer(regex, expression):
        try:
            value = float(match.group(1)) / float(match.group(2))
            new_expression = new_expression.replace(
                match.group(),
                f"{{value:{fmt}}}".format(value=value),
                1,
            )
        except Exception:
            continue
    return new_expression


def try_evaluate_latex(expression: str, fmt: str = "0.4e") -> str:
    try:

        def call():
            value = parse_latex(expression).evalf()  # type: ignore
            return f"{{value:{fmt}}}".format(value=value)

        return run_with_timeout(call, timeout_sec=5)
    except ImportError:
        raise
    except Exception:
        return expression


def try_evaluate_expr(expression: str, fmt: str = "0.4e") -> str:
    """Tries to evaluate a regular math string expressions like 5/11, 33*3..."""
    try:

        def call():
            value = parse_expr(expression).evalf()
            return f"{{value:{fmt}}}".format(value=value)

        return run_with_timeout(call, timeout_sec=5)
    except ImportError:
        raise
    except Exception:
        return expression


def try_expr_to_latex(expression: str) -> str:
    """Takes a math expression like 5/11 and converts it to LaTex \\frac{5}{11}."""
    try:

        def call():
            parsed = parse_expr(expression)
            return latex(parsed)

        return run_with_timeout(call, timeout_sec=5)
    except ImportError:
        raise
    except Exception:
        return expression


# new edge cases addressed for eval with code execution


def convert_scientific_notation_to_float(expr_str: str, fmt: str = ".5g") -> str:
    try:
        # Try to convert the string to a float
        float_value = float(expr_str)
        # Format the float value to 5 significant digits
        return f"{{value:{fmt}}}".format(value=float_value)
    except Exception:
        # If conversion fails, return the original string
        return expr_str


def latex_to_float(latex_str: str, fmt: str | None = None) -> str:
    # Replace all double backslashes with single ones for consistent processing
    formatted_latex = latex_str.replace("\\\\", "\\")

    # Define replacements for LaTeX to SymPy syntax
    def replace_frac(match: re.Match) -> str:
        # Handle \pi and other variables/constants in fractions
        numerator = match.group(1).replace("\\pi", "pi")
        denominator = match.group(2).replace("\\pi", "pi")
        return f"({numerator})/({denominator})"

    def replace_sqrt(match: re.Match) -> str:
        # Process \pi and other variables/constants in square roots
        argument = match.group(1).replace("\\pi", "pi")
        return f"sp.sqrt({argument})"

    # Correctly parse expressions starting with a fraction or mixed integer and fraction
    match = re.match(r"^(\d+)\\frac{([^{}]*)}{([^{}]*)}$", formatted_latex)
    if match:
        integer = match.group(1)
        numerator = match.group(2).replace("\\pi", "sp.pi")
        denominator = match.group(3).replace("\\pi", "sp.pi")
        formatted_latex = f"({integer})*(({numerator})/({denominator}))"  # Use addition to combine integer and fraction

    # Handle standalone fraction expressions correctly
    if re.match(r"^\\frac", formatted_latex):
        formatted_latex = "1*(" + formatted_latex + ")"

    # Recursively replace fractions and square roots
    while True:
        new_latex = re.sub(r"\\frac{([^{}]*)}{([^{}]*)}", replace_frac, formatted_latex)
        new_latex = re.sub(r"\\sqrt{([^{}]*)}", replace_sqrt, new_latex)
        if new_latex == formatted_latex:
            break
        formatted_latex = new_latex

    # Replace "sp.pi" with "pi" after all substitutions
    formatted_latex = formatted_latex.replace("sp.pi", "pi")

    # Convert the string to a SymPy expression
    try:
        sp.var("pi")  # Ensure 'pi' is defined as a symbol in SymPy
        sympy_expr = sp.sympify(formatted_latex, locals={"sqrt": sp.sqrt})
    except Exception:
        return latex_str

    # Evaluate the SymPy expression to a floating-point number
    try:
        float_result = float(sympy_expr.evalf())
        rounded_result = round(float_result, 3)
        if fmt is None:
            return str(rounded_result)
        return f"{{value:{fmt}}}".format(value=rounded_result)
    except Exception:
        return latex_str


def parse_complex(expr_str: str) -> str:
    # Replace 'I' with 'i' if 'I' is used for the imaginary unit
    expr_str = expr_str.replace("I", "i")

    try:
        # Parse the string using SymPy
        expr = sp.sympify(expr_str)

        # Check if the expression is a complex number
        if isinstance(expr, sp.Add) and any(isinstance(arg, sp.I) for arg in expr.args):
            return str(expr)
        else:
            return expr_str
    except Exception:
        return expr_str


def process_polynomial(expression: str) -> str:
    try:
        expression = expression.replace(" ", "").strip()
        if "^" in expression:
            expression = expression.replace("^", "**")

        expression = re.sub(r"(\d|\))([a-zA-Z\(])", r"\1*\2", expression)
        expression = re.sub(r"(\))(\()", r"\1*\2", expression)
        expression = re.sub(r"(pi|e)(\(|[a-zA-Z])", r"\1*\2", expression)
        # Parse the expression to a sympy object
        parsed_expr = sympify(expression.replace("^", "**"))

        # Simplify the expression
        simplified_expr = simplify(parsed_expr)
        expanded_expr = expand(simplified_expr)

        # Return the simplified expression
        return str(expanded_expr)
    except Exception:
        # Handle any exceptions that might occur during parsing or simplification
        return expression


def parse_set(set_str: str) -> str:
    # Handle LaTeX constants and common symbols
    set_str = (
        set_str.replace("\\infty", "oo")
        .replace("\\pi", "pi")  # Convert LaTeX \pi to sympy pi
        .replace("\\cup", "U")
        .replace("\\le", "")
        .replace("\\iny", "oo")
    )  # Assume '\\le' is a mistake
    try:
        # Convert LaTeX fractions to SymPy fractions and evaluate them
        set_str = re.sub(
            r"\\frac\{([^}]*)\}\{([^}]*)\}",
            lambda m: str(simplify(f"({m.group(1)})/({m.group(2)})")),
            set_str,
        )
    except Exception:
        return set_str

    # Find all interval expressions
    intervals = re.findall(r"\((.*?,.*?)\)", set_str)

    # If no intervals are found, return None or an appropriate error/message
    if not intervals:
        return set_str

    # Initialize set object for Union of intervals
    set_obj = None

    try:
        for interval in intervals:
            start, end = map(str.strip, interval.split(","))
            start = "-oo" if start == "-\\infty" else simplify(start)
            end = "oo" if end == "\\infty" else simplify(end)

            # Create Interval objects, assuming exclusive bounds for simplicity
            interval_obj = Interval(sympify(start), sympify(end), True, True)

            if set_obj is None:
                set_obj = interval_obj
            else:
                set_obj = Union(set_obj, interval_obj)

        return str(set_obj)

    except Exception:
        # Return error details in case of any exceptions during parsing or processing
        return set_str


FLOAT_FORMATS = [
    # e
    ".4e",
    ".5e",
    # g
    ".5g",
]


def evaluate_with_time_limit(
    func: Callable, e: str, fmt: str | None = None, seconds: float = 5
) -> Any:
    try:
        # with time_limit(seconds=seconds):
        def call():
            return func(e, fmt=fmt) if fmt else func(e)

        return run_with_timeout(call, timeout_sec=seconds)
    except Exception:
        return None


def get_evaluated_variants(expr: str, expr_regex: str = ".*") -> set[str]:
    """Get a list of evaluated variants of a math expression.

    Uses all the tricks in the book to evaluate a math expression.

    Args:
        expr: The math expression to evaluate.
        expr_regex: The regex pattern to extract the value from the
            final expression. Default: Use the whole expression.

    Returns:
        A list of evaluated variants of the math expression.
    """
    results = {expr}
    try:
        # with time_limit(seconds=10):
        def call():
            return normalize_final_answer(expr, expr_regex)

        expr = run_with_timeout(call, timeout_sec=10)
        results.add(expr)
    except Exception:
        pass

    # strip percentage
    expr = expr.rstrip("%")
    results.add(expr)
    results.add(unicode_to_latex(expr))

    evaluation_functions: list[tuple[Callable, bool]] = [
        (try_evaluate_latex, True),  # latex evaluate
        (try_evaluate_expr, True),  # sympy evaluate
        (try_evaluate_frac, True),  # frac evaluate
        (process_polynomial, False),  # polynomial evaluate
    ]

    for fmt in FLOAT_FORMATS:
        e = expr

        for func, use_fmt in evaluation_functions:
            result = evaluate_with_time_limit(func, e, fmt if use_fmt else None)
            if result is not None:
                e = result
                results.add(e)

    return results


def compare_expressions(
    expr_1: str,
    expr_2: str,
    expr_1_regex: str = ".*",
    expr_2_regex: str = ".*",
) -> bool:
    """Compare two math expressions for similarity.

    Uses all the tricks in the book to evaluate a math expression and compare
    them for similarity.

    Example:
        >>> compare_expressions("5/11", "0.45454545454545453")
        True
        >>> compare_expressions("5/11", "0.454")
        True
        >>> compare_expressions("5/11", "0.455")
        True
        >>> compare_expressions("5/11", "0.456")
        False
        >>> compare_expressions("2*4/11", "2 \\cdot \\frac{4}{11}")
        True
        >>> compare_expressions("2 - 4/11", "2 - \\frac{4}{11}")
        True
        >>> compare_expressions("(2 - 4) / 11", "2 - \\frac{4}{11}")
        False

    Args:
        expr_1: The first math expression to compare.
        expr_2: The second math expression to compare.
        expr_1_regex: The regex pattern to extract the value from the
            first expression answer. Default: Use the whole expression.
        expr_2_regex: The regex pattern to extract the value from the
            second expression answer. Default: Use the whole expression.

    Returns:
        True if the expressions are similar, False otherwise.
    """
    expr_1_set = get_evaluated_variants(expr_1, expr_1_regex)
    expr_2_set = get_evaluated_variants(expr_2, expr_2_regex)
    return len(expr_1_set.intersection(expr_2_set)) > 0


# -----------------------------------------------------------------------------------
# Compare expressions Ver.2
# -----------------------------------------------------------------------------------
def process_pi(expr: str) -> str:
    """
    Transforms pi to float number rounded by 4 digits after the point
    """
    # NOTE: problem with consecutive \\pi
    # multiple pi
    for n, e, _, s in re.findall(
        r"(\d+\.\d+|\d+)(e[+-]\d+)?(\\pi)([^a-zA-Z\d]|$)", expr
    ):
        nf = float(n + e)
        expr = expr.replace(n + e + "\\pi" + s, str(round(nf * np.pi, 4)) + s)
    # one pi
    for _, s in re.findall(r"(\\pi)([^a-zA-Z\d]|$)", expr):
        expr = expr.replace("\\pi" + s, "3.1415" + s)
    return expr


def process_sqrt(expr: str) -> str:
    """
    Calculates square root and returns float number
    """
    # number + sqrt
    for n1, n2 in re.findall(r"(\d+\.\d+|\d+)\\sqrt{(\d+\.\d+|\d+)}", expr):
        expr = expr.replace(
            n1 + "\\sqrt{" + n2 + "}", str(float(n1) * float(n2) ** 0.5)
        )
    for n1, n2 in re.findall(r"(\d+\.\d+|\d+)\\sqrt(\d+\.\d+|\d+)", expr):
        expr = expr.replace(n1 + "\\sqrt" + n2, str(float(n1) * float(n2) ** 0.5))
    # just sqrt
    for n in re.findall(r"\\sqrt{(\d+\.\d+|\d+)}", expr):
        expr = expr.replace("\\sqrt{" + n + "}", str(float(n) ** 0.5))
    for n in re.findall(r"\\sqrt(\d+\.\d+|\d+)", expr):
        expr = expr.replace("\\sqrt" + n, str(float(n) ** 0.5))
    return expr


def process_infinity(expr: str) -> str:
    """
    Generalizes different spellings of infinity to \\iny
    """
    expr = expr.replace("\\infty", "\\iny")
    if expr[-4:] in (" inf", ",inf"):
        expr = expr[:-3] + "\\iny"
    expr = expr.replace(" inf)", " \\iny)")
    expr = expr.replace(",inf)", ",\\iny)")
    if expr[:4] in ("inf ", "inf,"):
        expr = "\\iny" + expr[3:]
    expr = expr.replace("(inf ", "(\\iny ")
    expr = expr.replace("(inf,", "(\\iny,")
    if expr == "inf":
        expr = "\\iny"
    return expr


def process_symbols(expr: str) -> str:
    expr = process_pi(expr)
    expr = process_sqrt(expr)
    expr = process_infinity(expr)
    return expr


def compare_sets(set_1: set, set_2: set) -> bool:
    """
    Compare two sets of expressions if there are two same expressions or close enough floats.
    """
    if len(set_1.intersection(set_2)) > 0:
        return True

    for x in set_1:
        for y in set_2:
            # Handle different numeric types
            if isinstance(x, int | float) and isinstance(y, int | float):
                if math.isclose(x, y):
                    return True
            elif isinstance(x, complex) or isinstance(y, complex):
                # Convert to complex if needed
                x_complex = x if isinstance(x, complex) else complex(x, 0)
                y_complex = y if isinstance(y, complex) else complex(y, 0)

                # Check both real and imaginary parts
                if math.isclose(x_complex.real, y_complex.real) and math.isclose(
                    x_complex.imag, y_complex.imag
                ):
                    return True
    return False


def try_add(answers: set, expr: str) -> set:
    """
    Add an expression to the set of answers.
    If it can be represented as float, convert it to float before adding.
    It is necessary for more convenient comparison of answers
    """
    try:
        answers.add(float(expr))
    except Exception:
        answers.add(expr)
    return answers


THE_FINAL_ANSWER_REG = re.compile(
    r"""
    ^                   # Start of string
    (?:the\s+)?         # Optional 'the' followed by whitespace
    final               # Literal 'final'
    \s+                 # One or more whitespace characters
    answer              # Literal 'answer'
    (?:\s+is)?          # Optional 'is' with preceding whitespace
    \s*                 # Zero or more whitespace characters
    :?                  # Optional colon
    \s*                 # Zero or more whitespace characters
""",
    re.VERBOSE | re.IGNORECASE,
)


def compare_expressions_v2(
    expr_1: str,
    expr_2: str,
    expr_1_regex: str = ".*",
    expr_2_regex: str = ".*",
) -> bool:
    """Improved 'compare_expressions' version. Added:
    - process symbols (including pi, sqrt, infinity)
    - speed up comparisons for different types of expressions
    - compare float numbers if they are close enough
    - remove redundant "{}", prefixes
    - expand expressions (limited work)

    It is recommended to run this function as part of 'compare_expressions_complex' function.

    Args:
        expr_1: First expression.
        expr_2: Second expression.
        expr_1_regex: Regular expression for the first expression.
        expr_2_regex: Regular expression for the second expression.
    """
    # Remove prefixes with ""
    expr_1 = THE_FINAL_ANSWER_REG.sub("", expr_1).strip()
    expr_2 = THE_FINAL_ANSWER_REG.sub("", expr_2).strip()

    # fast comparison of numeric values
    try:
        ef1, ef2 = float(expr_1), float(expr_2)
        return compare_sets({ef1}, {ef2})
    except Exception:
        pass

    # rstrip of %
    answers_1 = {expr_1}
    expr_1 = expr_1.rstrip("%")
    answers_1 = try_add(answers_1, expr_1)

    answers_2 = {expr_2}
    expr_2 = expr_2.rstrip("%")
    answers_2 = try_add(answers_2, expr_2)

    if compare_sets(answers_1, answers_2):
        return True

    # remove redundant "{}"
    expr_1 = expr_1.replace("{}", "")
    answers_1 = try_add(answers_1, expr_1)

    expr_2 = expr_2.replace("{}", "")
    answers_2 = try_add(answers_2, expr_2)

    if compare_sets(answers_1, answers_2):
        return True

    # Normalize the expressions
    try:

        def call():
            return normalize_final_answer(expr_1, expr_1_regex)

        expr_1 = run_with_timeout(call, timeout_sec=10)
        answers_1 = try_add(answers_1, expr_1)
    except Exception:
        pass
    try:

        def call():
            return normalize_final_answer(expr_2, expr_2_regex)

        expr_2 = run_with_timeout(call, timeout_sec=10)
        answers_2 = try_add(answers_2, expr_2)
    except Exception:
        pass

    if compare_sets(answers_1, answers_2):
        return True

    # add unicode to latex
    expr_1 = unicode_to_latex(expr_1)
    answers_1 = try_add(answers_1, expr_1)
    expr_2 = unicode_to_latex(expr_2)
    answers_2 = try_add(answers_2, expr_2)
    if compare_sets(answers_1, answers_2):
        return True

    # parse latex and expand
    try:
        e1 = parse_latex(expr_1)
        answers_1 = try_add(answers_1, e1)
        e1 = expand(e1)
        answers_1 = try_add(answers_1, e1)
    except Exception:
        pass

    try:
        e2 = parse_latex(expr_2)
        answers_2 = try_add(answers_2, e2)
        e2 = expand(e2)
        answers_2 = try_add(answers_2, e2)
    except Exception:
        pass

    if compare_sets(answers_1, answers_2):
        return True

    # process symbols
    expr_1 = process_symbols(expr_1)
    answers_1 = try_add(answers_1, expr_1)
    expr_2 = process_symbols(expr_2)
    answers_2 = try_add(answers_2, expr_2)
    if compare_sets(answers_1, answers_2):
        return True

    # check lower answers
    answers_1 = try_add(answers_1, expr_1.lower())
    answers_2 = try_add(answers_2, expr_2.lower())
    if compare_sets(answers_1, answers_2):
        return True

    for fmt in FLOAT_FORMATS:
        e1 = expr_1
        e2 = expr_2

        # Try evaluating fractions
        try:

            def call_try_evaluate_frac_e1(e1: str = e1, fmt: str = fmt):
                return try_evaluate_frac(e1, fmt=fmt)

            e1_f = run_with_timeout(call_try_evaluate_frac_e1, timeout_sec=10)
            answers_1 = try_add(answers_1, e1_f)
        except Exception:
            e1_f = None
        try:

            def call_call_try_evaluate_frac_e2(e2: str = e2, fmt: str = fmt):
                return try_evaluate_frac(e2, fmt=fmt)

            e2_f = run_with_timeout(call_call_try_evaluate_frac_e2, timeout_sec=10)
            answers_2 = try_add(answers_2, e2_f)
        except Exception:
            e2_f = None
        if compare_sets(answers_1, answers_2):
            return True

        # Try adding the processes polynomials
        try:

            def call_process_polynomial_e1(e1: str = e1):
                return process_polynomial(e1)

            e1_p = run_with_timeout(call_process_polynomial_e1, timeout_sec=10)
            answers_1 = try_add(answers_1, e1_p)
        except Exception:
            e1_p = None
        try:

            def call_process_polynomial_e2(e2: str = e2):
                return process_polynomial(e2)

            e2_p = run_with_timeout(call_process_polynomial_e2, timeout_sec=10)
            answers_2 = try_add(answers_2, e2_p)
        except Exception:
            e2_p = None
        if compare_sets(answers_1, answers_2):
            return True

        # Try evaluating sympy expressions
        for num1, num2 in [(e1, e2), (e1_f, e2_f), (e1_p, e2_p)]:
            answers_1 = try_add(answers_1, try_evaluate_expr(num1, fmt=fmt))
            answers_2 = try_add(answers_2, try_evaluate_expr(num2, fmt=fmt))
            if compare_sets(answers_1, answers_2):
                return True

        # last resort, try evaluating latex
        for num1, num2 in [(e1, e2), (e1_p, e2_p)]:
            answers_1 = try_add(answers_1, try_evaluate_latex(num1, fmt=fmt))
            answers_2 = try_add(answers_2, try_evaluate_latex(num2, fmt=fmt))
            if compare_sets(answers_1, answers_2):
                return True

        # remove parenthesis
        for a in list(answers_1):
            if isinstance(a, str) and len(a) >= 3 and a[0] == "(" and a[-1] == ")":
                answers_1 = try_add(answers_1, a[1:-1])
        for a in list(answers_2):
            if isinstance(a, str) and len(a) >= 3 and a[0] == "(" and a[-1] == ")":
                answers_2 = try_add(answers_2, a[1:-1])
        if compare_sets(answers_1, answers_2):
            return True

    return compare_sets(answers_1, answers_2)


def process_range(expr: str, exact_bracket: bool = True) -> tuple:
    if not exact_bracket:
        expr = expr.replace("[", "(")
        expr = expr.replace("]", ")")
    braces = ""
    try:
        expr = expr.replace("\\le(", "(")
        expr = expr.replace("\\le[", "[")
        while expr[0] in "[(" and expr[-1] in "])":
            cnt = 1
            for s in expr[1:-1]:
                if s in "[(":
                    cnt += 1
                if s in "])":
                    cnt -= 1
                if cnt == 0:
                    return braces, [expr]
            braces = braces + expr[0] + expr[-1]
            expr = expr[1:-1]
        return braces, expr.split(",")
    except Exception:
        return "", [expr]


def process_matrix(expr: str) -> tuple:
    if expr.startswith("\\begin{pmatrix}") and expr.endswith("\\end{pmatrix}"):
        matrix_type = "pmatrix"
    elif expr.startswith("\\begin{bmatrix}") and expr.endswith("\\end{bmatrix}"):
        matrix_type = "bmatrix"
    else:
        return None, [expr]
    output1 = expr[15:-13].replace(r"\ ", "").replace(" ", "").split("\\")
    output2 = []
    for line in output1:
        output2.extend(line.split("&"))
    return matrix_type, output2


def compare_expressions_complex(
    expr_1: str,
    expr_2: str,
    expr_1_regex: str = ".*",
    expr_2_regex: str = ".*",
    exact_bracket: bool = True,
) -> bool:
    """
    Parsing of complex expressions:
    - matrices
    - intervals (braces)

    When complex object is detected, parts of it parsed separately one by one using 'compare_expressions_v2' function.

    """
    # TODO: apply recursive function and include \cup
    # check matrix parsing
    matrix_type_1, exprs_1 = process_matrix(expr_1)
    matrix_type_2, exprs_2 = process_matrix(expr_2)
    if (matrix_type_1 is None and matrix_type_2 is not None) or (
        matrix_type_1 is not None and matrix_type_2 is None
    ):
        return False
    if (
        matrix_type_1 is not None
        and matrix_type_2 is not None
        and matrix_type_1 == matrix_type_2
    ):
        if len(exprs_1) != len(exprs_2):
            return False
        for i in range(len(exprs_1)):
            c = compare_expressions_v2(
                expr_1=exprs_1[i],
                expr_2=exprs_2[i],
                expr_1_regex=expr_1_regex,
                expr_2_regex=expr_2_regex,
            )
            if c is False:
                return False
        return True

    # check braces parsing
    braces_1, exprs_1 = process_range(expr_1, exact_bracket)
    braces_2, exprs_2 = process_range(expr_2, exact_bracket)

    # braces makes sense only if we are talking about intervals which contains exactly 2 numbers
    if len(exprs_1) != len(exprs_2) or (
        exact_bracket and len(exprs_1) == 2 and braces_1 != braces_2
    ):
        return False
    for i in range(len(exprs_1)):
        c = compare_expressions_v2(
            expr_1=exprs_1[i],
            expr_2=exprs_2[i],
            expr_1_regex=expr_1_regex,
            expr_2_regex=expr_2_regex,
        )
        if c is False:
            return False
    return True
