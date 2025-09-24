# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Utility to create an aggregate outcomes for envs. These are meant to be passed
as the `Transition(outcomes=...)` argument.

Outcomes are packaged in a dict in general, but these utilities provide a simple
grammar for constructing outcome collections (and aggregating results from
multiple terminal transitions).

To build a set of outcomes, chain them like you would merge dictionaries by doing e.g.

passing(False) | compiling(True) | parsing(True).

Currently we support `float`, `bool`, and `int` for aggregation.
"""

import math
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from cwm.text.tokenizers import InstructTokenizer


def outcome(k: str, v: Any) -> dict[str, Any]:
    return {k: v}


def passing(passed: bool) -> dict[str, bool]:
    return outcome("pass", passed)


def successful_pass(outcomes: dict[str, Any]) -> bool:
    return outcomes["pass"]


def public_passing(passed: bool) -> dict[str, bool]:
    return outcome("public_pass", passed)


def passed() -> dict[str, bool]:
    return passing(True)


def failed() -> dict[str, bool]:
    return passing(False)


def compiling(compiled: bool) -> dict[str, bool]:
    return outcome("compile", compiled)


def successful_compile(outcomes: dict[str, Any]) -> bool:
    return outcomes["compile"]


def parsing(parsed: bool) -> dict[str, bool]:
    return outcome("parse", parsed)


def n_backtrack(s: str) -> int:
    pattern = (
        r"\bwait\b|"
        r"\bhold on\b|"
        r"\blet me (?:check|verify|confirm)\b|"
        r"\blet\'s (?:check|verify|confirm)\b|"
        r"\bI need to (?:check|verify|confirm)\b|"
        r"\bbut (?:perhaps|maybe)\b|"
        r"\balternatively\b|"
        r"\bseems incorrect\b|"
        r"\bright\?"
        r"\boh no\b|"
        r"\boh (?:perhaps|maybe)\b"
    )
    matches = re.findall(pattern, s, re.IGNORECASE)
    return len(matches)


def n_insight(s: str) -> int:
    pattern = r"\baha\b|" r"\bI (?:see|understand)\b"
    matches = re.findall(pattern, s, re.IGNORECASE)
    return len(matches)


def n_tokens(s: str, tokenizer: InstructTokenizer) -> int:
    return len(tokenizer.encode(s))


def thought_info(thought: str | None, tokenizer: InstructTokenizer) -> dict[str, int]:
    thought = thought or ""  # replace None[]
    return {
        "n_backtrack": n_backtrack(thought),
        "n_insight": n_insight(thought),
        "reasoning_len": len(thought),
        "n_tokens": n_tokens(thought, tokenizer),
    }


def successful_parse(outcomes: dict[str, Any]) -> bool:
    return outcomes["parse"]


def reasoning_found(found: bool) -> dict[str, bool]:
    return outcome("reasoning_found", found)


def n_reasoning_tokens(length: int) -> dict[str, int]:
    """Stores the length of the reasoning, 0 means no reasoning"""
    return outcome("n_reasoning_tokens", length)


def answer(answer: str | None) -> dict[str, str | None]:
    """Stores the answer (for majority voting), None means no answer"""
    return outcome("answer", answer)


@dataclass
class AnswerData:
    pass_value: bool
    answer: str | None = None
    length: int | float = 0


def get_combinations_with_limit(
    iterable: Sequence, k: int, max_comb: int = 100000
) -> Iterator[tuple]:
    total_combinations = math.comb(len(iterable), k)
    if total_combinations <= max_comb:
        yield from combinations(iterable, k)
    else:
        for _ in range(max_comb):
            yield tuple(random.sample(iterable, k))


def aggregate_outcomes(
    outcomes: list[dict[str, Any]],
) -> dict[str, float]:
    count = len(outcomes)
    keys = set(k for bo in outcomes for k in bo)
    aggregated = {}
    for k in keys:
        for bo in outcomes:
            if k not in bo:
                raise ValueError(
                    f"Key {k} not present in {bo} but recorded in other outcomes. Please check consistency."
                )
            tp = type(bo[k])
            assert tp in {float, bool, int}, f"Outcome type {tp} not yet supported"
        aggregated[k] = sum(float(bo[k]) for bo in outcomes) / count
    return aggregated


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Codex (https://arxiv.org/abs/2107.03374):
    $ E_{x_i \sim p, i \leq k}[ max x_i ] $
    estimated from n samples.
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def rank_score_at_k(n: int, k: int, scores_sorted: list[bool] | list[int]) -> float:
    """
    Unbiased rank-score@k from https://arxiv.org/abs/2404.00725:
    $ E_{(x_i, r_i) \sim p, i \leq k}[ x_{argmin_i r_i}] $
    estimated from n >= k samples, where r_i is the rank.
    :param n: total number of samples
    :param k: k in rank-score@k
    :param scores_sorted: a list of pass/scores. The list is sorted by the ranks assigned to examples by a ranker.
    """
    numerator_sum = 0
    for i in range(1, n - k + 2):
        numerator_sum += math.comb(n - i, k - 1) * scores_sorted[i - 1]
    score = numerator_sum / math.comb(n, k)
    return score


def short_1_at_k(
    n: int, k: int, pass_length_tup: list[tuple[bool, int | float]]
) -> float:
    """
    Implements the short-1@k unbiased evaluation from https://arxiv.org/abs/2505.17813:
    rank-score@k where the rank is given by the answer/reasoning length.
    :param n: total number of samples
    :param k: k in short-1@k
    :param pass_length_tup: a list of (pass, length) tuples.
    """
    sorted_list = sorted(pass_length_tup, key=lambda x: (x[1] <= 0, x[1]))
    pass_sorted = [a[0] for a in sorted_list]
    return rank_score_at_k(n, k, pass_sorted)


def _short_m_at_k_impl(
    n: int, k: int, m: int, answer_data_lst: list[AnswerData]
) -> float:
    total_pass_val = 0.0
    num_subsets = 0

    for subset in get_combinations_with_limit(answer_data_lst, k):
        num_subsets += 1
        sorted_subset = sorted(
            subset,
            key=lambda answer_data: (answer_data.length <= 0, answer_data.length),
        )[:m]

        answer_counts = {}
        for answer_data in sorted_subset:
            if answer_data.answer not in answer_counts:
                # we store -length in order to extract the max element later
                answer_counts[answer_data.answer] = [0, float("-inf"), 0]
                if answer_data.answer is None:
                    answer_counts[answer_data.answer][0] = float("-inf")
            answer_counts[answer_data.answer][0] = (
                answer_counts[answer_data.answer][0] + 1
            )
            answer_counts[answer_data.answer][1] = max(
                answer_counts[answer_data.answer][1], -answer_data.length
            )
            answer_counts[answer_data.answer][2] = (
                answer_counts[answer_data.answer][2] + answer_data.pass_value
            )

        max_item_count, max_item_len, max_item_sum_pass = max(answer_counts.values())
        total_pass_val += max_item_sum_pass / max_item_count

    return total_pass_val / num_subsets


def short_m_at_k(n: int, k: int, m: int, answer_data_lst: list[AnswerData]) -> float:
    """
    Implements the short-m@k from https://arxiv.org/abs/2505.17813.
    :param n: total number of samples
    :param k: k in majority@k or short-m@k
    :param m: m in short-m@k
    :param answer_data_lst: a list of AnswerData objects.
    """
    if m == 1:  # for short1@k we have an efficient implementation
        pass_length_tup = [
            (
                answer_data.pass_value,
                0 if answer_data.answer is None else answer_data.length,
            )
            for answer_data in answer_data_lst
        ]  # clean answers as shor1@k does not perform any majority
        return short_1_at_k(n, k, pass_length_tup)

    return _short_m_at_k_impl(n, k, m, answer_data_lst)


def majority_at_k(n: int, k: int, answer_data_lst: list[AnswerData]) -> float:
    """
    Implements the majority@k.
    :param n: total number of samples
    :param k: k in majority@k or short-m@k
    :param answer_data_lst: a list of AnswerData objects.
    """
    # adds artificial lengths for compatibility
    for answer_data in answer_data_lst:
        answer_data.length = random.random()
    # uses _short_m_at_k_impl with random lengths, and m=k
    return _short_m_at_k_impl(n, k, k, answer_data_lst)


def get_ks(aggregation_spec: dict[str, list[str]]) -> set[int]:
    return {
        int(agg[1:])
        for aggregation in aggregation_spec.values()
        for agg in aggregation
        if agg.startswith("@")
    }


def infer_n_samples(aggregation_spec: dict[str, list[str]]) -> int:
    ks = get_ks(aggregation_spec)
    if ks:
        max_k = max(ks)
        if max_k > 1:
            return 2 * max_k
        # 1 sample is enough for @1
    return 1


def validate_aggregation_spec(
    aggregation_spec: dict[str, list[str]], n_samples: int
) -> None:
    """
    An aggregation_spec is a specification of how to aggregate specific outcomes.
    Multiple aggregates can be specified for each outcome, such as:
        {
            "pass": ["@1", "mean"],
            "compile": ["@5"],
        }
    and so on.

    Supported aggregates are "@N" with N any integer, and "mean". Note that
    (a) for "@N", at least N samples must be provided.
        If using multiple different @N aggregates, at least max(N) samples are needed.
    (b) "@N" is only valid for boolean outcomes.
        This is not validated here, since we do not yet know at this stage the outcome types,
        but the aggregation function will raise an error if misused.
    """
    ks = get_ks(aggregation_spec)
    if ks:
        assert (
            n_samples >= max(ks)
        ), f"To run @k evaluation with ks = {sorted(ks)}, at least {max(ks)} samples_per_prompt must be passed. n_samples == {n_samples}"


def aggregate_outcomes_from_spec(
    aggregation_spec: dict[str, list[str]],
    sequences_of_samples: list[list[dict[str, Any]]],
) -> dict[str, float]:
    """
    Aggregates outcomes based on the provided spec, for sequences of multiple samples per item.
    Only boolean types can be aggregated with pass@k logic.
    Aggregate logic from spec is only used to aggregate across samples, aggregation across datapoints (first dimension of the sequences_of_samples sequence) is always averaged.
    Unspecified metrics default to mean.
    """

    outcome_keys = [key for key in sequences_of_samples[0][0].keys()]
    n_samples = len(sequences_of_samples[0])

    def check_outcomes_type(samples: list[dict[str, Any]], key: str, T: type | Any):
        for sample in samples:
            if not isinstance(sample[key], T):
                raise ValueError(
                    f"Type for outcome {key} in sample {sample} is {type(sample[key])}, expected {T}"
                )

    # Validate samples counts
    for samples in sequences_of_samples:
        assert n_samples == len(samples), "Inconsistent number of samples per example"

    # Compute pass_at_k or average depending on outcome type for each item
    aggregate_metrics = []
    for samples in sequences_of_samples:
        metrics = {}
        for key in outcome_keys:
            if key == "answer" and key not in aggregation_spec:
                continue  # no default aggregation for answers
            aggregates = aggregation_spec.get(key, ["mean"])
            for aggregate in aggregates:
                if aggregate.startswith("@"):
                    check_outcomes_type(samples, key, bool)
                    k = int(aggregate[1:])
                    metrics[f"{key}@{k}"] = pass_at_k(
                        n_samples, sum(int(a[key]) for a in samples), k
                    )
                elif aggregate == "mean":
                    check_outcomes_type(samples, key, bool | int | float)
                    metrics[f"{key}_mean"] = sum(a[key] for a in samples) / len(samples)
                elif match := re.fullmatch(r"max(?P<m>\d+)@(?P<k>\d+)", aggregate):
                    check_outcomes_type(samples, key, bool)
                    if not all("n_steps" in s for s in samples):
                        raise KeyError(
                            "max@ aggregation requires 'n_steps' key in outcomes."
                        )
                    m = int(match.group("m"))
                    k = int(match.group("k"))
                    metrics[f"{key}_max{m}@{k}"] = pass_at_k(
                        n_samples,
                        sum(int(a[key]) and a["n_steps"] <= m for a in samples),
                        k,
                    )
                elif match := re.fullmatch(r"short(?P<m>\d+)?@(?P<k>\d+)", aggregate):
                    check_outcomes_type(samples, key, bool)
                    k = int(match.group("k"))
                    m = int(match.group("m"))
                    if not all("n_reasoning_tokens" in s for s in samples):
                        raise KeyError(
                            "shortm@k aggregation requires 'n_reasoning_tokens' key in outcomes."
                        )
                    if m == 1:  # special case (no need for answers)
                        answer_data_lst = [
                            AnswerData(
                                answer="<DUMMY_ANSWER>",
                                length=s["n_reasoning_tokens"],
                                pass_value=s["pass"],
                            )
                            for s in samples
                        ]
                    else:
                        if not all("answer" in s for s in samples):
                            raise KeyError(
                                "shortm@k aggregation (m>1) requires 'answer' keys in outcomes."
                            )
                        answer_data_lst = [
                            AnswerData(
                                answer=s["answer"],
                                length=s["n_reasoning_tokens"],
                                pass_value=s["pass"],
                            )
                            for s in samples
                        ]
                    metrics[f"{key}_short{m}@{k}"] = short_m_at_k(
                        n_samples, k, m, answer_data_lst
                    )
                elif match := re.fullmatch(r"majority@(?P<k>\d+)", aggregate):
                    check_outcomes_type(samples, key, bool)
                    k = int(match.group("k"))
                    if not all("answer" in s for s in samples):
                        raise KeyError(
                            "majority@k aggregation requires 'answer' keys in outcomes."
                        )
                    answer_data_lst = [
                        AnswerData(answer=s["answer"], pass_value=s["pass"])
                        for s in samples
                    ]
                    metrics[f"{key}_majority@{k}"] = majority_at_k(
                        n_samples, k, answer_data_lst
                    )
                else:
                    raise ValueError(f"Aggregate {aggregate} not supported.")
        aggregate_metrics.append(metrics)

    # Average all outcomes across items
    return {
        key: sum(float(am[key]) for am in aggregate_metrics) / len(aggregate_metrics)
        for key in aggregate_metrics[0].keys()
    }
