# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
This scripts downloads a huggingface dataset, concatenates all splits, and saves it as a jsonl file.

python -m evals.scripts.download_math dataset=<HF dataset name> save_path=<path of file>
"""

# ruff: noqa: DTZ001
from dataclasses import dataclass

from cwm.common.params import load_from_cli
from datasets import concatenate_datasets, load_dataset


@dataclass(frozen=True)
class Args:
    save_path: str = ""
    dataset: str = ""  # for example HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024, yentinglin/aime_2025
    max_workers: int = 32


def main(args: Args) -> None:
    assert args.save_path
    assert args.dataset
    dataset_dict = load_dataset(
        args.dataset,
    )
    dataset = concatenate_datasets(list(dataset_dict.values()))
    dataset = dataset.map(
        lambda x, i: {"task_id": f"{args.dataset}/{i}"}, with_indices=True
    )
    print(f"Saving dataset with {len(dataset)} items.")
    dataset.to_json(args.save_path, num_proc=args.max_workers, lines=True)


if __name__ == "__main__":
    args = load_from_cli(Args)
    main(args)
