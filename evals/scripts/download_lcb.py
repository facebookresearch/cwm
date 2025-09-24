# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
This scripts downloads and filters LCB:
- v5: 2024-08-01 to 2025-02-01
- v6: 2024-08-01 to 2025-05-01

python -m evals.scripts.download_lcb save_path=<path of file>
"""

# ruff: noqa: DTZ001
from dataclasses import dataclass
from datetime import datetime

from cwm.common.params import load_from_cli
from datasets import load_dataset


@dataclass(frozen=True)
class Args:
    save_path: str = ""
    dataset: str = "livecodebench/code_generation_lite"  # other datasets might require adapting map_fn
    split: str = "test"
    version: str = "v5"
    max_workers: int = 32


def main(args: Args) -> None:
    assert args.save_path
    start_date, end_date = {
        "v5": (datetime(2024, 10, 1), datetime(2025, 2, 1)),
        "v6": (datetime(2024, 8, 1), datetime(2025, 5, 1)),
    }[args.version]
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        trust_remote_code=True,
        version_tag="v1_" + args.version,
    )
    dataset = dataset.filter(
        lambda x: start_date <= datetime.fromisoformat(x["contest_date"])
        and datetime.fromisoformat(x["contest_date"]) < end_date
    )
    print(f"Saving dataset with {len(dataset)} items.")
    dataset.to_json(args.save_path, num_proc=args.max_workers, lines=True)


if __name__ == "__main__":
    args = load_from_cli(Args)
    main(args)
