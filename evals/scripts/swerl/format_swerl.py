# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
This script converts swebench type data into SWE-RL format, for training and evaluation.

    # Converting SWE-bench Verified for evaluation
    python -m evals.scripts.swerl.format_swerl \
        save_path=swebench_verified_swerl.jsonl \
        dataset=princeton-nlp/SWE-bench_Verified \
        split=test \
        namespace=swebench
"""

from dataclasses import dataclass

from swebench.harness.test_spec.test_spec import make_test_spec

from cwm.common.params import load_from_cli
from datasets import load_dataset


@dataclass
class Args:
    save_path: str
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    namespace: str = "swebench"
    max_workers: int = 32


def map_fn(d: dict, args: Args) -> dict:
    id: str = d["instance_id"].replace("__", "_1776_")
    image_key: str = args.namespace + "/" + "sweb.eval.x86_64." + id + "+latest"

    test_spec = make_test_spec(d)
    d["repo_root_path"] = "/testbed"
    d["eval_script"] = test_spec.eval_script
    d["docker_url"] = "docker.io/" + image_key.removesuffix("+latest")
    return d


def main(args: Args) -> None:
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.map(
        map_fn,
        fn_kwargs={"args": args},
        num_proc=1,
    )
    dataset = dataset.rename_column("problem_statement", "issue")

    dataset = dataset.shuffle(seed=666)
    print(f"Saving dataset with {len(dataset)} items.")
    dataset.to_json(args.save_path, num_proc=args.max_workers, lines=True)


if __name__ == "__main__":
    args = load_from_cli(Args)
    main(args)
