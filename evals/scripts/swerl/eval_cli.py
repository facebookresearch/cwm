# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
A fast evaluation harness (5min on SWE-bench Verified with enough threads) for any data in SWE-RL format, including SWE-bench.
The output format is compatible with the SWE-bench submission requirements, and it can serve as a drop-in
replacement for the SWE-bench evaluation harness.
We currently support only the Modal backend.

Example evaluation on SWE-bench Verified:

1. Convert SWE-bench Verified to SWE-RL format (if you don't have it yet):

    python -m cwm.rl.swerl.scripts.format_swerl \
        save_path=swebench_verified_swerl.jsonl \
        dataset=princeton-nlp/SWE-bench_Verified \
        split=test \
        namespace=swebench

2. Run evaluation:

    # Gold setting
    python -m cwm.rl.swerl.scripts.eval_cli \
        data_file=swebench_verified_swerl.jsonl \
        eval_dir=sbv-gold-$(date +%Y%m%d_%H%M%S)

    # Prediction setting (change all_preds.jsonl to your prediction file)
    python -m cwm.rl.swerl.scripts.eval_cli \
        data_file=swebench_verified_swerl.jsonl \
        pred_file=all_preds.jsonl \
        eval_dir=sbv-pred-$(date +%Y%m%d_%H%M%S)

    # Test noop patches
    python -m cwm.rl.swerl.scripts.eval_cli \
        data_file=swebench_verified_swerl.jsonl \
        eval_dir=sbv-noop-$(date +%Y%m%d_%H%M%S) \
        test_noop_patch=True

Some A/B testing comparing our CLI and the official submission results. Since some SWE-bench tests are flaky,
the results may vary slightly between runs, but they should be very close.

Setting                 Official   Our CLI
------------------------------------------
Gold                     100.0     98.0
OpenHands + Sonnet4      70.4      70.4
GLM-4.5                  64.2      64.6
Qwen3-Coder-32-A3        51.0      51.0
SWE-RL                   41.2      41.2
"""

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from cwm.common.params import load_from_cli
from cwm.rl.swerl.eval_backend.eval import EvalResult, eval_instance_default

logging.getLogger("cwm.rl.swerl.modal_backend").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class Args:
    # Path to the eval (benchmark) file that follows the SWE-RL format,
    # which can be formatted using cwm/rl/swerl/scripts/format_swerl.py
    # Example: SWE-bench Verified file formatted using the above script.
    data_file: str
    # Path to dump the evaluation results
    eval_dir: str
    # Path to the prediction file in SWE-bench submission format, required keys:
    # instance_id, model_patch.
    # If None, we will use the "patch" field in the data_file to evaluate gold patches.
    pred_file: str | None = None

    # run = False will skip running the evaluation, and only aggregate results from existing eval_dir/eval.jsonl
    run: bool = True

    # Timeout in seconds for each instance
    timeout: float = 300.0
    # Set to a non-zero value to evaluate on a random subset, useful for quick testing
    num_instances_to_eval: int = -1
    # True to test noop patches, useful to ensure the eval harness does not have false negatives
    test_noop_patch: bool = False
    random_seed: int = field(default=42)
    # Increase the workers to speed up evaluation.
    # Note if you use Modal, you may need to increase your plan.
    # Otherwise, you can hit rate limit errors and evaluation can produce false positives (env_error).
    max_workers: int | None = None
    backend: str = "modal"


NOOP_PATCH = """diff --git a/this_is_invisible.py b/this_is_invisible.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/this_is_invisible.py
@@ -0,0 +1 @@
+# This is a commented out line
"""


def main(args: Args) -> None:
    logger.info(f"Running test_swebench_eval with CLI args: {asdict(args)}")

    if args.run:
        with open(args.data_file) as f:
            dataset: list[dict] = [json.loads(line) for line in f]
        # make sure test_noop_patch and pred_file are not set at the same time
        assert not (args.test_noop_patch and args.pred_file)

        if args.pred_file is not None:
            # pred_dataset can contain N samples for one instance. We flatten it
            with open(args.pred_file) as f:
                pred_dataset: list[dict] = [json.loads(line) for line in f]
            dataset_dict = {x["instance_id"]: x for x in dataset}
            new_dataset = list[dict]()
            for x in pred_dataset:
                d = dataset_dict[x["instance_id"]]
                newd = {**d}
                newd["patch"] = x["model_patch"]
                new_dataset.append(newd)
            dataset = new_dataset

        else:
            random.seed(args.random_seed)
            if args.num_instances_to_eval > 0:
                indices = random.sample(
                    range(len(dataset)), k=args.num_instances_to_eval
                )
                dataset = [dataset[i] for i in indices]
            if args.test_noop_patch:
                # Add noop patch to each instance
                print("Adding noop patch to each instance")
                dataset = [{**instance, "patch": NOOP_PATCH} for instance in dataset]

        eval_file = Path(args.eval_dir) / "eval.jsonl"
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with (eval_file.parent / "args.json").open("a") as f:
            f.write(json.dumps(asdict(args), indent=2) + "\n")

        if eval_file.exists():
            with eval_file.open("r") as f:
                existing_results: list[dict] = [json.loads(line) for line in f]
            existing_results_dict: dict[tuple[str, str], dict] = {
                (result["instance_id"], result["patch"]): result
                for result in existing_results
            }
            print(f"Found {len(existing_results)} existing results in {eval_file}")
            print("Length of dict", len(existing_results_dict))
        else:
            existing_results_dict = {}

        # Returns (cached, result)
        def run_eval(instance: dict) -> tuple[bool, EvalResult]:
            key = (instance["instance_id"], instance["patch"])
            if key in existing_results_dict:
                cached_result = existing_results_dict[key]
                result = EvalResult(
                    outcome=cached_result["outcome"],
                    message=cached_result["message"],
                )
                return True, result

            eval_dir = Path(args.eval_dir) / instance["instance_id"]
            eval_result = eval_instance_default(
                instance,  # type: ignore
                instance["patch"],
                eval_dir,
                args.timeout,
                workdir=instance["repo_root_path"],
                backend=args.backend,
            )
            return False, eval_result

        print(f"Running {len(dataset)} instances")

        pbar = tqdm(total=len(dataset), desc="Evaluating instances")
        instance_success_dict = dict[str, list[bool]]()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_eval, instance): instance for instance in dataset
            }
            print(f"Writing results to {eval_file}")
            with eval_file.open("a") as f:
                for future in as_completed(futures):
                    instance = futures[future]
                    instance_id: str = instance["instance_id"]
                    try:
                        cached, result = future.result()
                    except Exception as e:
                        print(f"Error ({instance_id}): {type(e)}: {e}")
                        cached = False
                        result = EvalResult(
                            outcome="env_error", message=f"{type(e)}: {e}"
                        )
                    d = dict(
                        instance_id=instance_id,
                        patch=instance["patch"],
                        outcome=result.outcome,
                        message=result.message,
                    )
                    instance_success_dict.setdefault(instance_id, []).append(
                        result.outcome == "pass"
                    )
                    if not cached:
                        f.write(json.dumps(d) + "\n")
                        f.flush()
                    n_resolved = sum(
                        1 for v in instance_success_dict.values() if any(v)
                    )
                    pbar.set_description(
                        f"Pass@Any: {n_resolved} / {len(instance_success_dict)}"
                    )
                    pbar.update(1)
        pbar.close()

        print(len(instance_success_dict), "instances evaluated")
        print(
            "pass@any",
            sum(1 for v in instance_success_dict.values() if any(v))
            / len(instance_success_dict),
        )
        pass_at_1 = sum(
            sum(v) / len(v) for v in instance_success_dict.values() if v
        ) / len(instance_success_dict)
        print("pass@1:", pass_at_1)

    else:
        eval_file = Path(args.eval_dir) / "eval.jsonl"
        instance_success_dict = dict[str, list[bool]]()
        with eval_file.open("r") as f:
            for line in f:
                cached_result = json.loads(line)
                instance_id = cached_result["instance_id"]
                instance_success_dict.setdefault(instance_id, []).append(
                    cached_result["outcome"] == "pass"
                )
        print(len(instance_success_dict), "instances evaluated")
        print(
            "pass@any",
            sum(1 for v in instance_success_dict.values() if any(v))
            / len(instance_success_dict),
        )
        print(
            "pass@1",
            sum(sum(v) / len(v) for v in instance_success_dict.values() if v)
            / len(instance_success_dict),
        )


if __name__ == "__main__":
    args = load_from_cli(Args)
    main(args)
