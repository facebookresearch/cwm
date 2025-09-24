# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import random
from collections import Counter
from dataclasses import dataclass

import msgspec
from tqdm.auto import tqdm

from cwm.common.params import load_from_cli


@dataclass
class Args:
    # Glob pattern to find trajectory files.
    # It's expected to be generated from a CWM-RL eval run and follow
    # the RL trajectory format. E.g., "${dump_dir}/trajectories/${task_dataset_id}/*.jsonl"
    traj_glob_pattern: str
    # Path to dump the pred file. By default, we don't save anything.
    save_path: str = ""
    traj_save_path: str = ""
    model_name_or_path: str = "cwm-tts-majority-voting"
    # Max samples for subsampling. -1 means use all samples.
    max_samples: int = -1


def main(args: Args) -> None:
    traj_dict: dict[str, list[dict]] = {}
    full_traj_dict: dict[str, list[dict]] = {}  # Store full trajectories
    for p in glob.glob(args.traj_glob_pattern):
        with open(p) as f:
            for line in tqdm(f):
                d: dict = msgspec.json.decode(line)
                transitions: list = d["rollouts"][0]["traj"]["transitions"]
                instance_id: str = transitions[0]["info"]["instance_id"]
                patch: str = transitions[-1].get("info", {}).get("pred_patch", "") or ""
                passing: bool = transitions[-1]["outcomes"]["pass"]
                traj_dict.setdefault(instance_id, []).append(
                    dict(patch=patch, passing=passing)
                )
                # Store full trajectory data
                if args.traj_save_path:
                    full_traj_dict.setdefault(instance_id, []).append(d)

    print("Applying a shuffle to avoid bias from trajectory ordering")
    # Note this randomization introduces some variance if you set different seeds
    # but ensures determinism for the same seed.
    random.seed(args.model_name_or_path)
    for instance_id in traj_dict:
        random.shuffle(traj_dict[instance_id])

    if args.max_samples > 0:
        print(f"Subsampling to max {args.max_samples} samples per instance")

    # Separate counting and outcome tracking
    patch_counts: dict[str, Counter[str]] = {}  # instance_id -> Counter of patches
    patch_outcomes: dict[
        str, dict[str, list[bool]]
    ] = {}  # instance_id -> patch -> list of outcomes

    for instance_id in traj_dict:
        results = traj_dict[instance_id]

        # Create indices for potential subsampling
        if args.max_samples > 0 and len(results) > args.max_samples:
            # Sample randomly to reduce the bias that faster trajectories are saved first
            results = results[: args.max_samples]

        for result in results:
            patch = result["patch"]
            if not patch:
                # Skip empty patches
                continue
            passing = result["passing"]
            # Count patches (selection logic)
            if instance_id not in patch_counts:
                patch_counts[instance_id] = Counter()
            patch_counts[instance_id][patch] += 1

            # Track outcomes separately
            if instance_id not in patch_outcomes:
                patch_outcomes[instance_id] = {}
            if patch not in patch_outcomes[instance_id]:
                patch_outcomes[instance_id][patch] = []
            patch_outcomes[instance_id][patch].append(passing)

    # Detect and report flaky tests
    flaky_instances: list[tuple[str, str, list[bool]]] = []
    for instance_id, patch_dict in patch_outcomes.items():
        for patch, outcomes in patch_dict.items():
            if len(set(outcomes)) > 1:  # Mixed True/False outcomes
                flaky_instances.append((instance_id, patch, outcomes))

    if flaky_instances:
        print(f"\nFlaky tests detected ({len(flaky_instances)} patch-instance pairs):")
        for instance_id, _, outcomes in flaky_instances:
            pass_count: int = sum(outcomes)
            total_count: int = len(outcomes)
            print(f"  {instance_id}: patch passed {pass_count}/{total_count} times")

    # Select most frequent patch for each instance (without considering outcomes)
    selected_patches: list[dict] = []
    best_trajectories: list[dict] = []  # Store the best trajectories

    for instance_id, counter in patch_counts.items():
        most_common_patch, occurrences = counter.most_common(1)[0]
        assert occurrences <= args.max_samples or args.max_samples < 0

        # Get the outcome for the selected patch (use majority vote if flaky)
        outcomes = patch_outcomes[instance_id][most_common_patch]
        passing = sum(outcomes) > len(outcomes) / 2  # Majority vote

        selected_patches.append(
            dict(
                model_name_or_path=args.model_name_or_path,
                instance_id=instance_id,
                model_patch=most_common_patch,
                occurrences=occurrences,
                passing=passing,
            )
        )

        if not args.traj_save_path:
            continue

        # Find and store the best trajectory for this instance
        # Get all trajectories for this instance that have the most common patch
        full_trajs = full_traj_dict[instance_id]
        for traj in full_trajs:
            transitions = traj["rollouts"][0]["traj"]["transitions"]
            traj_patch = transitions[-1].get("info", {}).get("pred_patch", "") or ""
            if traj_patch == most_common_patch:
                # This is one of the best trajectories
                best_trajectories.append(traj["rollouts"][0])
                break  # Take the first occurrence of the best patch

    print(f"Total instances: {len(traj_dict)}")
    num_passing: int = sum(1 for p in selected_patches if p["passing"])
    print(f"Passing instances: {num_passing} ({num_passing / len(traj_dict):.2%})")
    if args.save_path:
        print(f"Saving to {args.save_path}")
        with open(args.save_path, "w") as f:
            for patch_dict in selected_patches:
                f.write(msgspec.json.encode(patch_dict).decode() + "\n")

    if args.traj_save_path:
        print(
            f"Saving {len(best_trajectories)} best trajectories to {args.traj_save_path}"
        )
        solve_rate = sum(
            traj["traj"]["transitions"][-1]["outcomes"]["pass"]
            for traj in best_trajectories
        ) / len(best_trajectories)
        print(f"Solve rate in traj file: {solve_rate:.2%}")
        with open(args.traj_save_path, "w") as f:
            for traj in best_trajectories:
                f.write(msgspec.json.encode(traj).decode() + "\n")


if __name__ == "__main__":
    args = load_from_cli(Args)
    main(args)
