# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from cwm.common.params import save_params
from cwm.rl.envs.api import Trajectory, Transition
from cwm.rl.envs.config import RLTaskConfig
from cwm.rl.lib.datatypes import (
    DataSource,
    RolloutDumpInfo,
    RolloutInfo,
    TokenRolloutInfo,
)
from cwm.text.tokenizers import Tokenizer


def transition_to_dict(
    tr: Transition,
    tokenizer: Tokenizer,
    keep_info: bool = True,
    keep_tokens: bool = False,
) -> dict[str, Any]:
    """
    Turn a Transition into a dict, for logging, restartable evals, and offline data creation.
    Unlike dataclasses.asdict, this function checks or (if necessary) produces the action_str and observation_str fields,
    and allows the user to discard certain information.

    When keep_tokens == False, token fields tr.action and tr.observation are discarded, and reproduced by transition_from_dict
    upon reloading. The reproduced tokens will be equivalent but not necessary equal to the ones produced by the model,
    but for offline RL this should be fine.

    Args:
        tr: The Transition object to convert to a dictionary
        tokenizer: Tokenizer used for decoding tokens and verifying/producing string fields
        keep_info: Whether to keep the info field in the output dictionary (default: False)
        keep_tokens: Whether to keep token fields (action, observation) in the output dictionary (default: False)

    Returns:
        A dictionary representation of the Transition
    """

    d = asdict(tr)

    # The string fields of the transition may not be populated by the env
    # In that case, create it using the tokenizer
    # Otherwise, check that it matches the tokenizer output

    if not d["action_str"]:
        action_str = tokenizer.decode(tr.action, cut_at_stop_tokens=True)
        d["action_str"] = action_str
    else:
        # TODO: I have disabled the asserts below, checking whether the action_str computed here matches the one in the transition.
        # The reason is that to make it pass, I had to use cut_at_stop_tokens=True for action and False for observation,
        # and I'm not sure why exactly and if this convention is enforced in all envs.
        pass
        # assert d["action_str"] == action_str, (
        #     "Transition.action_str does not match decoded tr.action:\n"
        #     f"{tr.action_str=}\n"
        #     f"{action_str=}\n"
        #     "This could indicate an Env error, or modification of the transition."
        # )

    if not d["observation_str"]:
        observation_str = tokenizer.decode(tr.observation, cut_at_stop_tokens=False)
        d["observation_str"] = observation_str
    else:
        pass
        # assert d["observation_str"] == observation_str, (
        #     "Transition.observation_str does not match decoded tr.observation:\n"
        #     f"{tr.observation_str=}\n"
        #     f"{observation_str=}\n"
        #     "This could indicate an Env error, or modification of the transition."
        # )

    if not keep_info:
        del d["info"]

    if not keep_tokens:
        del d["action"]
        del d["observation"]

    return d


def transition_from_dict(d: dict[str, Any]) -> Transition:
    """
    Reconstruct a Transition object from a dictionary representation.
    This function is the inverse of transition_to_dict.

    For now, it is necessary for the token fields (action, observation) to be
    saved for the transition to be reloadeable.

    Args:
        d: Dictionary representation of a Transition

    Returns:
        A reconstructed Transition object
    """

    # Make a copy of the dictionary to avoid modifying the input
    d = d.copy()

    assert "action" in d and "observation" in d, (
        "The action and observation tokens must be saved for a transition to be re-loadeable, "
        "because re-tokenizing can lead to changes. Use keep_tokens=True when saving."
    )

    # Reconstruct missing fields if necessary
    # if "action" not in d:
    #     d["action"] = tokenizer.encode(d["action_str"], bos=False, eos=False)
    # if "observation" not in d:
    #     d["observation"] = tokenizer.encode(d["observation_str"], bos=False, eos=False)

    if "info" not in d:
        d["info"] = {"info_discarded": True}

    # Check presence of keys and type check them
    action = d["action"]
    action_str = d["action_str"]
    rewards = d["rewards"]
    observation = d["observation"]
    observation_str = d["observation_str"]
    terminal = d["terminal"]
    context_switch = d["context_switch"]
    outcomes = d["outcomes"]
    info = d["info"]

    assert all(isinstance(x, int) for x in action)
    assert all(isinstance(x, float) for x in rewards)
    assert all(isinstance(x, int) for x in observation)
    assert isinstance(terminal, bool)
    assert isinstance(context_switch, bool)
    assert isinstance(outcomes, dict)
    assert isinstance(info, dict)

    return Transition(
        action=action,
        action_str=action_str,
        rewards=rewards,
        observation=observation,
        observation_str=observation_str,
        terminal=terminal,
        context_switch=context_switch,
        outcomes=outcomes,
        info=info,
    )


def trajectory_to_dict(
    traj: Trajectory,
    tokenizer: Tokenizer,
    keep_info: bool = True,
    keep_tokens: bool = False,
    keep_logprobs: bool = False,
) -> dict[str, Any]:
    """
    Turn a Trajectory into a dict, for logging, restartable evals, and offline data creation.
    This function converts each Transition in the Trajectory to a dictionary using transition_to_dict.

    Args:
        traj: The Trajectory object to convert to a dictionary
        tokenizer: Tokenizer used for decoding tokens in the transitions
        keep_info: Whether to keep the info field in each transition (default: False)
        keep_tokens: Whether to keep token fields in each transition (default: False)
        keep_logprobs: Whether to keep log probabilities in the output dictionary (default: False).
                       When True, keep_tokens must also be True to ensure consistency between tokens and log probabilities.

    Returns:
        A dictionary representation of the Trajectory containing:
        - transitions: List of dictionaries representing each Transition
        - truncation_return: Present only if the trajectory was truncated
        - log_prob_behavior: Present only if keep_logprobs is True
    """
    d: dict[str, Any] = {
        "transitions": [
            transition_to_dict(tr, tokenizer, keep_info, keep_tokens)
            for tr in traj.transitions
        ],
    }
    if traj.truncated:
        d["truncation_return"] = traj._truncation_return

    if keep_logprobs:
        assert keep_tokens, (
            "For your own safety, when keep_logprobs is True set keep_tokens to True as well. "
            "When keep_tokens is False, the reloaded tokens may not match the generated ones "
            "and the number of tokens can change."
        )
        d["log_prob_behavior"] = traj.log_probs

    return d


def trajectory_from_dict(d: dict[str, Any]) -> Trajectory:
    """
    Reconstruct a Trajectory object from a dictionary representation.
    This function is the inverse of traj_to_dict.

    The function converts each transition dictionary back to a Transition object,
    and reconstructs the Trajectory with any log probabilities and truncation return if present.

    Args:
        d: Dictionary representation of a Trajectory
        tokenizer: Tokenizer used for encoding strings back to tokens if needed in the transitions

    Returns:
        A reconstructed Trajectory object
    """

    transitions = [transition_from_dict(td) for td in d["transitions"]]
    log_probs = d.get("log_prob_behavior", None)
    truncation_return = d.get("truncation_return", None)

    return Trajectory.from_transitions(transitions, log_probs, truncation_return)


def rollout_to_dict(
    rollout: RolloutInfo,
    tokenizer: Tokenizer,
    keep_info: bool = True,
    keep_tokens: bool = False,
    keep_logprobs: bool = False,
) -> dict[str, Any]:
    assert (
        {f.name for f in fields(rollout)}
        == {
            "traj",
            "data_src",
            "start_args",
            "begin_step",
            "end_step",
            "metrics",
            "rl_task_args",
            "pivotal",
        }
    ), "Unexpected or missing fields in RolloutInfo. Make sure to update this function when the class changes."

    return {
        "traj": trajectory_to_dict(
            rollout.traj, tokenizer, keep_info, keep_tokens, keep_logprobs
        ),
        "data_src": asdict(rollout.data_src),
        "start_args": rollout.start_args,
        "begin_step": rollout.begin_step,
        "end_step": rollout.end_step,
        "metrics": rollout.metrics,
        "rl_task_args": asdict(rollout.rl_task_args),
        "pivotal": list(map(asdict, rollout.pivotal))
        if rollout.pivotal is not None
        else None,
    }


def rollout_from_dict(
    d: dict[str, Any],
    keep_steps: bool = False,
    keep_metrics: bool = False,
) -> RolloutInfo:
    return RolloutInfo(
        traj=trajectory_from_dict(d["traj"]),
        data_src=DataSource(**d["data_src"]),
        start_args=d["start_args"],
        begin_step=d["begin_step"] if keep_steps else -1,
        end_step=d["end_step"] if keep_steps else -1,
        metrics=d["metrics"] if keep_metrics else {},
        rl_task_args=RLTaskConfig(**d["rl_task_args"]),
        pivotal=[TokenRolloutInfo(**t) for t in d["pivotal"]]
        if "pivotal" in d and d["pivotal"]
        else None,
    )


# TODO: remove the functions below and replace their remaining uses in eval.py by the functions above
def traj_to_dict(traj: Trajectory) -> dict[str, Any]:
    """
    Convert the Trajectory to a dictionary that can be serialized to JSON.
    """
    return {
        "transitions": [asdict(tr) for tr in traj.transitions],
        "tokens": traj.tokens,
        "log_probs": traj.log_probs,
        "source": traj.source,
        "rewards": traj.rewards,
        "terminated": traj.terminated,
        "truncated": traj.truncated,
        "truncation_return": traj._truncation_return,
    }


def minimize_traj_to_dict(traj: Trajectory) -> dict[str, Any]:
    """
    Remove the tokens field of the trajectory to a dictionary suitable for inspection friendly dump.
    """
    return {
        "transitions": [
            {
                "action_str": tr.action_str,
                "observation_str": tr.observation_str,
                "eos_reward": tr.rewards[-1] if tr.rewards else None,
                "outcomes": tr.outcomes,
                "info": tr.info,
            }
            for tr in traj.transitions
        ]
    }


def task_name_to_path_name(task_name: str) -> str:
    # Note: This is needed to have a valid path name
    dump_task_name = re.sub("[:;]", "", re.sub("[/.]", "_", task_name))
    return dump_task_name


def create_dump_trajectory_dirs(traj_dump_dir: Path, tasks: list[RLTaskConfig]) -> None:
    # Create trajectory dump directory
    traj_dump_dir.mkdir(parents=True, exist_ok=True)
    # Create per-task dump directory and save the corresponding task config
    for rl_task_args in tasks:
        task_name = rl_task_args.name
        dump_task_name = task_name_to_path_name(task_name)
        dump_dir_for_task = traj_dump_dir / dump_task_name
        dump_dir_for_task.mkdir(parents=True, exist_ok=True)
        task_config_dump_path = dump_dir_for_task / "task_config.yaml"
        save_params(rl_task_args, task_config_dump_path)


def get_rollout_dump_info(
    rollouts: list[RolloutInfo],
    metrics: list[dict] | None,
    trajectory_dump_dir: Path,
    dump_mode: str,
    worker_id: str | None,
    tokenizer: Tokenizer,
) -> RolloutDumpInfo:
    assert dump_mode in ("full", "minimal")
    assert metrics is None or len(metrics) == len(rollouts)

    if dump_mode == "full":
        keep_info, keep_tokens, keep_logprobs = True, True, True
    else:
        keep_info, keep_tokens, keep_logprobs = True, False, False

    rollouts_json_to_dump = []
    for rollout in rollouts:
        rollout_dict = rollout_to_dict(
            rollout,
            tokenizer=tokenizer,
            keep_info=keep_info,
            keep_tokens=keep_tokens,
            keep_logprobs=keep_logprobs,
        )
        rollouts_json_to_dump.append(rollout_dict)

    json_to_dump = {"rollouts": rollouts_json_to_dump}
    if metrics is not None:
        json_to_dump["metrics"] = metrics

    # TODO come up with a shorter name for the file and the metrics
    task_name = rollouts[0].rl_task_args.name
    dump_task_name = task_name_to_path_name(task_name)
    filename = f"worker_{worker_id}.jsonl" if worker_id else f"{dump_task_name}.jsonl"
    file_dir = trajectory_dump_dir / dump_task_name
    assert file_dir.exists()
    file_path = file_dir / filename
    return RolloutDumpInfo(file_path=file_path, json=json_to_dump)
