# Copyright (c) Meta Platforms, Inc. and affiliates.

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cwm.rl.envs.api import AbstractRewardFn, Env
from cwm.text.datatypes import BaseTextDatum, DictDatum


@dataclass
class RLTaskConfig:
    # The name of the EnvConfig from the registry
    env_config: str

    # The name of the reward function from the reward registry
    reward_fn: str

    # Path of the data file, if required to start an episode
    path: str | None = None

    # Init args used to overwrite env configuration (if needed)
    init_args: dict | None = None

    # Yield n samples for each prompt, to compute pass@k at eval
    samples_per_prompt: int | None = None

    # Aggregation setting for outcomes.
    # Dict should be a mapping of outcome key to aggregate tags, e.g. {"pass": ["@1"]}.
    # Multiple tags can be provided per outcome.
    # Supported aggregations currently are `mean` and `@N` with N an integer
    metrics_spec: dict[str, list[str]] = field(default_factory=dict[str, list[str]])

    @property
    def name(self) -> str:
        base = self.env_config + ":" + self.reward_fn + ":" + (self.path or "")
        args = (
            "_".join(f"{k}={v}" for k, v in sorted(self.init_args.items()))
            if self.init_args
            else ""
        )
        return base + ";" + args


# The EnvConfig specifies the configuration arguments of the Env constructor.
# Together with a method for producing inputs for Env.start() (e.g. loading
# prompts from a certain dataset), this fully defines an RL task.
# (The Env may take additional runtime arguments such as tokenizer; see build_env)
@dataclass
class EnvConfig:
    name: str
    cls: type[Env]
    init_kwargs: dict[str, Any]


# Indivial Env implementations should define EnvConfigs and add them to the registry using register()
ENVS_REGISTRY: dict[str, EnvConfig] = {}


def register_env(config: EnvConfig) -> None:
    name = config.name
    if name in ENVS_REGISTRY:
        raise ValueError(f"EnvConfig {name} already exists.")
    ENVS_REGISTRY[name] = config


def get_env_config(name: str) -> EnvConfig:
    if name not in ENVS_REGISTRY:
        raise ValueError(f"No EnvConfig registered under the name {name}")
    return ENVS_REGISTRY[name]


REWARDS_REGISTRY: dict[str, Callable[..., AbstractRewardFn]] = {}


def register_reward_fn(
    name: str,
    reward_fn: Callable[..., AbstractRewardFn],
) -> None:
    if name in REWARDS_REGISTRY:
        raise ValueError(f"RewardFn {name} already exists.")
    REWARDS_REGISTRY[name] = reward_fn


def get_reward_fn(
    name: str,
) -> Callable[..., AbstractRewardFn]:
    if name not in REWARDS_REGISTRY:
        raise ValueError(f"No RewardFn registered under the name {name}")
    return REWARDS_REGISTRY[name]


def build_env(name: str, **runtime_kwargs: Any) -> Env:
    """
    Build an Env from a config given by name.
    The Env constructor is called using init_kwargs specified in the config,
    supplemented with runtime_kwargs.
    """
    cfg = get_env_config(name)

    params = inspect.signature(cfg.cls.__init__).parameters
    for name in cfg.init_kwargs:
        assert (
            name in params
        ), f"Invalid keyword argument {name} for {cfg.cls} constructor"
    kwargs = cfg.init_kwargs | {
        arg: value for arg, value in runtime_kwargs.items() if arg in params
    }

    return cfg.cls(**kwargs)


def build_env_overrides(task: RLTaskConfig, **runtime_kwargs: Any) -> Env:
    return build_env(
        task.env_config,
        **(
            dict(task.init_args and sorted(task.init_args.items()) or [])
            | runtime_kwargs
        ),
    )


# Our dataloaders will return tuples with (task.name, env instance, jsonl dict)
class TaskDatum(BaseTextDatum[tuple[RLTaskConfig, Env, dict]]):
    pass


# Convert DictDatum read from jsonl to a TaskDatum to be passed to Env.start()
def to_task_datum(task: RLTaskConfig, env: Env, dict_datum: DictDatum) -> TaskDatum:
    return TaskDatum(val=(task, env, dict_datum.val), src=dict_datum.src)


# For evals, we don't include the environment object in data points since
# they're sent across the wire (and will hence be pickled).
class TaskIdxDatum(BaseTextDatum[tuple[RLTaskConfig, int, dict]]):
    def id(self) -> str:
        rl_task_args_name = self.val[0].name
        # Source path is already path of rl_task_args_name
        return f"{rl_task_args_name}|l{self.src.line_no}|p{self.src.pos}"  # type: ignore


def to_task_idx_datum(
    task: RLTaskConfig, env_idx: int, dict_datum: DictDatum
) -> TaskIdxDatum:
    return TaskIdxDatum(val=(task, env_idx, dict_datum.val), src=dict_datum.src)
