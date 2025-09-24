# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass, field
from pathlib import Path

from cwm.fastgen.generate import GenArgs
from cwm.logging.metrics import (
    JsonlMetricsLogger,
    MetricsLogger,
)
from cwm.rl.envs.config import RLTaskConfig

logger = logging.getLogger()


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: str | None = None


@dataclass
class FastGenArgs(GenArgs):
    # Add to base class
    tp_size: int = 1
    vocab_parallel: bool = False
    loss_parallel: bool = False

    # Overriding defaults from base class
    use_sampling: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    logprobs: bool = False
    max_batch: int = 128
    max_gen: int | None = None
    handover_frequency: int = 4


@dataclass
class SetupArgs:
    spawn_method: str = "forkserver"
    torch_init_timeout: int = 600
    cuda_matmul_allow_tf32: bool = False
    cuda_allow_bf16_reduced_precision_reduction: bool = True
    autograd_detect_anomaly: bool = False


@dataclass
class RLEvalArgs:
    """Evaluation args"""

    dump_dir: str = ""
    dump_mode: str = "full"  # full, minimal
    # We only dump these keys from start args (None means all keys are dumped):
    keep_start_args: list[str] = field(
        default_factory=lambda: ["task_id", "instance_id"]
    )
    perf_log_freq: float = 60.0  # in seconds
    checkpoint_dir: str = ""
    metric_log_dir: str | None = None

    seed: int = 42
    num_rollout_threads: int = 8
    data_queue_size: int = 1000000

    # Allows to pass explicitly the global_step
    # instead of relying on pre-defined parsing rule
    # from checkpoint name
    global_step: int | None = None

    tasks: list[RLTaskConfig] = field(default_factory=list)

    # Might want to set this to false for very long evals to avoid the final aggregation
    # OOMing when loading the full metrics file. Aggregation can always be run
    # separately from the saved metrics if needed.
    run_metrics_aggregation: bool = True

    gen_args: FastGenArgs = field(default_factory=FastGenArgs)
    setup: SetupArgs = field(default_factory=SetupArgs)

    tokenizer: TokenizerArgs | None = None  # default loads tokenizer in params.json

    log_level: str = "info"
    max_exceptions: int = 3

    def __post_init__(self) -> None:
        if self.metric_log_dir is None:
            self.metric_log_dir = self.dump_dir


def build_metrics_logger(
    dump_dir: Path,
    tag: str,
) -> MetricsLogger:
    jsonl_mlogger = JsonlMetricsLogger(dump_dir, tag=tag)
    logger.info(f"Instantiated {tag} metrics logger. ")
    return jsonl_mlogger
