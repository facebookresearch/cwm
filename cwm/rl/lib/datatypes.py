# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from pathlib import Path

from cwm.rl.envs.api import Trajectory
from cwm.rl.envs.config import RLTaskConfig
from cwm.text.datatypes import BaseTextDatum


@dataclass(frozen=True)
class DataSource:
    # maps to BaseTextDatum.Source, used to identify problems instances (prompts)
    # for a task, for instance when aggregating results in eval.py
    path: str
    line_no: int
    pos: int

    @classmethod
    def from_base_text_datum_source(cls, src: BaseTextDatum.Source) -> "DataSource":
        return cls(str(src.path), src.line_no, src.pos)


@dataclass
class PivotalCandidate:
    token_pos_idx: int  # position relative to the first action
    token_id: int
    success_delta: float = -1
    n_attempts: int = -1
    n_success: int = -1


@dataclass
class TokenRolloutInfo:
    token_pos_idx: int  # position relative to the first action. -1 -> s0
    token_id: int | None
    n_attempts: int
    n_success: int
    success_delta: float | None = None
    is_pivotal: bool = False
    offset_token_pos_idx: int | None = (
        None  # offset + token_pos_idx = absolute position
    )

    def success(self) -> float:
        return self.n_success / self.n_attempts

    @classmethod
    def from_pivotal(cls, pivotal: PivotalCandidate, offset_token_pos_idx: int | None):
        return TokenRolloutInfo(
            token_pos_idx=pivotal.token_pos_idx,
            token_id=pivotal.token_id,
            n_attempts=pivotal.n_attempts,
            n_success=pivotal.n_success,
            success_delta=pivotal.success_delta,
            is_pivotal=True,
            offset_token_pos_idx=offset_token_pos_idx,
        )


@dataclass
class RolloutDumpInfo:
    file_path: Path
    json: dict


@dataclass
class RolloutInfo:
    """
    A Trajectory with extra info on the rollout process, i.e. how it was generated.
    """

    traj: Trajectory

    # Original data source (identifies the problem instance uniquely together with rl_task_args)
    data_src: DataSource

    # Filtered arguments passed to env.start() to produce the initial state & prompt,
    # cf. RLWorkerArgs.keep_start_args
    start_args: dict

    # Model step at begin and end of rollout
    begin_step: int
    end_step: int

    # Number of exceptions encountered during rollout
    # TODO num_exceptions: int = 0

    # Needed to replay the environment.
    rl_task_args: RLTaskConfig

    metrics: dict[str, float] = field(default_factory=dict)

    pivotal: list[TokenRolloutInfo] | None = None

    @property
    def is_valid(self) -> bool:
        return not self.traj.is_empty
