# Copyright (c) Meta Platforms, Inc. and affiliates.

from cwm.rl.envs import api, outcomes


def reward_unroll(action: list[int], reward: float) -> list[float]:
    assert len(action) > 0, "Single reward requires actions"
    return [0.0] * (len(action) - 1) + [float(reward)]


def _pass_terminal(tr: api.Transition) -> float:
    assert tr.terminal, "Terminal transition expected for pass reward"
    return 2.0 * float(outcomes.successful_pass(tr.outcomes)) - 1.0


def _pass(tr: api.Transition) -> float:
    if not tr.terminal:
        return 0.0
    return _pass_terminal(tr)


class PassOnlyRewardFn(api.RewardFn):
    @property
    def range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def __call__(self, tr: api.Transition) -> list[float]:
        return reward_unroll(tr.action, _pass(tr))
