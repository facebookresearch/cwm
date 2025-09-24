# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
The Env interface provides a simple and uniform way to define RL problems.

The main classes and methods are:
Env:
- start: begin a new episode by constructing a State and Transition,
  based on some external inputs such as a prompt.
- step: given a State and action (tokens), advance the state and produce a Transition.

Envs are parameterized and so must be configured to fully define an RL problem.
This handled via the functionality in config.py.

State: these objects represent the (hidden) state of the environment. They
also encapsulate resources that may have been acquired for the respective
episode. `state.close()` frees these resources. For convenience, State objects
can be used as context managers. States can be copied via `clone()`, although
this may not be supported by all Envs.

Transition: dataclass containing all information the agent may use for inference & learning.
Includes the latest action, observation, rewards and miscellaneous other info.
Note that rewards are no longer added to the transition by the environment itself.
They are instead configured through a separate reward function (see below).

Trajectory: convenience class for tracing an episode.
Keeps track of the Transitions from an episode, and implements the canonical way to format
observation and action tokens into a sequence (context) that can be fed to an LLM.

RewardFn: defines a reward function abstraction.
Reward functions are defined separately from environments, and they can be combined with environments flexibly.
A reward function can use anything in a transition to compute the reward sequence
(although most commonly it should only require the outcomes returned by the environment `step`).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Self


@dataclass(init=False)
class Transition:
    """
    At each time step, the Env emits a Transition tr, containing exactly the information
    (no more no less) that may be used by the agent for inference and learning.
    Use of any information about the State that is not in the transition is considered
    cheating and leads to numbers (returns) that are not directly comparable.

    The transition contains the most recent action and the resulting observation,
    both of which are assumed to be lists of tokens.
    A typical sequence model policy would see the full history of actions and observations,
    when choosing the next action, but each Transition only contains the current ones.
    Convenient and consistent history tracing and context formatting is handled by the Trajectory class.

    The transition also contains a list of rewards (exactly one for each action token),
    as well as flags indicating terminal states, and a free-form info dict.
    Upon construction a single reward is allowed, in which case it is assumed
    to apply to the last action.
    """

    # The action just taken
    action: list[int]

    # For human consumption only; Do not re-tokenize.
    action_str: str | None

    # The env may emit a reward at each time step (action token)
    # Guaranteed to have the same length as action
    rewards: list[float] | None

    # The observation produced by the Env in response to the action
    observation: list[int]

    # For human consumption only; Do not re-tokenize.
    observation_str: str | None

    # True if the environment transitioned into a terminal state
    terminal: bool

    # Whether the Env suggests a context switch
    # (i.e. discarding history and using only the latest observation as prompt)
    context_switch: bool

    # Collection of quantitative outcomes such as `pass`, `compile`, `runtime`, etc.
    # Some may be env specific, some more general.
    outcomes: dict[str, Any]

    # Arbitrary additional information
    info: dict

    def __init__(
        self,
        *,
        action: list[int] | None = None,
        action_str: str | None = None,
        rewards: list[float] | None = None,
        reward: float | None = None,
        observation: list[int] | None = None,
        observation_str: str | None = None,
        terminal: bool = False,
        context_switch: bool = False,
        outcomes: dict[str, Any] | None = None,
        info: dict | None = None,
    ):
        assert not terminal or (
            observation is None or len(observation) == 0
        ), "Terminal transitions must have no observation"
        assert (
            not terminal or not context_switch
        ), "Terminal transitions must have no context switch"
        assert (
            reward is None or rewards is None
        ), "Specify either 'reward', 'rewards', or neither"

        self.action = [int(x) for x in action] if action is not None else []
        self.action_str = action_str

        if reward is not None:
            assert len(self.action) > 0, "Single reward requires actions"
            self.rewards = [0.0] * (len(self.action) - 1) + [float(reward)]
        elif rewards is not None:
            self.rewards = [float(x) for x in rewards]
        elif self.action == []:
            self.rewards = []
        else:
            self.rewards = None
        assert not self.rewards or len(self.action) == len(
            self.rewards
        ), f"{len(self.action)=} != {len(self.rewards)=}"

        self.observation = (
            [int(x) for x in observation] if observation is not None else []
        )
        self.observation_str = observation_str
        self.terminal = terminal
        self.context_switch = context_switch
        self.outcomes = outcomes if outcomes is not None else {}
        self.info = info if info is not None else {}

    def add_rewards(self, rewards: list[float]) -> None:
        assert len(self.action) == len(
            rewards
        ), f"{len(self.action)=} != {len(rewards)=}"
        self.rewards = rewards


class AbstractRewardFn(ABC):
    """
    After each environment `step`, a reward is computed through a reward function
    and added to the transition.
    The first transition of the episode (the return of `start`) is ignored as it
    contains no outcomes.

    Reward functions are pure functions of transitions (they just read transitions and return a sequence of scalars).
    They are however implemented as a class so they can be paired with properties such as reward ranges,
    which can be used to compute baselines, determine regularisation parameters, etc.
    """

    @property
    @abstractmethod
    def range(self) -> tuple[float, float]: ...

    @abstractmethod
    def __call__(self, tr: Transition) -> list[float]: ...


class RewardFn(AbstractRewardFn):
    def __init__(self, *args, **kwargs):
        # Constructor ignoring any state-related arguments, provided for convenience
        pass


class State:
    def clone(self) -> Self:
        raise NotImplementedError("This Env does not support cloning.")

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class Env(ABC):
    """
    A token-based RL Environment, AKA a (parameterized) Partially Observable Markov Decision Process (POMDP).

    The Env has two methods:
    - start: begin a new episode (given some env-specific arguments such as a prompt / jsonl-derived dict)
    - step: transition to the next state (given state and action)

    A typical Env can be configured via Env.__init__ and (the distribution of) Env.start() arguments,
    with each configuration corresponding to a different POMDP problem.
    To enable uniform Env construction, named configurations are registered via the functionality in registry.py.
    """

    @abstractmethod
    def max_action_len(self, state: State) -> int:
        """
        This is the only action space specification the env _must_ provide, as it is necessary for any sampler to have this constraint.
        """

    @abstractmethod
    def start(self, episode_args: dict | None = None) -> tuple[State, Transition]:
        """
        Start a new episode by creating an initial State and a Transition containing the initial observation (prompt).

        Args:
            episode_args: arguments used to determine the initial state of the episode.
                The particular argument types depend on the Env subclass.
                For instance, it could be a dict loaded from jsonl, containing a prompt from a dataset.
        Returns:
            initial_state: The initial State of the episode.
            initial_transition: The Transition into the initial state, containing the first observation.
        """

    @abstractmethod
    def step(self, state: State, action: list[int]) -> Transition:
        """
        Apply the action to advance the given State and return the resulting Transition.

        The returned Transition contains all and only those variables the agent may use to choose future actions:
        observation tokens, the last action, rewards, and miscellaneous info.
        The returned tr.action should always equal the action given as input.

        Args:
            state: current State.
            action: list of token ids
        Returns:
            transition: the transition into the next state, containing the last action, observation, and reward among other information.
        """


class Trajectory:
    """
    The Trajectory class makes it easy to keep track of everything that happened during an episode/rollout,
    and provides a canonical way to produce the aligned sequences of tokens, token source (agent or environment),
    log probs and rewards.

    The sequence of tokens that is currently visible to the agent is called the context, and can be
    retrieved via the traj.context attribute. Usually, new actions and observations are simply appended
    to the context, allowing the agent to see the whole history. Alternatively, a context switch may be
    introduced, after which the agent will see a new context / prompt, and will no longer see the history.
    A complete trajectory thus consists of one or more contexts (lists of tokens), each of which
    forms a sequence.

    For pretty printing of trajectories, see rl.envs.utils.print_traj.
    """

    def __init__(self) -> None:
        # The raw unprocessed sequence of transitions encountered during the trajectory so far
        self.transitions: list[Transition] = []

        # The transitions are processed into a sequence of contexts, each of which has a list of tokens
        # and for each token the log_probs, source, and reward:
        self.tokens: list[list[int]] = [[]]
        self.log_probs: list[list[float]] = [[]]
        self.source: list[list[Literal["env", "agent"]]] = [[]]
        self.rewards: list[list[float]] = [[]]

        # The status of the current trajectory
        self.terminated: bool = False
        self.truncated: bool = False
        self._truncation_return: float | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.transitions) == 0

    @classmethod
    def from_transitions(
        cls,
        transitions: list[Transition],
        log_probs: list[list[float]] | None = None,
        truncation_return: float | None = None,
    ) -> Self:
        """
        Create a new Trajectory from a list of Transition objects.

        Args:
            transitions: List of Transition objects to include in the trajectory
            log_probs: Optional list of log probability lists, obtained from Trajectory.log_probs
            truncation_return: Optional float indicating the assumed return for the (unrealized) continuation of the trajectory.
              If a value is provided, self.truncate will be set to True.

        Returns:
            A new Trajectory containing all the provided transitions
        """
        trajectory = cls()

        for tr in transitions:
            trajectory.append(tr)

        if log_probs is not None:
            # We can't easily recover the per-transition log-probs from a trajectory, so we use the per-context transitions as stored in Trajectory
            assert all(
                len(tr_lps) == len(lps)
                for (tr_lps, lps) in zip(trajectory.log_probs, log_probs, strict=True)
            )
            trajectory.log_probs = log_probs

        if truncation_return is not None:
            trajectory.truncate(truncation_return)

        return trajectory

    def append(self, tr: Transition, log_probs: list[float] | None = None) -> None:
        assert not self.terminated, "Cannot append to a terminated trajectory."
        assert (
            tr.rewards is not None
        ), "Transition rewards must be defined when appending to trajectory."
        if log_probs is None:
            # treat actions as prob 1 if no log_probs are provided
            log_probs = [0.0] * len(tr.action)

        self.transitions.append(tr)

        self.tokens[-1] += tr.action
        self.log_probs[-1] += log_probs
        self.source[-1] += ["agent"] * len(tr.action)
        self.rewards[-1] += tr.rewards
        self.assert_valid()

        self.maybe_context_switch()

        self.tokens[-1] += tr.observation
        self.log_probs[-1] += [0.0] * len(tr.observation)
        self.source[-1] += ["env"] * len(tr.observation)
        self.rewards[-1] += [0.0] * len(tr.observation)
        self.assert_valid()

        self.terminated = tr.terminal

    def assert_valid(self) -> None:
        assert (
            len(self.tokens[-1])
            == len(self.source[-1])
            == len(self.rewards[-1])
            == len(self.log_probs[-1])
        ), (
            f"Trajectory attributes don't have the same length!\n"
            f"Tokens ({len(self.tokens[-1])}): {self.tokens[-1]}\n"
            f"Source ({len(self.source[-1])}): {self.source[-1]}\n"
            f"Rewards (({len(self.rewards[-1])}): {self.rewards[-1]}\n"
            f"Log probs ({len(self.log_probs[-1])}): {self.log_probs[-1]}\n"
        )

    @property
    def context(self) -> list[int]:
        """The context to be used by the policy to determine its next action."""
        return self.tokens[-1]

    @property
    def ncontexts(self) -> int:
        """The number of contexts in this trajectory"""
        return len(self.tokens)

    @property
    def nsteps(self) -> int:
        """
        The number of calls to env.step()
        This equals the number of transitions minus one for the initial transition returned by env.start()
        """
        return len(self.transitions) - 1

    def return_to_go(self) -> list[list[float]]:
        assert (
            self.terminated != self.truncated
        ), "Trajectory must be either terminated or truncated to compute return to go"
        assert not self.truncated or self._truncation_return is not None

        # Compute return-to-go as a reverse cumsum across contexts
        result: list[list[float]] = [
            [float("nan")] * len(self.rewards[ci]) for ci in range(self.ncontexts)
        ]
        r2g = self._truncation_return if self.truncated else 0.0
        assert r2g is not None
        for ci in reversed(range(self.ncontexts)):
            context_length = len(self.rewards[ci])
            for ti in reversed(range(context_length)):
                r2g += self.rewards[ci][ti]
                result[ci][ti] = r2g

        return result

    def truncate(self, truncation_return: float) -> None:
        """
        If we had to truncate the trajectory before hitting a terminal state, for instance because of a time limit
        or context length, we need to mark the trajectory as truncated and provide an assumed return for the (unrealized)
        future of the trajectory, as this information is required by many RL algorithms.
        Typically, we would use the reward assigned for failure as the truncation_return, or use a value function.

        NOTE: this function only marks the trajectory as truncated but does not actually truncate the sequences.
        """
        self.truncated = True
        self._truncation_return = truncation_return

    def maybe_context_switch(self) -> None:
        """
        This method decides based on the current trajectory whether to do a context switch.
        In that case, the method will append a new empty list to tokens, source, log_probs and rewards,
        representing the new context. Additionally, it may rewrite the latest observation
        self.transitions[-1].observation, which will serve as the prompt for the new context.

        Various context switching / history rewriting strategies may be implemented by
        overwriting this method. By default, a context switch is performed if the last transition
        has tr.context_switch == True
        """
        if self.transitions[-1].context_switch:
            self.tokens.append([])
            self.log_probs.append([])
            self.rewards.append([])
            self.source.append([])


def reference_rollout(
    env: Env,
    reward_fn: AbstractRewardFn,
    agent: Callable,
    start_args: dict | None = None,
    max_seq_len: int = 4096,
) -> Trajectory:
    """
    Simple reference implementation of a rollout, to be used for testing.
    """
    traj = Trajectory()
    state, tr = env.start(start_args)
    with state:
        traj.append(tr)
        while not tr.terminal and len(traj.context) < max_seq_len:
            max_gen = min(
                env.max_action_len(state),
                max_seq_len - len(traj.context),
            )
            action = agent(traj.context, max_gen)
            tr = env.step(state, action)
            tr.add_rewards(reward_fn(tr))
            assert tr.action == action
            traj.append(tr)

    return traj
