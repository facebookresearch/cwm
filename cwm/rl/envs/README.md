# CWM RL Inference Environments

Envs in CWM define the environment dynamics of specific "problems" that our agents can solve.
They are abstractions of (configurable) POMDPs, so we can cover the general case of environments that are only partially observable.

The base environment interface is defined by just two methods, `start` and `step`, as well as an auxiliary action length limiter.

Their meanings are aligned with traditional RL environment design:

- `start` returns the initial state of the environment, as well as an initial observation for the agent;
- `step` maps a state and an agent action to a new state and observation of the environment reaction.

`step` is intentionally implemented as "sort of" functional: the environment does not track its own state, but instead expects one to be provided at each step.
This makes branching and cloning much more straightforward, as a single instance of an env need ever be referenced to have access to the full dynamics, and only states need to be cloned.
**NB** _however_ there is no guarantee (mostly because of performance reasons) that the input state of `step` will be left unmodified, i.e. `step` is not _pure_.
If tracking of steps is needed (e.g. for branching rollouts), an explicit clone of each step should be used.

The only strong assumption we are making in these environments is that the action space is limited to sequences of _tokens_, i.e. we are restricting ourselves to tokenisable spaces (in the most common cases text, although this need not always be the case), implying that communication between the environment and the agent will most commonly happen via a shared tokenizer.
This is not _strictly_ necessary, as an env could be implemented purely in token space and not need a tokenizer, but this is likely not advisable for obvious reasons.

## Abstractions

On top of the Env abstraction above we provide two types strictly paired with the Env.
These are generic abstractions that apply to any Env subtype, regardless of their inner functionality, as they are directly related to the environment methods; for more specific Env abstractions see below.

### Transition

Environment observations are bundled into a more expressive abstraction called a transition, which you may have noticed from the snippet above is the actual return type of `start` and `step`.

The main purpose of a transition is to bundle environment response observations, observed rewards for the current action, and state whether the episode has terminated.
This information is then used to build a...

### Trajectory

A trajectory is essentially the history tracking of an episode's observations.

Trajectories are constructed by the rollout methods, and are mostly a utility provided for this purpose - there is no specific restriction to using this abstraction when keeping track of an episode's history in custom rollouts, but it should be quite useful to use in all but the most exotic cases.

## Dialogs

Dialog environments are a special (though still quite general) form of environment, consisting of several conversational turns between the environment (most commonly dubbed the "user") and the agent (usually the "assistant").

The [DialogEnv](/cwm/rl/envs/dialog.py) provides some utilities for generating dialogs, but it's not a strict necessity for building dialog envs in general - any env that relies on messages being exchanged, usually using a header identifying the role in the conversation for each turn and using the tokenizer's `{encode|decode}_message` methods, is considered a dialog env.

See also the `cwm/rl/envs/utils/` package for some dialog utilities that should help build at least some code dialog environments with minimal extra code and avoid boilerplate.

## Configuration

As mentioned in the [RL README](/cwm/rl/README.md), environments are not quite full POMDPs, as they need to be configured with a data source of initial prompts to provide to `start` - in this they differ from the older classical RL environments, where the (distribution of) initial state(s) was commonly part of the environment itself.

We made this decision because the same env dynamics can be applied to a variety of different initial prompts coming from different datasets, and the split in implementation helps with modularity.
Plus it is reasonable to expect that datasets will change over time more than the dynamics of environments, so allowing flexibility in their configuration will limit the amount of code churn.

The task configuration and environments are registered and built with utilities available in `cwm/rl/envs/config.py`.
