# CWM RL inference code

This folder contains all the code for agentic / reasoning evaluation of Code World Model.

## Code

* `rl/lib/` contains the generation API and trajectory handling code.
* `rl/envs/` contains environment definitions to run inference on.
Envs in our case define only environment dynamics, i.e. a `start` method for initialising the state and create an initial observation for the agent, and a `step` method to evolve the state through an agent's action and provide a subsequent observation.
Extending envs with a data source for information to provide to `start` fully defines the [POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) of a classical RL environment, which here we call a `task`.
* `rl/swerl/` provides a tool-based execution environment for **evaluating** LLM agents on software engineering tasks.
