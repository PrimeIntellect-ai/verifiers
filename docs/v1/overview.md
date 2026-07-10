# Overview

verifiers is a framework for defining tasks, running agents and harnesses, scoring them on set tasks, and using those for evaluations and reinforcement learning.

The following concepts are important when creating or running environments, be it for evals or training:

## Environment Hub

The [Environment Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) is Prime Intellect's collection of user-created environments which are installable and ready to use with verifiers.

## Environment

An environment combines a _taskset_ and a _harness_. Often, it is sufficient to only define a _taskset_ in an environment, as harnesses are meant to be combined with tasksets.

## Taskset

A taskset is the collection and loader for the tasks—the _what_ of an environment. Each task combines a serializable `TaskData` row (prompt, files, references, resource requirements) with its task class's behavior (lifecycle hooks, tools, metrics, and rewards). The taskset's `load()` method constructs those objects and declares their task/config types through `Taskset[TaskT, ConfigT]`.

## Harness

A harness is the program the model is run in, e.g. Claude Code, Codex or mini-swe-agent.

## Toolset

A set of tools defined by the environment that are installed as MCP servers into the harnesses that support them.

## Trace

A trace records the message graph, rewards, metrics, errors, etc. When using verifiers for training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), it stores additional information such as tokens and logprobs, built incrementally using [renderers](https://github.com/PrimeIntellect-ai/renderers).

## Documentation

- [Getting started](getting_started.md) - How to install verifiers and the needed skills.
- [Architecture](architecture.md) — An overview about the architecture and runtime of verifiers
- [Environments](environments.md) — How to create environments
  - [Harbor Environments](harbor.md) — How to create Harbor-based environments
- [Evaluation](evaluation.md) — How to run said environments
- [Harnesses](harnesses.md) — How to build custom harnesses

For the documentation for legacy environments, go to [the v0 documentation](../v0/overview.md).
