# Overview

verifiers is a framework for defining tasks, running agents and harnesses, scoring them on set tasks, and using those for evaluations and reinforcement learning.

The following concepts are important when creating or running tasksets, be it for evals or training:

## Environment Hub

The [Environment Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) is Prime Intellect's collection of user-created tasksets which are installable and ready to use with verifiers.

## Taskset

A taskset is the collection and loader for the work to evaluate or train on. Each task combines a serializable `TaskData` row (prompt, files, references, resource requirements) with its task class's behavior (lifecycle hooks, tools, metrics, and rewards). The taskset's `load()` method constructs those objects and declares their task/config types through `Taskset[TaskT, ConfigT]`.

## Harness

A harness is the program the model is run in, e.g. Claude Code, Codex or mini-swe-agent.

## Environment

The control flow between agents: how many run on one task, in what order, judged how across the finished set. The default is the single-agent case; `--env.id` pairs a reusable interaction (best-of-n, a grader) with any taskset. One env-rollout is one episode — a flat `list[Trace]`, one agent-stamped trace per run, linked through each trace's shared `episode` stamp (`EpisodeInfo`).

## Agent

An `Agent` is a reusable (harness × model × runtime policy) value with one executable arrow — `agent.run(task) -> Trace` — the building block environments hand to `run()`, and a scripting surface of its own (see [Agent](agent.md)).

## Toolset

A set of tools defined by the taskset that are installed as MCP servers into the harnesses that support them.

## Trace

A trace records the message graph, rewards, metrics, errors, and one per-call record (`ModelCall`) per provider exchange (its model, sampling, finish reason, usage, timing, and any error), etc. When using verifiers for training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), it stores additional information such as tokens and logprobs, built incrementally using [renderers](https://github.com/PrimeIntellect-ai/renderers).

## Documentation

- [Getting started](getting_started.md) - How to install verifiers and the needed skills.
- [Architecture](architecture.md) — An overview about the architecture and runtime of verifiers
- [Tasksets](tasksets.md) — How to create tasksets
  - [Harbor Tasksets](harbor.md) — How to create Harbor-based tasksets
- [Evaluation](evaluation.md) — How to evaluate tasksets
- [Harnesses](harnesses.md) — How to build custom harnesses
- [Agent](agent.md) — run tasks, place runs into shared sandboxes, chain traces into new tasks
- [Multi-agent environments](environments.md) — Multiple agents, the control flow between them, and cross-agent rewards

For the documentation for legacy environments, go to [the v0 documentation](../v0/overview.md).
