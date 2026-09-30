# Overview

verifiers runs agents on tasks and scores their work. Use it to evaluate models or train them with reinforcement learning.

These are the main pieces:

## Environment Hub

The [Environment Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) is a collection of tasksets you can install and run with verifiers.

## Taskset

A taskset loads tasks with its `load()` method. Each task has data, such as a prompt and reference answer, and code to set it up and score the result. `TaskData` holds the data; `Task` holds the code.

## Harness

A harness runs the model and its tools. Examples include Claude Code, Codex, and mini-swe-agent.

## Agent

An agent combines a model, a harness, and a runtime. The runtime is where the program runs, such as a local container or a remote sandbox. One agent run is called a **rollout** and produces a trace.

## Environment

An environment decides which agents run and in what order. Their traces form one **episode**. The default environment runs one agent; others can run several attempts, a judge, or a simulated user.

## Toolset

A toolset adds task-specific tools to the harness through MCP. The harness must support MCP to use them.

## Trace

A trace records the agent's messages, tool calls, scores, and errors. It also records each model call's settings, token usage, and timing. Training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) adds tokens and log probabilities using [renderers](https://github.com/PrimeIntellect-ai/renderers).

## Documentation

- [Getting started](getting_started.md) - How to install verifiers and the needed skills.
- [Architecture](architecture.md) — How verifiers works behind-the-scenes
- [Building environments](building-environments.md) — Build a package and separate agent execution from grading
- [Tasksets](tasksets.md) — How to create tasksets
  - [Harbor Tasksets](harbor.md) — How to create Harbor-based tasksets
- [Evaluation](evaluation.md) — How to evaluate tasksets
- [Runtimes](runtimes.md) — Containers, files, resources, and network access
- [Validation and trace inspection](debugging.md) — Check setup, diagnose failures, and re-score saved runs
- [Harnesses](harnesses.md) — How to build custom harnesses
- [Agent](agent.md) — How to run standalone agents
- [Env](env.md) — How to build multi-agent environments
- [GEPA](gepa.md) — Optimize a taskset's system prompt
