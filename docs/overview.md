# Overview

Verifiers is a framework for defining tasks, running agents and harnesses, scoring them on set tasks, and using those for evaluations and reinforcement learning.

The following concepts are important when creating or running environments, be it for evals or training:

## Environment Hub

The [Environment Hub] is Prime Intellects collection of (user-created) environments which are installable and ready to use with Verifiers.

## Environment

An environment combines a _taskset_ and a _harness_. Often, it is sufficient to only define a _taskset_ in an environment, as harnesses are meant to be combined with tasksets.

## Taskset

A taskset is a collection of tasks, i.e., the _what_. A task is a prompt, a set of files, etc. A taskset also defines the scoring mechanism (known as reward).

## Harness

A harness is the program the model is run in, e.g. Claude Code, Codex or mini-swe-agent.

## Toolset

A set of tools defined by the environment that are installed as MCP servers into the harnesses that support them.

## Trace

A trace records the message graph, rewards, metrics, errors, etc. When using Verifiers for training, it stores additional information such as logprobs.

## Documentation

- [Getting started](getting_started.md) - How to install the CLI and the needed skills.
- [Architecture](architecture.md) — An overview about the architecture and runtime of Verifiers
- [Environments](environments.md) — How to create environments
  - [Harbor Environments](harbor.md) — How to Harbor-based environments
- [Evaluation](evaluations.md) — How to run said environments
- [Training](training.md) — How to train on the environments
- [Harnesses](harnesses.md) — How to build custom harnesses
