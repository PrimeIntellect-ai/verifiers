# Experimental Environment Classes

This folder contains newer and more experimental environment classes that may still have some sharper edges. They are subject to change and may be removed without notice.

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## CliAgentEnv

Environment for running full agent code inside sandboxes. Extends `MultiTurnEnv` to reuse the rollout loop, but intercepts the agent's API requests via an HTTP proxy server. Each agent request triggers one rollout step. Useful for evaluating autonomous agents that make their own API calls.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.
