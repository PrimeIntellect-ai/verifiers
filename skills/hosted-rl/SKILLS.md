---
name: prime-intellect-hosted-rl
description: >
  Build and train RL environments on Prime Intellect's Lab platform using verifiers.
  Use when working with: verifiers environments, prime CLI, Prime Intellect hosted training,
  RL reward functions, rubrics, or TOML training configs.
---

# Prime Intellect Lab: Hosted RL Training

## Quick Start

```bash
uv tool install prime
prime login
prime lab setup
prime env init my-env
prime env install my-env
prime eval run my-env -m openai/gpt-4.1-mini -n 10 -r 1
prime rl run configs/lab/my-env.toml
```

## Core Concepts

An **environment** = dataset + harness + rubric, packaged as a Python module exposing `load_environment()`.

A **rubric** wraps reward functions that score model completions.

**Hosted training** runs on Prime Intellect infrastructure via TOML configs.

## Reference Files

Load these as needed:

- **[ENVIRONMENTS.md](references/ENVIRONMENTS.md)** - Environment structure, dataset schema, environment types, tools, lifecycle hooks
- **[RUBRICS.md](references/RUBRICS.md)** - Reward functions, built-in rubrics, parsers
- **[CONFIGS.md](references/CONFIGS.md)** - TOML schemas for training and evaluation
- **[CLI.md](references/CLI.md)** - Common CLI commands quick reference

## Official Documentation

For current information (models, full API, CLI reference), always check:

- **Verifiers docs**: https://docs.primeintellect.ai/verifiers/overview
- **CLI reference**: https://docs.primeintellect.ai/cli-reference/introduction
- **GitHub (verifiers)**: https://github.com/PrimeIntellect-ai/verifiers
- **GitHub (prime-rl)**: https://github.com/PrimeIntellect-ai/prime-rl
- **Environments Hub**: https://app.primeintellect.ai/dashboard/environments

Run `prime rl models` for the current list of supported training models.

## Critical Rules

- Never override `rollout()` — use hooks (`setup_state`, `env_response`, `@vf.stop`, `@vf.cleanup`)
- Always create a `Rubric` with reward functions
- Heavy setup in `__init__()`, per-rollout state in `setup_state()`
- Environment variables: ONLY for API keys (validate with `vf.ensure_keys()`). Hardcode dataset names.
- Check canonical examples in `verifiers/envs/` before designing custom environments
