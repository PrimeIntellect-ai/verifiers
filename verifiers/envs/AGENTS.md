# Environments

**WIP** -- subject to change.

## Overview

Environment classes are organized into three categories:

| Category | Location | Description |
|----------|----------|-------------|
| **Core** | `verifiers/envs/` | Stable base classes and sandbox environments |
| **Integrations** | `verifiers/envs/integrations/` | Third-party library wrappers (lazy-loaded, require extras) |
| **Experimental** | `verifiers/envs/experimental/` | Newer environments with sharper edges |

Avoid storing global state in the environment. Instead, use the state argument to the environment's methods.

## Core Environments

Base classes for building environments:

- `SingleTurnEnv` - Single response Q&A tasks
- `MultiTurnEnv` - Multi-turn interactions (games, simulations, agents)
- `ToolEnv` - Stateless, idempotent tools passed as Python functions with all arguments exposed to the model. For managing additional state, use `StatefulToolEnv`.
- `StatefulToolEnv` - Tools requiring per-rollout state (e.g. sandbox ID). Use `update_tool_args` to inject state into tool calls. See `sandbox_env.py` for an example.
- `SandboxEnv` - Sandboxed container execution using `prime` sandboxes. All sandbox setup logic should be included in the start command and queued via `setup_state`, but not awaited—await resources only when first needed to overlap provisioning with rollout. See `python_env.py` for an example.
- `PythonEnv` - Persistent Python REPL in sandbox

## Integrations

Third-party library wrappers that require additional dependencies:

| Environment | Extra | Install Command |
|-------------|-------|-----------------|
| `TextArenaEnv` | `ta` | `uv add 'verifiers[ta]'` |
| `ReasoningGymEnv` | `rg` | `uv add 'verifiers[rg]'` |

When developing in the `verifiers` repo:
```bash
uv sync --extra ta   # for TextArenaEnv
uv sync --extra rg   # for ReasoningGymEnv
```

### TextArenaEnv

Wrapper for text-based TextArena environments (games, simulations). When adding new TextArena environments, investigate the `textarena` source code to determine the observation format. Often you'll want to re-render observations via `feedback_fn`, as `verifiers` doesn't allow overwriting past messages (only concatenation), and many TextArena games return full game state rather than turn-level diffs.

### ReasoningGymEnv

Wrapper for [reasoning-gym](https://github.com/reasoning-gym/reasoning-gym) procedural datasets. Supports single datasets or composite mixtures via `DatasetSpec`.

## Experimental

See `verifiers/envs/experimental/README.md` for documentation on:
- `GymEnv` - Universal Gym-compatible environment runner
- `MCPEnv` - MCP server integration
- `CliAgentEnv` - Custom agent code in sandboxes
- `HarborEnv` - Harbor-format task loading
- `RLMEnv` - Recursive Language Models
