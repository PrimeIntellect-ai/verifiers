# AGENTS.md

This guide covers best practices for contributing to the core `verifiers` library and for and developing new environments (e.g. in `environments/`).

Other relevant `AGENTS.md` files:
- `tests/AGENTS.md`
- `verifiers/envs/AGENTS.md`

## Setup

We use `uv` for developing `verifiers`.
```bash
# Install uv (first time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup blank project if needed
uv init && uv venv --python 3.12
source .venv/bin/activate
```

## General Guidelines (Core + Environments)

### Code Style & Typing
- **Formatting**: Strict `ruff` enforcement. All PRs must pass `ruff check --fix .`
- **Typing**: Explicit types preferred
  - **OK**: `cast(...)`, `assert ...` for type narrowing
  - **SOMETIMES OK**: Untyped args for simple cases (e.g., reward functions) 
  - **NOT OK**: `# type: ignore` without strong justification

### Error Handling Philosophy  
- **Fail fast, fail loud** - No defensive programming or silent fallbacks
- **Minimize branching** - Prefer single code paths; every `if`/`try` needs justification
- **Example**: Missing API key → immediate failure, not fallback

## Core Repository Development

For PRs to `verifiers` core (e.g. `verifiers/verifiers/`):
```bash
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers

# CPU-only development:
uv sync

# GPU-based trainer development:
uv sync --all-extras && uv pip install flash-attn --no-build-isolation

# Install pre-commit hooks:
uv run pre-commit install
```

### Dependencies
- Avoid new core dependencies
- Use optional extras for non-essential features
- Exception: tiny deps that simplify widely-used code

### Testing  
- `uv run pytest` with discovery under `tests/`
- Write simple, deterministic unit tests
- Update tests when changing functionality

### Documentation
- Keep concise and actionable
- Update relevant pages when behavior changes
- Avoid content duplication
- See `docs/source/development.md` for authoritative walkthrough

### Scope
- Small, focused diffs
- One change per PR
- Backward compatibility is only desirable if it can be done without introducing excessive maintenance burden
- Delete dead code (don't guard it)

### Checklist

Before a PR:

```bash
# Run style + lint checks:
uv run ruff check --fix .
uv run pre-commit run --all-files
```

Ensure docs and tests are updated if necessary, and dead code is deleted.  Strive for minimal, surgical diffs.

## Developing Environments

Environment APIs live in `verifiers/envs/`. Actual environment implementations go in `environments/` (whether in the `verifiers` repo, or in a standalone project where you are developing new environments). Always reuse existing patterns.

### Quick Start

```bash
# Add verifiers
uv add verifiers            # core
uv add 'verifiers[all]'     # + training

# Scaffold new environment
vf-init new-environment

# Install + test
vf-install new-environment
vf-eval new-environment -n 5 -m gpt-4.1-mini
```

### Requirements
- Define `load_environment(...)` function
- Include `environments/<name>/pyproject.toml`
- Use module folders for multi-file projects
- Include upstream dependencies when porting existing projects

### Environment Patterns

Choose the appropriate base class from `verifiers/envs/` or by borrowing from other examples:

| Pattern | Base Class | When to Use | Key Methods |
|---------|------------|-------------|-------------|
| **Single-turn** | `SingleTurnEnv` | Q&A tasks, single response | Dataset + reward functions |
| **Multi-turn** | `MultiTurnEnv` | Games, simulations, agents | Add `is_completed`, `env_response` |
| **Stateless tools** | `ToolEnv` | Idempotent Python functions | Pass `tools: list[Callable]` |
| **MCP tools** | `MCPEnv` | MCP server integration | See `environments/mcp_env/` |
| **Stateful tools** | `StatefulToolEnv` | Tools requiring state | Use `setup_state`, `update_tool_args` |

**Important**:
- Never override `rollout()` - use the provided loop
- Always create a `Rubric` with reward functions for rewards/metrics
- Preprocess data in `load_environment()`
- Heavy setup in `__init__()`, per-rollout in `setup_state()`
- **Always** look for relevant canonical examples (in `verifiers/envs` or `environments`) before designing your own patterns

### Configuration Guidelines

- **Environment variables**: ONLY for API keys (document in README)
- **Hardcode**: Dataset names, URLs, defaults  
- **Arguments**: Only for essential customization via `load_environment()`
- **State keys**: Only rely on what your env explicitly sets

### State Management
Default state includes: `prompt`, `completion`, `responses`, `turn`, `timing`, `task`, `info`. 
Only depend on keys your environment explicitly manages.


### Checklist
- Guidelines here are followed
- Environment works with `vf-eval -s` test run, outputs look sensible