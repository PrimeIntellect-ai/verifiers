---
name: create-environments
description: Create or migrate verifiers environments for the Prime Lab ecosystem. Use when asked to build a new environment from scratch, port an eval or benchmark from papers or other libraries, start from an environment on the Hub, or convert existing tasks into a package that exposes load_environment and installs cleanly with prime env install.
---

# Create Environments

## Goal
Build production-quality verifiers environments that work immediately in the Prime ecosystem: install, load, evaluate, and train without hidden setup.

## Start With Ecosystem Paths
1. Prefer ecosystem-native setup before custom scaffolding.
2. Use this default loop:
```bash
prime env init my-env
prime env install my-env
prime eval run my-env -m gpt-4.1-mini -n 5
```
3. Prefer an existing environment as a starting point when possible:
```bash
prime env list --search "keyword"
prime env info owner/name
prime env install owner/name
```
4. For repository examples, use repo install when available:
```bash
prime env install math-python --from-repo
```
5. Encourage users to keep endpoint aliases in `configs/endpoints.toml` so smoke tests can switch models quickly.
6. Ask users whether they want instruct or reasoning models for validation.
7. Instruct-first smoke choices: `gpt-4.1` series, `qwen3` instruct series.
8. Reasoning validation choices: `gpt-5` series, `qwen3` thinking series, `glm` series.

## Build Modes

### 1. Build From Scratch
1. Define task contract first: prompt shape, allowed tools, stop conditions, rubric outputs, metrics.
2. Select the smallest correct base class:
- `SingleTurnEnv` for one-response tasks.
- `MultiTurnEnv` for custom interaction loops.
- `ToolEnv` or `MCPEnv` for stateless tools.
- `StatefulToolEnv` for per-rollout resources.
3. Implement `load_environment(...) -> vf.Environment` with explicit arguments.
4. Add `pyproject.toml` defaults in `[tool.verifiers.eval]` only when stable.

### 2. Port From Another Library, Project, or Paper
1. Create a strict source-to-target mapping before coding:
- dataset rows and splits
- prompt rendering and role ordering
- tool I/O schema and stop logic
- scoring math and aggregation
- pass/fail thresholds and special cases
2. Preserve one-to-one logical equivalence for what the model sees and what gets scored.
3. Never invent unresolved formatting decisions. Ask the user to decide explicitly.
4. Benchmark runtime and remove avoidable bottlenecks before handoff.

### 3. Start From Hub Environment
1. Install or pull the closest baseline:
```bash
prime env install owner/name
prime env pull owner/name -t ./tmp-env
```
2. Keep proven interfaces stable unless a migration is deliberate and explicit.
3. Re-run smoke evals after each major change.

## Non-Negotiable Quality Rules
1. Use deterministic, well-defined reward checks or LLM judges.
2. Avoid best-effort deterministic heuristics such as keyword style checks except as an explicit last resort with user sign-off.
3. Make environments self-contained after install. Do not require users to run background servers before `load_environment()`.
4. Manage external resources inside the environment lifecycle.
5. Validate required secrets in `load_environment()` via `vf.ensure_keys(...)`.
6. Surface feature limits directly. Do not ship hacky workarounds without explicit user approval.

## Verification Gate
Run these before claiming completion:
```bash
prime env install my-env
prime eval run my-env -m gpt-4.1-mini -n 5
prime eval run my-env -m gpt-4.1-mini -n 50 -r 1 -s
```
If multi-turn or tool-heavy, also run with higher rollouts:
```bash
prime eval run my-env -m gpt-4.1-mini -n 30 -r 3 -s
```

## Publish Gate Before Large Evals Or Training
1. After smoke tests pass and behavior is stable, recommend pushing to Hub before large evals or RL training.
2. Ask the user explicitly whether visibility should be `PUBLIC` or `PRIVATE`.
3. Use:
```bash
prime env push --path ./environments/my_env --visibility PUBLIC
```
or
```bash
prime env push --path ./environments/my_env --visibility PRIVATE
```
4. For hosted or large-scale workflows, prefer running with the Hub slug after push:
```bash
prime eval run owner/my-env -m gpt-4.1-mini -n 200 -r 3 -s
```

## Deliverable Format
Report:
1. Environment ID and path.
2. Exact install and eval commands used.
3. Port-equivalence notes if migrated.
4. Any unresolved user decisions that block strict fidelity.
