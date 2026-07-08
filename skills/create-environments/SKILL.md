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
prime env init my-env --v1
prime env install my-env
prime eval run my-env -m openai/gpt-4.1-mini -n 5
```
Use `prime env init my-env --v1 --with-harness` when the environment owns an
explicit reusable harness.
3. Treat `prime eval run` as the canonical eval path. It saves results automatically, so do not add `--skip-upload` unless the user explicitly requests that deviation.
4. Prefer an existing environment as a starting point when possible:
```bash
prime env list --search "keyword"
prime env info owner/name
prime env install owner/name
```
5. For repository examples, use repo install when available:
```bash
prime env install math-python --from-repo
```
6. Encourage users to keep endpoint aliases in `configs/endpoints.toml` so smoke tests can switch models quickly.
7. Ask users whether they want instruct or reasoning models for validation.
8. Instruct-first smoke choices: `gpt-4.1` series, `qwen3` instruct series.
9. Reasoning validation choices: `gpt-5` series, `qwen3` thinking series, `glm` series.

## Build Modes

### 1. Build From Scratch
1. Define task contract first: prompt shape, allowed tools, stop conditions, rubric outputs, metrics.
2. Select the smallest correct base class:
- `SingleTurnEnv` for one-response tasks.
- `MultiTurnEnv` for custom interaction loops.
- `ToolEnv` or `MCPEnv` for stateless tools.
- `StatefulToolEnv` for per-rollout resources.
- `CliAgentEnv` for running agent binaries in sandboxes with API interception. Override `get_sandbox_resources(state)` for per-instance resources, `build_env_vars(state)` for custom env vars.
- V1 `vf.Task`/`vf.Taskset` (`import verifiers.v1 as vf`) for the current task-centric pattern: the `Task` subclass carries the data fields AND the behavior — `@vf.reward`/`@vf.metric`/`@vf.group_reward`/`@vf.stop` signals, `tools()`/`user()` wiring, `setup`/`finalize`/`validate` hooks — and the `Taskset` is a thin loader (config in, tasks out). Use this for new work that needs typed per-task data, config-plugged judges, MCP toolsets, user simulators, or in-sandbox verification.
3. For v1, start from the generated template. Edit the `Task` subclass for task fields and behavior (`@vf.*` signals, `tools()`, `user()`, lifecycle hooks), `TasksetConfig` for user-facing knobs, and `Taskset.load()` to build the task list — one concrete task type per taskset, enforced at load. Add a harness class only for reusable execution behavior.
4. Export the plugin classes from the package's `__init__.py` — the loaders select them by id (`--taskset.id`, `--harness.id`); there is no `load_environment` shim in v1:
```python
from my_env.taskset import MyTaskset

__all__ = ["MyTaskset"]
```
5. For v0 environments, keep the existing `vf.Environment` patterns and preserve v0 compatibility.
6. Add `pyproject.toml` defaults in `[tool.verifiers.eval]` only when stable.

### V1 Authoring Rules
1. Keep v1 environment entrypoints tiny: `import verifiers.v1 as vf`, a `TasksetConfig` subclass for user-facing knobs, a `Task` subclass for the data + behavior, a `Taskset[MyTask, MyConfig]` subclass implementing `load`, an optional `Harness` — all exported via `__all__`. No loader functions, no `load_environment` shim; selection is by id.
2. Resolve config at load time and bake the results into task fields, so tasks come out self-contained — hooks and rewards read what they need off `self`. Do not pass live objects through config. Do not introduce Parser/Rubric wrappers; parsing is ordinary Python.
3. Read the model's output off the typed trace (`trace.last_reply`, `trace.messages`). Read another signal's recorded result with `trace.metric(name, default)`. Per-rollout runtime state lives on `trace.state`, typed via `Task[MyState]` — never on the frozen task.
4. Plug config-driven grading through `TasksetConfig.judges` (built-ins `reference` / `rubric`, or a judge package id) instead of hand-rolling LLM-judge rewards; judges run after the task's own `@reward`s.
5. Put settings as leaf fields on the taskset or harness config that owns them. A field must not shadow a Task method or an inherited signal name (`user` -> `user_config`) — that is rejected at class definition.

### V1 Taskset/Harness Shape
1. Put task data, tools, user behavior, metrics, and rewards on the `Task` subclass; the `Taskset` is just the loader (`load()` + config + judges). One concrete task type per taskset.
2. Use the base `vf.Harness` unless the harness owns a reusable execution adapter such as a CLI, framework program, sandboxed program, or nested harness flow.
3. Avoid one-off harness classes whose only purpose is to hold task behavior. That behavior belongs behind the taskset.
4. Keep small example environments direct. Do not add private helper layers, duplicate loader paths, or optional knobs unless they clarify a real reusable boundary.
5. Use the current config shape consistently:
```toml
[[eval]]
env_id = "owner/my-env"

[eval.taskset]
num_tasks = 100

[eval.harness]
max_turns = 8
```
For package-only composition, omit `env_id` and select loader packages through
child config ids:
```toml
[[eval]]

[eval.taskset]
id = "harbor"

[eval.harness]
id = "mini-swe-agent"
```
6. In code, use the current class-based config shape:
```python
import verifiers.v1 as vf


class ReverseTask(vf.Task):
    answer: str

    @vf.reward(weight=1.0)
    async def exact(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip() == self.answer)


class ReverseConfig(vf.TasksetConfig):
    num_tasks: int = 5
    """How many tasks to build."""


class ReverseTaskset(vf.Taskset[ReverseTask, ReverseConfig]):
    def load(self) -> list[ReverseTask]:
        words = ["abc", "hello", "prime"][: self.config.num_tasks]
        return [
            ReverseTask(idx=i, prompt=f"Reverse {word!r}.", answer=word[::-1])
            for i, word in enumerate(words)
        ]
```
7. Use `prime env init my-env --v1` as the reference shape when an implementation starts to drift.

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
prime eval run my-env -m openai/gpt-4.1-mini -n 5
prime eval run my-env -m openai/gpt-4.1-mini -n 50 -r 1 -s
```
If multi-turn or tool-heavy, also run with higher rollouts:
```bash
prime eval run my-env -m openai/gpt-4.1-mini -n 30 -r 3 -s
```
For repo example environments, also use the package-install path when packaging or dependencies changed:
```bash
uv run pytest tests/test_envs.py -k my_env -vv
```

## Publish Gate Before Large Evals Or Training
1. After smoke tests pass and behavior is stable, recommend pushing to Hub before large evals or RL training.
2. Ask the user explicitly whether visibility should be `PUBLIC` or `PRIVATE`.
3. Use:
```bash
prime env push my-env --visibility PUBLIC
```
or
```bash
prime env push my-env --visibility PRIVATE
```
4. For hosted or large-scale workflows, prefer running with the Hub slug after push:
```bash
prime eval run owner/my-env -m openai/gpt-4.1-mini -n 200 -r 3 -s
```

## Synthetic Data
1. Ask users for preferences on which LLMs to use for synthetic data generation and curation before implementation.
2. Prefer generating synthetic data from raw source documents whenever possible instead of relying only on hand-authored prompts.
3. Use LLM orchestration (planner/generator/validator loops) to improve sample quality and diversity.
4. Use back-translation: start from complete materials and decompose them into incomplete tasks, criteria, or partial artifacts that the model must reconstruct.
5. Use fan-out subtopic sampling from LLMs to expand coverage and avoid overfitting to a narrow slice of the domain.

## Deliverable Format
Report:
1. Environment ID and path.
2. Exact install and eval commands used.
3. Port-equivalence notes if migrated.
4. Any unresolved user decisions that block strict fidelity.
