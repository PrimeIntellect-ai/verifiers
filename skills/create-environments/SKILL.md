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
prime eval run my-env -m openai/gpt-4.1-mini -n 5
```
Use `prime env init my-env --with-harness` when the environment owns an
explicit harness.
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
- V1 `vf.Env` with `vf.Taskset[Config]`/`vf.Harness[Config]` for the current taskset/harness environment pattern that separates the task collection from the rollout runner. Use this for new taskset/harness work that needs config-driven metrics, rewards, toolsets, user functions, endpoint interception, or sandboxed Python/command programs. Framework programs should build clients from `state.get_endpoint_config(api="chat")`.
3. For v1, import `verifiers as vf`, define typed taskset/harness config subclasses when needed, bind them with `class MyTaskset(vf.Taskset[MyTasksetConfig])` and `class MyHarness(vf.Harness[MyHarnessConfig])`, and expose `load_environment(config: MyEnvConfig) -> vf.Env`.
4. For v0 environments, keep the existing `vf.Environment` patterns and preserve v0 compatibility.
5. Add `pyproject.toml` defaults in `[tool.verifiers.eval]` only when stable.

### V1 Authoring Rules
1. Keep v1 environment entrypoints tiny: `import verifiers as vf`, define top-level `@vf.reward` / `@vf.metric` functions, define `TasksetConfig` / `HarnessConfig` subclasses for user-facing knobs, define `Taskset[Config]` / `Harness[Config]` classes, then expose `load_environment` with explicit `vf.Env(taskset=MyTaskset(config=config.taskset), harness=MyHarness(config=config.harness))` objects.
2. Keep shared dependencies behind the taskset or harness that owns them. Use bindings as the canonical injection path; prefer serializable loader paths for bound objects in config, and use no-arg loader callables only for Python-only construction. Do not pass already-instantiated resource objects through environment loaders. Do not introduce v1 Parser/Rubric wrappers; parsing is ordinary Python.
3. Use `vf.get_messages(state.get("completion") or [], role="assistant")` when reading state completions. The helper returns typed message objects and should not receive `None`.
4. Use `program.channels` for v1 program protocol/channel selection. Do not use stale `program.tools` terminology.
5. Use `load_taskset(config)` / `load_harness(config)` only when they make the explicit object boundary clearer. Put behavior on the taskset or harness class, and use typed config objects as the only construction values.

### V1 Taskset/Harness Shape
1. Put task data, task-owned tools, user behavior, metrics, rewards, and task-specific configuration on the `Taskset`.
2. Use the base `vf.Harness` unless the harness owns a reusable execution adapter such as a CLI, framework program, sandboxed program, or nested harness flow.
3. Avoid one-off harness classes whose only purpose is to hold task behavior. That behavior belongs behind the taskset.
4. Keep small example environments direct. Do not add private helper layers, duplicate loader paths, or optional knobs unless they clarify a real reusable boundary.
5. Use the current config shape consistently:
```toml
[[env]]
id = "owner/my-env"

[env.taskset]
num_examples = 100
split = "train"

[env.harness]
max_turns = 8
```
6. In code, use the current class-based config shape:
```python
import verifiers as vf


@vf.reward(weight=1.0)
async def contains_answer(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    _default_rewards = (contains_answer,)

    def rows(self) -> list[dict[str, object]]:
        rows = [
            {
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "split": "train",
                "max_turns": 1,
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]


class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig = MyTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_environment(config: MyEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config.taskset),
        harness=vf.Harness(config=config.harness),
    )
```
7. Do not add root env config knobs. Put settings as leaf fields on the taskset or harness config that owns them.

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
