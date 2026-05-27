# v1 Environment Contract

This file is the authoring contract for environment builders using the v1
Taskset/Harness pattern. Use it as the first source of truth when adding,
migrating, or reviewing v1 environments.

The contract is intentionally strict. When a rule feels stricter than necessary,
prefer the strict interpretation and relax it only after a concrete environment
proves the looser surface is needed.

## Contract Surface

1. Environment modules MUST expose `load_environment(config: vf.EnvConfig)`.
   Do not subclass `vf.EnvConfig` just to narrow child config types.
2. Environment modules SHOULD construct `vf.Env` from component loaders:
   `vf.load_taskset(config=config.taskset)` and, when needed,
   `vf.load_harness(config=config.harness)`.
3. Environment modules with custom taskset fields MUST expose
   `load_taskset(config: TasksetConfigType)`. Environments with custom harness
   fields MUST expose `load_harness(config: HarnessConfigType)`.
4. Public child loader config parameters MUST be typed as one concrete Pydantic
   config type. Do not construct config objects in function signatures, and do
   not advertise unions of mappings, base configs, and specific configs.
5. `load_environment` MUST NOT mirror taskset or harness fields as keyword
   arguments. The root env loader receives the typed envelope; the child configs
   own all environment-specific settings.

## Why The Types Matter

Verifiers uses loader annotations to decide how config should be parsed before
your loader runs. The type annotation is not cosmetic.

1. If your environment exposes `load_taskset(config: MyTasksetConfig)`, TOML
   `[env.taskset]` validates directly against `MyTasksetConfig`.
2. If your environment exposes `load_harness(config: MyHarnessConfig)`, TOML
   `[env.harness]` validates directly against `MyHarnessConfig`.
3. Without a child factory, the matching section validates against the base
   `vf.TasksetConfig` or `vf.HarnessConfig`.
4. The `load_environment` envelope stays loose as `vf.EnvConfig`. The framework
   coerces `config.taskset` and `config.harness` inside `vf.load_taskset(...)`
   and `vf.load_harness(...)` using the child factory annotations.
5. `vf.Env(taskset=vf.load_taskset(config=config.taskset))` is the
   default construction path.
6. Root env kwargs behave differently from TOML child sections. TOML
   `[env.taskset]` and `[env.harness]` route into the env config envelope; CLI
   `-a` passes loader kwargs. This is why CLI child config overrides must be
   nested under `{"config": ...}`.

## Config Envelope

`vf.EnvConfig` is an envelope with exactly two child sections:

```python
def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

1. `EnvConfig.taskset` is typed as `vf.TasksetConfig`.
2. `EnvConfig.harness` is typed as `vf.HarnessConfig`.
3. `EnvConfig` subclasses MUST NOT define additional root fields. If a field is
   specific to the environment, it belongs on the taskset or harness config that
   owns the behavior.
4. `taskset` and `harness` MUST NOT be `None`. Omit the section to use the
   default config.

## Ownership Rules

Taskset configs own task behavior:

- tasks, eval tasks, splits, IDs, cache locations, and dataset sizing
- task prompts, user behavior, tools, tool bindings, and task-local objects
- task-specific scoring, metrics, rewards, advantages, and policy weights
- task-specific difficulty, category, filtering, and sampling controls

Harness configs own rollout and execution behavior:

- framework programs, command programs, sandboxes, and lifecycle wiring
- model/client/sampling defaults for standalone harnesses
- harness prompts, harness toolsets, harness bindings, and harness-local objects
- rollout limits, trajectory handling, and execution-specific scoring hooks

Do not put harness settings on taskset configs or taskset settings on harness
configs. Do not add an environment-level config class just to carry fields that
already belong to a taskset or harness.

## Canonical Patterns

### Default Harness

Use this when the taskset owns the environment and the base harness is enough.

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self) -> vf.Tasks:
        rows = [
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "answer": "4",
                "split": "train",
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]

    @vf.reward(weight=1.0)
    async def exact(self, task, state) -> float:
        return float(state.get("answer") == task["answer"])


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
```

### Custom Harness

Use this when the harness owns a reusable execution mechanism or config surface.

```python
import verifiers as vf



async def run(task, state) -> vf.State:
    return state


class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self) -> vf.Tasks:
        rows = [
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "answer": "4",
                "split": "train",
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]


class MyHarnessConfig(vf.HarnessConfig):
    timeout_seconds: int = 120
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env:run")


class MyHarness(vf.Harness[MyHarnessConfig]):
    config: MyHarnessConfig


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_harness(config: MyHarnessConfig) -> MyHarness:
    return MyHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Beginner templates should not subclass `EnvConfig`, pass loader functions to
`vf.load_taskset`, assert child config types, put defaults on runtime subclasses,
or pass constructor kwargs to `vf.Taskset(...)`.

## External Configuration

TOML config routes v1 environment config through child sections:

```toml
[env.taskset]
split = "eval"

[env.harness]
max_turns = 20
```

CLI `-a` passes loader kwargs. For v1 child config overrides, nest under the
`config` loader argument:

```bash
prime eval run my-env -a '{"config":{"taskset":{"split":"eval"},"harness":{"max_turns":20}}}'
```

Raw mappings are boundary values for TOML, CLI, JSON, and other external input
surfaces. Inside Python environment code, use typed config objects.

## Type Boundaries

1. Import the public API with `import verifiers as vf`.
2. Use Pydantic config models for structured configuration.
3. Treat `Mapping[str, object]` as an explicit boundary type. Accept it only for
   intentionally dynamic payloads such as task rows, protocol messages,
   sandbox/program specs, or config fields that store arbitrary user objects.
4. Prefer a named alias such as `ConfigMap`, `TaskRow`, or `Objects` over
   spelling a broad mapping type in a user-facing signature.
5. Do not use raw `Any` in v1 environment code. If a value is intentionally
   arbitrary, give that boundary a named type in `verifiers.v1.types`.
6. Avoid `object`, broad unions, and untyped mappings unless arbitrary user data
   is genuinely the contract.
7. Do not bypass Pydantic validation for v1 config fields. Treat
   `SkipValidation` and equivalent escape hatches as outside the contract unless
   a boundary explicitly requires unvalidated framework-owned values.

## Environment Structure

1. Keep environment files as wiring: taskset construction, harness construction,
   and small policy choices that compose the two.
2. Prefer v1-native framework objects over stdlib-shaped user code. User-facing
   environment code should read as Verifiers code first.
3. Keep shared dependencies behind the taskset or harness that owns them.
   Bindings are the canonical injection path for shared resources. Prefer
   serializable factory paths in config; Python-only construction may use
   factory callables directly. Required Taskset and Toolset factory parameters
   must be supplied through bindings. Do not pass pre-initialized resource
   objects through environment loaders.
4. Compose related categories inside one taskset only when they share the same
   harness lifecycle and scoring contract.
5. Expose explicit typed loaders for separate v1 envs when categories need
   different tasksets, harnesses, or lifecycle behavior.
6. Keep dual v0/v1 loaders explicit only while migration is intentionally
   dual-stack.
7. Do not wrap v1 `Env` objects in the v0 `EnvGroup`.
8. Do not add thin intermediate harness types that only restate `Harness`.
   Reusable command agents should be direct `Harness` subclasses with their own
   typed config only when they have real behavior to own.
9. Do not add heterogeneous `TasksetGroup` routing as a substitute for a real v1
   suite abstraction.
10. Avoid module globals in environment files. Acceptable globals are imports,
    immutable literals, factory functions, and carefully managed process-level
    resource constraints such as locks or semaphores.
11. Do not make users manipulate paths, package resources, or other stdlib
    details when the framework can express the intent directly.

## Naming

1. Config names should describe the thing being configured, not the layer that
   happens to carry it.
2. Do not use vague names that only repeat the component name. Prefer
   `repo_ref`, `env_vars`, or `dataset_split` over names like `rlm_ref`,
   `rlm_env`, or `config`.
3. Public fields should be stable enough to appear in TOML and examples. If a
   field is still experimental or internal, keep it behind a less public
   abstraction.

## Contract Review

Before adding or changing a v1 environment, verify that:

1. `load_environment` has one public config argument.
2. The env config root has only `taskset` and `harness`.
3. Every environment-specific field belongs to either the taskset config or the
   harness config.
4. `load_taskset` and `load_harness` use one concrete config type each.
5. Raw mappings appear only at external or intentionally dynamic boundaries.
6. Validation bypasses, broad unions, and root env kwargs are absent unless the
   file has a clear boundary-level reason for them.
