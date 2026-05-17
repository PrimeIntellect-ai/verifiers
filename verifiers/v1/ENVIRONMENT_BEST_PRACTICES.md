# v1 Environment Contract

This file is the authoring contract for environment builders using the v1
Taskset/Harness pattern. Use it as the first source of truth when adding,
migrating, or reviewing v1 environments.

The contract is intentionally strict. When a rule feels stricter than necessary,
prefer the strict interpretation and relax it only after a concrete environment
proves the looser surface is needed.

## Contract Surface

1. Environment modules MUST expose `load_environment(config: EnvConfigType)`.
   `EnvConfigType` is `vf.EnvConfig` for base taskset/harness configs, or an
   environment-local `vf.EnvConfig` subclass that binds concrete child config
   types.
2. Environment modules SHOULD expose `load_taskset(config: TasksetConfigType)`
   whenever the taskset has any reusable construction logic.
3. Environment modules SHOULD expose `load_harness(config: HarnessConfigType)`
   when the harness has reusable construction logic or a custom config type.
   Environments using the default harness MAY construct `vf.Harness` directly in
   `load_environment`.
4. Public `load_taskset` and `load_harness` config parameters MUST be typed as
   one concrete Pydantic config type with an explicit config-object default such
   as `config: MyTasksetConfig = MyTasksetConfig()`. Do not advertise unions of
   mappings, base configs, and specific configs.
5. `load_environment` MUST NOT mirror taskset or harness fields as keyword
   arguments. The root env loader receives the typed envelope; the child configs
   own all environment-specific settings.

## Why The Types Matter

Verifiers uses loader annotations to decide how config should be parsed before
your loader runs. The type annotation is not cosmetic.

1. If `load_environment` is annotated with `config: vf.EnvConfig`, TOML
   `[env.taskset]` validates against `vf.TasksetConfig` and `[env.harness]`
   validates against `vf.HarnessConfig`.
2. If your environment has custom taskset fields, `load_environment` MUST use an
   `EnvConfig` subclass whose `taskset` annotation is your concrete
   `TasksetConfig` subclass. Otherwise those TOML fields are rejected before
   your loader can convert or forward them.
3. If your environment has custom harness fields, the same rule applies to the
   `harness` annotation.
4. The config object that reaches `load_environment` is already validated and
   typed. Do not reconstruct child config objects just to recover their type.
5. `load_taskset(config: MyTasksetConfig)` and
   `load_harness(config: MyHarnessConfig)` make the reusable construction
   contract explicit. They also give tests and examples one stable place to
   exercise child config behavior.
6. Root env kwargs behave differently from TOML child sections. TOML
   `[env.taskset]` and `[env.harness]` route into the env config envelope; CLI
   `-a` passes loader kwargs. This is why CLI child config overrides must be
   nested under `{"config": ...}`.

## Config Envelope

`vf.EnvConfig` is a typed envelope with exactly two child sections:

```python
class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig = MyTasksetConfig()
    harness: MyHarnessConfig = MyHarnessConfig()
```

1. `EnvConfig.taskset` MUST be typed as a `vf.TasksetConfig` subclass.
2. `EnvConfig.harness` MUST be typed as a `vf.HarnessConfig` subclass.
3. `EnvConfig` subclasses MUST NOT define additional root fields. If a field is
   specific to the environment, it belongs on the taskset or harness config that
   owns the behavior.
4. `taskset` and `harness` MUST NOT be `None`. Omit the section to use the
   default config.
5. Nested config defaults should be explicit objects, e.g.
   `harness: MyHarnessConfig = MyHarnessConfig()`. Do not use
   `Field(default_factory=...)` for v1 config defaults.

## Ownership Rules

Taskset configs own task behavior:

- task sources, eval sources, splits, IDs, cache locations, and dataset sizing
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
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn="my_env.rewards:exact")
    ]


class MyTaskset(vf.Taskset):
    config_type = MyTasksetConfig

    def rows(self) -> list[dict[str, object]]:
        rows = [
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "answer": "4",
                "split": "train",
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]


class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig = MyTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_taskset(config: MyTasksetConfig = MyTasksetConfig()) -> vf.Taskset:
    return MyTaskset(config=config)


def load_environment(config: MyEnvConfig = MyEnvConfig()) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
```

### Custom Harness

Use this when the harness owns a reusable execution mechanism or config surface.

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyTaskset(vf.Taskset):
    config_type = MyTasksetConfig

    def rows(self) -> list[dict[str, object]]:
        rows = [
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "answer": "4",
                "split": "train",
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]


class MyHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env.programs:run")
    timeout_seconds: int = 120


class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig = MyTasksetConfig()
    harness: MyHarnessConfig = MyHarnessConfig()


def load_taskset(config: MyTasksetConfig = MyTasksetConfig()) -> vf.Taskset:
    return MyTaskset(config=config)


def load_harness(config: MyHarnessConfig = MyHarnessConfig()) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(config: MyEnvConfig = MyEnvConfig()) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

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

1. Import the public API with `import verifiers as vf`. `verifiers.v1` remains
   available for framework-internal tests and narrow module-level checks, but
   user environment code should use the top-level namespace.
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
3. Put shared extractor and factory import refs in `TasksetConfig.objects` and
   wire them with `TasksetConfig.bindings`; live objects belong only in runtime
   mutation APIs.
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
10. Do not write global helper functions in environment files. Rare exceptions
    are process-level handles, such as a lock, where a module-level object is the
    cleanest way to assert process-wide control.
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
