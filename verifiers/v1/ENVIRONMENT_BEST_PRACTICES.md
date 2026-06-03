# v1 Environment Contract

This is the strict authoring contract for v1 Taskset/Harness environments. Use
it before adding, migrating, reviewing, or agent-generating v1 environment code.
The full walkthrough is in `docs/byo-harness.md`.

## Golden Loader Shape

Environment modules expose one root loader:

```python
def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

When the environment has a custom taskset config, expose a typed child loader:

```python
def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)
```

When the environment has a custom harness config, expose a typed child loader:

```python
def load_harness(config: MyHarnessConfig) -> MyHarness:
    return MyHarness(config=config)
```

Those child loader annotations are the config contract. `load_environment`
stays typed as `vf.EnvConfig`; the framework uses the child annotations to
coerce `config.taskset` and `config.harness`.

## Hard Rules

1. Import Verifiers as `import verifiers as vf`.
2. Use `vf.Taskset`, `vf.Harness`, and `vf.Env` for new reusable environments.
3. Use `XXXConfig` Pydantic config classes for structured settings.
4. Put task behavior on the taskset config/class.
5. Put execution behavior on the harness config/class.
6. Keep `load_environment(config: vf.EnvConfig)` as-is; implement the config surface through taskset and harness configs.
7. Do not accept root loader kwargs for taskset or harness fields.
8. Do not subclass `vf.Env` for ordinary environment packages.
9. Do not subclass `vf.EnvConfig` just to narrow child config types.
10. Do not override `Taskset.__init__`, `Harness.__init__`, or `User.__init__`.
11. Do not synthesize fallback configs, accept `None`, or mutate configs inside
    loaders.
12. Do not put system messages inside `task["prompt"]`.
13. Do not pass non-serializable callables or `Path` objects through config;
    use import-ref strings.
14. Do not hide one-off behavior in private helper methods or detached helper
    functions at the bottom of an environment file.

Break these rules only when there is a concrete, documented framework-boundary
reason. Do not add escape hatches to make a local implementation easier.

## Start With Tasksets

Start with a self-contained taskset and the base harness. Tool-use tasks,
LLM-judged tasks, multimodal tasks, sandboxed tools, and simple multi-turn user
simulations should all be tasksets first.

Add a custom harness only when the environment owns a reusable execution
protocol: command agents, third-party agent frameworks, browser or desktop
loops, endpoint interception, primary sandbox placement, or program execution
that can attempt arbitrary tasks.

## Ownership Rules

Tasksets own:

- task loading and split selection;
- task prompts, answers, task metadata, and task controls;
- task-owned toolsets and users;
- task-specific setup, update, stop, cleanup, metrics, rewards, and advantages;
- task-owned objects, bindings, artifacts, files, dirs, and sandbox overrides.

Harnesses own:

- rollout execution and model/client defaults;
- programs, command agents, framework adapters, endpoint interception, and
  protocol translation;
- primary sandbox placement and reusable execution setup;
- harness-owned toolsets, objects, bindings, artifacts, metrics, and cleanup.

If a tool defines the task's action space, observations, or success condition,
it belongs to the taskset. If a class only describes how a model attempts any
task, it belongs to the harness.

## Config Rules

- Config fields should be serializable and stable enough to appear in TOML.
- Static system prompts belong in the owning config:
  `TasksetConfig.system_prompt` for task policy and
  `HarnessConfig.system_prompt` for execution policy.
- System prompt resolution is per task: `task["system_prompt"]` overrides
  `TasksetConfig.system_prompt` for the taskset side, then
  `HarnessConfig.system_prompt_strategy` resolves that side against the harness
  side.
- The default system prompt strategy is `HT`; available strategies are `HT`,
  `TH`, `H_OR_T`, `T_OR_H`, `H`, `T`, and `REJECT`.
- File-backed GEPA prompts should use `vf.SystemPromptConfig(path="...")`.
- Override `load_system_prompt(config)` only for computed prompt loading.
- Shared dependencies use `objects` and `bindings`; object entries are loader
  specs, not pre-initialized objects.
- Users are configured with `UserConfig` subclasses and implemented by
  `User.get_response(...)`, not by passing callable users.
- Tools are exposed through `vf.Toolset`; tasks only show/hide toolsets and
  tools.
- Runtime-only resources live on `state` or runtime-managed owners, not on task
  data or config.
- Do not add generic split config fields that duplicate `load_tasks(split=...)`.
  Use config only when a split choice is a real taskset setting
  rather than an adapter detail.

## Task Rules

Task records are JSON-serializable and become immutable `vf.Task` objects during
rollout. Common top-level fields are:

- `prompt`
- `system_prompt`
- `answer`
- `info`
- `max_turns`
- `toolsets`
- `tools`
- `sandbox`
- `program`
- `artifacts`

Users should not need to manage task IDs. Include upstream IDs only as ordinary
task metadata when they matter.

Prefer returning a `datasets.Dataset` directly when the source already exposes
standard task columns such as `question` and `answer`; the framework derives
`prompt` from `question`. Transform rows only to match the task contract, add
real reference fields, create multimodal content, or attach per-example state
that the rollout actually uses.

Do not copy config defaults into every task row. Use `max_turns`, `sandbox`,
`program`, and tool visibility fields in task records only when they genuinely
vary by example.

## Tools And Sandboxes

Sandboxed tools are normal tools. Put them in `vf.Toolset`, pass runtime
resources through bindings or state helpers, and keep task rows serializable.

Avoid local protocol scaffolding for ordinary sandbox handles. If an example
needs a local framework protocol class, hidden state-key convention, or custom
tool wrapper just to call a sandboxed tool, fix the public API or docs instead.

## Failure Behavior

Fail fast. Do not add fallback parsing, fallback imports, compatibility aliases,
broad best-effort branches, or silent degraded behavior to make a local
environment more permissive.

Mutable process globals are a smell. Runtime clients, sessions, sandboxes,
caches, and registries should be owned by config-bound objects, lifecycle
methods, state, or runtime-managed owners.

## Review Checklist

Before approving a v1 environment:

1. The root loader is `load_environment(config: vf.EnvConfig)`.
2. Child loaders use one concrete config type each.
3. Environment-specific fields are not on `EnvConfig`.
4. No subclass overrides final constructors.
5. No bottom-of-file helper clutter exists for single-use logic.
6. Taskset/harness ownership is clear.
7. Config values are serializable.
8. System prompt handling is config-first.
9. Task rows do not duplicate config defaults or framework-managed IDs.
10. Dataset records are returned directly when no transformation is needed.
11. Toolsets and users use first-class v1 objects.
12. The base harness is used unless a reusable protocol requires a custom one.
13. Failure paths are strict and explicit.
14. The environment has been validated through install/load/eval, not just
    imported.
