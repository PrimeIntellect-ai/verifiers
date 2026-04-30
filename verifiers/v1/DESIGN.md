# Verifiers v1 Design Notes

This is a design workspace, not a locked spec. The purpose of this document is
to record the few choices we are confident about, keep candidate API sketches
small, and leave a large open surface for implementation feedback.

The first v1 pass should validate top-level APIs before hardening internals.
When a concept has not clicked into a concrete implementation point, do not name
it yet.

## Goals

- Keep the public surface small and composable.
- Make Python authoring and TOML iteration both pleasant.
- Preserve the existing trainer/env boundary unless implementation proves it is
  insufficient.
- Keep task and state serializable across rollout and scoring boundaries.
- Avoid rebuilding the old env hierarchy with new names.

## Settled

### Target Authoring Signatures

The v1 design should start from the signatures we want authors to use.
`Environment` compatibility matters, but it should not determine every internal
type shape.

Env package entrypoints:

```python
def load_taskset(config=None) -> vf.Taskset: ...
def load_harness(config=None) -> vf.Harness: ...
def load_environment(taskset_config=None, harness_config=None) -> vf.Environment: ...
```

Rollout-scoped decorated functions:

```python
async def reward(task: vf.Task, state: vf.State) -> float: ...
async def metric(task: vf.Task, state: vf.State) -> float: ...
```

Group-scoped functions are likely plural, but the exact declaration mechanism is
not settled:

```python
async def group_reward(tasks: list[vf.Task], states: list[vf.State]) -> float: ...
```

Taskset and harness constructors should be simple. Whether `config` is the only
argument is still open, but users should not have to manually merge config
overlays with Python defaults.

```python
taskset = MathTaskset(config=config)
harness = BasicHarness(config=config)
```

### `vf.Env` Adapts To `vf.Environment`

`vf.Env` must subclass `vf.Environment`. It is the taskset/harness
implementation of the environment abstraction, not a totally separate
trainer-facing boundary.

The existing clients and env workers currently send these values:

```python
async def run_group(
    group_inputs: list[RolloutInput],
    client: Client | ClientConfig,
    model: str,
    sampling_args: SamplingArgs,
    ...
) -> list[RolloutOutput]: ...
```

Those inputs describe the current adapter boundary, not the canonical v1 API.
The task/group shape is open. It may stay as grouped rollout inputs, become a
task plus rollout count, or become another task-oriented request shape:

```python
async def run_group(
    task: vf.Task,
    num_rollouts: int,
    ...
) -> list[vf.State]: ...
```

The internals are allowed to change. If `Environment` is carrying legacy
MultiTurnEnv-specific behavior that blocks a clean v1 shape, that behavior can
move down into `MultiTurnEnv`.

The core v1 shape may become closer to:

```python
async def rollout(...) -> State: ...
async def score_group(states: list[State], ...) -> list[State]: ...
```

or a cleaner `run_and_score_group` pipeline. The important constraint is that
trainer-provided runtime controls such as client/model/sampling can be bridged
into the v1 flow without making task rows carry runtime control data.

### Decorator-In-Class Pattern

Decorated functions defined on a `Taskset` class belong to that taskset.
Decorated functions defined on a `Harness` class belong to that harness.

```python
class MathTaskset(vf.Taskset):
    @vf.reward(weight=1.0)
    async def exact_answer(task: vf.Task, state: vf.State) -> float:
        return float(state.get("answer") == task["answer"])


class BasicHarness(vf.Harness):
    @vf.metric
    async def num_turns(task: vf.Task, state: vf.State) -> float:
        return float(len(state.get("trajectory", [])))
```

Decorator names are not configurable. The registry key is always the function
name.

```toml
[env.taskset.scoring.exact_answer]
weight = 0.5

[env.harness.scoring.num_turns]
skip = true
```

Plain reusable functions may still use decorators for metadata, but ownership is
decided by where they are attached:

```toml
[env.taskset.scoring.format_penalty]
ref = "my_project.rewards:format_penalty"
weight = -0.1
```

The TOML path decides that `format_penalty` belongs to the taskset.

### Existing Objects Can Be Extended Without Subclassing

Subclassing is not the only way to add a metric, reward, tool, or requirement to
an existing taskset or harness. The normal overlay path should also work through
constructor arguments, config, or a small programmatic method.

Programmatic extension should look roughly like this:

```python
@vf.metric
async def num_tool_calls(task: vf.Task, state: vf.State) -> float:
    return float(len(state.get("tool_calls", [])))


def load_taskset(config=None) -> vf.Taskset:
    taskset = ExistingTaskset(config=config)
    taskset.add_metric(num_tool_calls)
    return taskset
```

Constructor extension should also be possible when it reads more clearly:

```python
def load_taskset(config=None) -> vf.Taskset:
    return ExistingTaskset(
        config=config,
        metrics=[num_tool_calls],
    )
```

The exact method names are not settled. The constraint is that extending an
existing taskset should not require creating another subclass just to attach one
function.

Nested config comes in at ownership boundaries. The env loader dispatches config
to the taskset and harness; each owner merges config with its own defaults.

```toml
[env.taskset.scoring.num_tool_calls]
ref = "my_project.metrics:num_tool_calls"
skip = false
priority = 20
```

The merge order should be simple and framework-owned:

1. collect decorated class defaults;
2. add constructor/programmatic entries;
3. apply TOML/config overrides by function name;
4. validate and produce a sorted signal list.

Users should not manually merge default Python definitions with TOML overrides.
The exact nesting below `scoring` is still open; the stable idea is owner path,
function name, metadata.

### Subclasses Are Definition Surfaces

Taskset and harness subclasses are allowed as one-level definition surfaces.
They should package defaults, decorated functions, tools, and requirements.
They should not create a new runtime hierarchy.

This is allowed:

```python
class MyTaskset(vf.Taskset):
    # Task loading shape intentionally omitted here.

    @vf.reward(weight=1.0)
    async def unit_tests(task: vf.Task, state: vf.State) -> float:
        ...


class OpenCode(vf.Harness):
    # CLI/sandbox declaration shape intentionally omitted here.

    @vf.metric
    async def patch_size(task: vf.Task, state: vf.State) -> float:
        ...
```

This should not be the normal extension mechanism:

```python
class MyHarness(vf.Harness):
    async def run(...):
        ...
```

The shared runner machinery should stay in the base implementation.
Subclasses describe what to run, not a new way to run it.

### Harnesses Do Not Hardcode Models

Harness packages should not hardcode concrete model IDs. Running model X in
harness Y must be possible without editing harness code.

The current client/env-worker boundary already carries `client`, `model`, and
`sampling_args`. If this proves insufficient for per-group model selection, we
should extend that boundary deliberately after validating the specific gap.

## Minimal Env Sketch

This sketch is intended to validate the authoring shape, not every internal
type name.

```python
import verifiers as vf


class MathTaskset(vf.Taskset):
    # Task data loading is intentionally not specified in this sketch.

    @vf.reward(weight=1.0)
    async def exact_answer(task: vf.Task, state: vf.State) -> float:
        return float(state.get("answer") == task["answer"])


class BasicHarness(vf.Harness):
    @vf.metric
    async def num_turns(task: vf.Task, state: vf.State) -> float:
        return float(len(state.get("trajectory", [])))


def load_taskset(config=None) -> vf.Taskset:
    return MathTaskset(config=config)


def load_harness(config=None) -> vf.Harness:
    return BasicHarness(config=config)


def load_environment(taskset_config=None, harness_config=None) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=load_harness(harness_config),
    )
```

TOML override:

```toml
[env.taskset.scoring.exact_answer]
weight = 0.5

[env.taskset.scoring.format_penalty]
ref = "my_project.rewards:format_penalty"
weight = -0.1
priority = 20

[env.harness.scoring.num_turns]
skip = false
```

Resolution should be framework-owned:

1. Discover decorated functions on the taskset and harness classes.
2. Key each function by `fn.__name__`.
3. Apply TOML entries under the matching owner path.
4. Import new functions that provide `ref`.
5. Drop entries with `skip = true`.
6. Validate signatures and metadata.
7. Produce a sorted signal list that simple scoring functions can execute.

Users should not manually merge defaults with TOML overlays.

## Open Surface

### Taskset Internals

Not settled:

- how task data is attached to a taskset;
- how lazy dataset loading is expressed;
- how arbitrary task rows become immutable `Task` objects;
- how group-level setup works;
- how dynamic tasksets and task DAGs should look;
- which top-level class fields are allowed.

### Harness Internals

Not settled:

- exact runner interface under `Harness`;
- whether harness execution is always endpoint-backed in the first slice;
- how CLI, sandboxed CLI, and in-process execution share one runner;
- how custom execute logic is expressed without allowing arbitrary `run`
  overrides;
- how multiple model uses are declared;
- which top-level class fields are allowed.

### Config And TOML

Not settled:

- exact Pydantic model names;
- how config overlays are passed into `load_taskset` and `load_harness`;
- whether config should be accepted directly by constructors or resolved before
  construction;
- selector semantics such as "only these scoring functions";
- whether a custom config escape hatch exists at all.

High-confidence rule: TOML should override named local defaults without forcing
users to write merge logic.

### Scoring

Not settled:

- whether the public words are `metric`, `reward`, `signal`, or something else;
- whether metrics and rewards are separate buckets or one signal model with
  weights;
- how group-stage functions are declared;
- where advantage computation fits;
- whether any compatibility path from existing `Rubric` is core or external.

High-confidence rule: v1 should not require users to think in terms of the old
`Rubric` object when authoring new taskset/harness envs.

### Post-Run Hooks

Not settled:

- whether `render` is public;
- whether custom `stop` is public;
- exact names and stages for cleanup;
- how teardown is registered for atexit-style resources;
- how cleanup interacts with scoring when resources must survive until group
  scoring.

### Tools And Toolsets

Not settled:

- exact toolset config shape;
- callable tool versus MCP tool representation;
- hidden binding syntax;
- callable-to-MCP adaptation rules;
- sandbox sharing rules for tools;
- whether toolsets get their own package/library later.

### Model And Runtime Selection

Not settled:

- whether the current `client, model, sampling_args` boundary is enough for
  per-group model decisions;
- where auxiliary model configuration belongs;
- how to distinguish model endpoint from optimization policy;
- how much model routing should be visible in public config;
- whether trainer/env worker request envelopes need extension.

High-confidence rule: do not hardcode concrete model IDs in harness packages.

## Next Validation Work

Implement only enough to validate the API surface:

1. Minimal `vf.Env(vf.Environment)` that composes a taskset and harness.
2. Decorator discovery on taskset/harness classes.
3. TOML override of a default reward by function name.
4. TOML addition of a new reward by `ref`.
5. A basic harness using the shared runner.
6. One opinionated harness such as OpenCode without overriding the core runner.

Anything not needed for those six checks should stay unnamed or internal.
