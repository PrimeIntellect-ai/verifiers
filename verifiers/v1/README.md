# Verifiers v1 Workspace

`verifiers.v1` is the workspace for the next environment architecture. The
goal is to make the stable v1 surface feel like functions plus typed configs,
while preserving the execution scale needed for sandboxes, endpoint forwarding,
tool servers, group scoring, and hosted eval/training workers.

This package is intentionally separate from `verifiers.envs.experimental`.
Code in this folder should converge toward the release-facing API rather than
carry every compatibility concern from the current environment stack.

## Design Thesis

The core user-facing objects are:

- `Task`: immutable, serializable input row.
- `State`: mutable, serializable rollout record.
- `TasksetConfig`: how tasks are loaded, shaped, and scored.
- `HarnessConfig`: how a runner executes a task into a state.
- `Env`: the resolved product of a taskset and harness.

The core user-facing extension mechanism is a small set of functions with
strict signatures:

```python
async def init_state(task: vf.Task, state: vf.State) -> vf.State: ...
async def execute(task: vf.Task, state: vf.State) -> vf.State: ...
async def metric(task: vf.Task, state: vf.State) -> float: ...
async def reward(task: vf.Task, state: vf.State) -> float: ...
async def cleanup(task: vf.Task, state: vf.State) -> vf.State: ...
async def teardown() -> None: ...
```

`stop` and `render` need sharper boundaries before they become public v1
extension points. `stop` is natural for framework-owned loops, but it is less
obvious when the rollout loop lives entirely inside `execute`. `render` may be a
framework-owned state normalization pass rather than an API users need to touch
directly.

Group-stage functions opt in explicitly and use plural arguments:

```python
async def group_metric(tasks: list[vf.Task], states: list[vf.State]) -> float: ...
async def group_reward(tasks: list[vf.Task], states: list[vf.State]) -> float: ...
async def group_cleanup(tasks: list[vf.Task], states: list[vf.State]) -> list[vf.State]: ...
```

## Runtime Boundary

User code should not depend on a live in-process `resources` object. Runtime
objects are internal implementation details used to instantiate clients,
sandboxes, MCP servers, endpoint proxies, caches, and other service handles.

The public boundary should be serializable config plus serializable
`Task`/`State`. Runtime resolution can still create rich internal objects, but
those objects should be reachable only through framework-managed runners and
tool handles.

## Resolution Model

The v1 resolution path should be:

```text
TasksetConfig
  + HarnessConfig
  + signal/function configs
        |
        v
strict, sorted function lists
```

The first implementation slice should prefer simple functions over consolidated
manager objects. Once the function boundaries are clear, repeated patterns can
be collapsed behind internal helpers.

## Lifecycle

The intended v1 lifecycle is:

```text
init_group
  -> init_state
  -> harness.run / runner.execute
  -> rollout state rendering
  -> rollout metrics and rewards
  -> rollout cleanup
  -> group state rendering
  -> group metrics and rewards
  -> advantage
  -> group cleanup
  -> teardown
```

Rollout-stage functions receive one `task` and one `state`. Group-stage
functions receive `tasks` and `states`. Group-stage functions should require an
explicit `stage="group"` declaration so a single-rollout function cannot
accidentally become group-aware.

Teardown is process-level cleanup for framework-managed services such as atexit
handlers, local servers, endpoint proxies, and long-lived sandbox resources. It
is distinct from rollout or group cleanup, which prepares serializable states
and releases resources tied to a completed rollout or group.

## Harness And Runner Boundary

`Harness` is endpoint-backed by definition. Model calls go through the standard
endpoint/interception boundary; v1 should not maintain a separate no-intercept
generation path.

The base `Harness` owns the shared endpoint loop, tool exposure, scoring, and
lifecycle machinery. Opinionated subclasses such as `OpenCode(Harness)` are
fully instantiable harness packages that add defaults and config fields, but
they should not introduce a separate lifecycle layer.

The placement/runtime config for Python, CLI, sandboxed CLI, and remote workers
is still open. The first v1 pass should not force a generic config object before
the examples earn it.

The confident piece is that all of these forms should attach to the same
endpoint/interception boundary.

## Toolsets

Toolsets are first-class packaging for tools. A toolset can carry:

- callable tools,
- MCP tools,
- hidden bindings,
- setup requirements,
- rollout or group lifecycle handlers,
- runtime transport preferences.

Harnesses declare what tool transports they can consume. Tool resolution should
choose the best compatible representation, including adapting callable tools
behind framework-managed MCP/server handles when needed.

## Signals

Metrics and rewards are signals. Metrics are always single-rollout unless
explicitly promoted to a group stage. Rewards may be rollout-stage or
group-stage. The metric/reward split is still under design: the key invariant is
that signal execution has explicit stage semantics, not that every signal must
be exposed through separate public concepts. `Rubric` should not be part of the
v1 public API; compatibility adapters may translate existing rubrics into signal
configs outside the core v1 surface.

Harnesses can contribute rollout metrics about their own execution. Tasksets
own correctness rewards. Both may contribute rollout cleanup. Group rewards,
group metrics, advantages, and group cleanup are resolved at the
environment/group stage.

## Config Requirements

Configs should be expressible as Python objects and TOML. Python configs may
accept live callables. TOML configs should reference importable symbols by name.

Config objects should be Pydantic-compatible where useful, with strict field
names and nested subconfigs for complex areas such as:

- models and policies,
- endpoints,
- sandboxes,
- tools and toolsets,
- scoring signals,
- cleanup/render lifecycle,
- logging and artifacts.

## Model And Policy Split

Harnesses should not hardcode concrete model IDs. Concrete model configuration
comes from the trainer through a per-group invocation/control envelope. This
keeps the path to "run model X in harness Y" independent of harness code.

Models describe endpoints and generation protocols. Policies describe which
model calls are part of optimization and scoring. The exact v1 shape for
multiple model uses inside one harness is still open.

The trainer-env boundary has a control plane in addition to task delivery.
Tasks and states remain the env-author contract; runtime controls carry
per-group model configuration, seeds, budgets, placement, logging, and other
trainer-owned decisions.

## Open Design Questions

- What is the smallest strict `Task`/`State` schema that still allows arbitrary
  dataset rows and flexible transcripts?
- Should `execute` be a decorator, a config field, or a named method on a
  runner implementation?
- Does v1 expose `stop`, or is stopping only a concern for built-in
  framework-owned loops?
- Does v1 expose `render`, or should rendering remain a fixed state
  normalization step around scoring?
- Should metrics and rewards be distinct public concepts, or one signal concept
  with reward weights and stage semantics?
- How much dependency metadata must a portable executor declare to run inside a
  sandbox?
- What is the exact TOML shape for referencing callables, toolsets, and signal
  functions?
- Where should group setup state live when prompt randomization or dynamic
  tasksets need stable per-group choices?
- How should dynamic tasksets expose task DAGs, generated tasks, and cross-worker
  coordination without making every eval stateful?
- How should sandbox retention be configured across rollout scoring, group
  scoring, and cleanup?

## First Implementation Slice

The first v1 slice should prove the strict boundary:

1. Define minimal `Task` and `State` types with serialization enforcement.
2. Define Pydantic config shells for tasksets, harnesses, toolsets, models, and
   lifecycle signals.
3. Implement one endpoint-backed harness runner whose public functions only see
   `task` and `state`.
4. Implement one callable toolset with hidden bindings resolved through config.
5. Implement rollout metrics, rollout cleanup, and one group reward.
6. Port one simple environment and one sandbox/tool environment into `v1`.
