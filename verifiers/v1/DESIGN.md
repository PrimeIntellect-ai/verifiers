# Verifiers v1 Design State

This is the current design state for the v1 pass. It records decisions that are
implemented or directly reflected in the active examples, plus the open points
that still need validation.

## Current Commitments

### One Runtime Owner

`Harness` owns runtime resolution. `Taskset` contributes task rows, signals,
tools, user behavior, and task-level runtime requests. `Env` attaches the
taskset to the harness and bridges the existing `vf.Environment` worker API.

Runtime objects are internal. User-facing code works with serializable `Task`
and `State`. In-process Python programs and tools can ask for the active
`runtime` by name when they need tool handles or nested harness calls.

`verifiers.v1.State` is the public `verifiers.types.State`; v1 adds helper
constructors and serializability checks to the shared type rather than
introducing a second state class.

### Harness Run Contract

```python
state = await harness.run(task, state=None)
```

`Harness.run` completes the rollout-scope lifecycle:

1. initialize/setup state;
2. run the program through the interception endpoint;
3. collect declared artifacts;
4. sync trajectory-derived fields;
5. record timing;
6. run rollout metrics and rewards;
7. run rollout cleanup;
8. release rollout-scoped sandboxes and MCP sessions;
9. if no group boundary exists, run group cleanup and release group-scoped
   resources;
10. validate state serialization.

Handled `vf.Error` instances are serialized into `state["error"]` and still flow
through scoring and cleanup. Other exceptions raise.

Standalone harnesses can bind a client/model at construction. `Env` still owns
the eval/training boundary and writes per-invocation model controls into state.
State runtime controls take precedence for a specific rollout.

### Env Group Contract

`Env` subclasses `vf.Environment`. It accepts the current worker inputs
(`RolloutInput`, client, model, sampling args), converts inputs to `Task`
objects, writes model/client controls into state runtime metadata, dispatches
rollouts, then runs group scoring and group cleanup.

Tasksets bridge into the current worker schema by emitting v0-shaped dataset
rows. The full canonical task row is JSON-serialized into `info["task"]`;
`Taskset.to_task` and base state initialization both accept `info` as either a
mapping or JSON string and hydrate `Task` from that payload when present.

`run_group` scores by default. It is the convenience path for eval/training
workers and lightweight callers that do not need custom rollout dispatch.

`Taskset.init_group(task, num_rollouts)` is the customization point for
group-consistent prompt/task/state setup. It returns one task and one state per
rollout.

### Program Forms

The base `Harness` supports:

- default OpenAI-compatible tool loop;
- Python callable program;
- command program;
- explicit base program;
- sandbox placement for command, entrypoint, and base programs.

All model calls go through the interception endpoint so trajectory capture has
one implementation path.

### Client Protocols

The interception endpoint is a runtime protocol boundary, not a harness
subclass boundary. OpenAI Chat Completions is the default protocol. Anthropic
Messages and OpenAI Responses are selected through `ClientConfig.client_type` /
endpoint registry `type`, and the resolved protocol is stored in serializable
runtime state for the harness and interception endpoint.

### Toolsets

`Toolset` is first-class packaging for callable tools, nested toolsets, and MCP
stdio servers. It can carry:

- `tools`;
- `show` or `hide`;
- hidden `bindings`;
- lazy `objects`;
- `sandbox` requirements;
- cleanup/teardown functions.

Harness programs use `load_tools_from_state(state, runtime=runtime)`. Tool
handles do not live in state; state stores only serializable refs such as tool
names and sandbox ids.

MCP tools can be passed as `MCPTool(...)` objects in code, or as `{command=...,
args=[...]}` specs inside toolset config.

Toolset sandboxes support `scope="rollout"`, `"group"`, and `"global"`.
Rollout-scoped sandboxes are released after rollout scoring and cleanup.
Group-scoped sandboxes survive until group scoring and cleanup. Global sandboxes
live for the harness runtime and are released at teardown.

Primary program sandboxes use the same scoped lease mechanism. A sandboxed
program with `scope="group"` reuses one primary sandbox for compatible rollouts
in the same group; `scope="global"` keeps the primary sandbox until harness
teardown. Toolsets can declare `sandbox="program"` to receive the active primary
program sandbox handle as a hidden tool argument.

MCP server sessions follow the same scope vocabulary as their owning toolset.

Toolset lazy objects use the same scope vocabulary. Read-only toolsets default
to global objects. `write=True` toolsets default to rollout-scoped objects, so
mutable task simulators and databases can stay out of `State` while remaining
isolated per rollout.

Programs can discover and call resolved tools through the interception endpoint:
`GET /vf/tools` returns provider-agnostic tool schemas.
`GET /vf/tools?protocol=...` renders the same tools for OpenAI Chat
Completions, OpenAI Responses, or Anthropic Messages. `POST /vf/tools/{name}`
calls a tool with hidden bindings resolved by the host runtime. Command/sandbox
programs also receive `VF_TOOLS_JSON`, `VF_TOOL_DEFS_JSON`, `VF_TOOL_BASE_URL`,
and endpoint auth env vars.

MCP tools are normalized into callable handles for the runtime. Callable tools
can also be exposed to external programs as stdio MCP tools when a harness uses
`tool_protocol="mcp"`. The generated MCP server is official MCP plumbing around
the runtime's resolved tool surface, not a separate user-facing tool type.

### Users

Tasksets and harnesses may define at most one user. A direct callable is wrapped
as `User(fn=...)`. `User` supports `scope`, `bindings`, and lazy `objects`,
mirroring tool object lifetimes. Users may also request scoped sandboxes, using
the same lifecycle rules as toolset sandboxes.

`transcript` is a default binding. It resolves to the current prompt plus
completion when available, falling back to trajectory-derived completion.

### Signals

Metrics, rewards, and advantages are signal functions.

Rollout signals:

```python
@vf.metric
async def metric_name(task, state) -> float: ...


@vf.reward(weight=1.0)
async def reward_name(task, state) -> float: ...
```

Group signals:

```python
@vf.reward(stage="group")
async def group_reward_name(tasks, states) -> list[float]: ...


@vf.advantage
async def advantage_name(tasks, states) -> list[float]: ...
```

Function names are signal names. Name collisions hard fail. Metrics, rewards,
and advantages are added through taskset/harness config fields or constructor
args; `scoring` config only skips or overrides metadata for existing names.
Absent an explicit advantage signal, group scoring writes
`reward - mean(group_reward)`.

`Rubric` is not part of the v1 authoring path. Compatibility adapters can be
added around the v1 signal runner where needed.

### Stop Conditions

Stop handlers can be contributed by tasksets, harnesses, or toolsets via
decorators or constructor/config lists. The built-in state stop condition treats
`state["done"]` / `state["is_completed"]` as a generic finish-tool signal and
sets `stop_condition` from the handler name.

### Cleanup And Teardown

`@vf.cleanup` is the user extension point for in-place state changes after
program execution and scoring. It can target rollout or group stage.

`@vf.teardown` is process/runtime cleanup for long-lived services. There is no
public `render` decorator; framework-owned rendering is part of harness state
sync.

### Nested Harness Runs

`runtime.run_harness(child, task, parent_state=state)` starts a child rollout
from inside a program/tool. The child receives a fresh `trajectory_id` and
rollout-scoped resources, while inheriting the parent `group_key`, model,
sampling args, and model client unless the child harness or child state overrides
them. The parent records each child call in `state["child_rollouts"]`; child
metrics remain namespaced inside the child state rather than being merged into
parent metrics.

## Current Examples

- `reverse_text_v1`: single-turn default harness plus rollout reward.
- `alphabet_sort_v1`: multi-turn default harness plus taskset user callable.
- `wiki_search_v1`: callable toolset with lazy Chroma object bindings.
- `math_python_v1`: sandboxed callable Python tool with rollout cleanup.
- `mcp_search_v1`: stdio MCP toolset consumed by the default harness.
- `opencode_harbor_v1`: sandboxed command harness plus rollout reward that runs
  Harbor tests before sandbox cleanup.
- `nested_harness_v1`: tool binding that calls another harness.
- `hello_subagent_v1`: deterministic parent harness that must call child
  harnesses through a tool to solve the task.
- `hello_rlm_v1`: sandboxed RLM CLI command harness with checkout upload,
  install, endpoint interception, metrics collection, and trajectory filtering.
- `dspy_flights`: importable DSPy entrypoint configured against the v1
  interception endpoint.
- `tau2_bench`: taskset-owned tools and user simulator backed by a
  rollout-scoped session object.

## Open Questions

- Which config surfaces should become nested subconfigs once program, sandbox,
  tool, and user settings grow more options.
- Whether a generic channel registry is warranted. The current v1 shape has no
  public channel registry: stable cross-cutting concepts are promoted config
  fields, while local/custom concepts live on `TasksetConfig` or
  `HarnessConfig` subclasses.
- How to represent model and policy separately when a harness uses auxiliary
  model calls that should not train the primary policy.
- How broad the protocol adapter surface should become beyond OpenAI Chat
  Completions, Anthropic Messages, and OpenAI Responses.
- How group setup should work for prompt randomization, dynamic tasks, and
  stateful tasksets.
- How far runtime context inheritance should go for deeply nested harness calls
  beyond model controls and group membership.
- Whether cleanup names need sharper semantics as more resources participate in
  rollout and group stages.
