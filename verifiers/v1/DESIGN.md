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
and `State`, plus tool handles loaded from state.

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

### Program Forms

The base `Harness` supports:

- default OpenAI-compatible tool loop;
- Python callable program;
- command program, with optional primary sandbox.

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

Harness programs use `load_tools_from_state(state)`. Tool handles do not live in
state; state stores only serializable refs such as tool names and sandbox ids.

MCP tools can be passed as `MCPTool(...)` objects in code, or as `{command=...,
args=[...]}` specs inside toolset config.

Toolset sandboxes support `scope="rollout"`, `"group"`, and `"global"`.
Rollout-scoped sandboxes are released after rollout scoring and cleanup.
Group-scoped sandboxes survive until group scoring and cleanup. Global sandboxes
live for the harness runtime and are released at teardown.

MCP server sessions follow the same scope vocabulary as their owning toolset.

Toolset lazy objects use the same scope vocabulary. Read-only toolsets default
to global objects. `write=True` toolsets default to rollout-scoped objects, so
mutable task simulators and databases can stay out of `State` while remaining
isolated per rollout.

### Users

Tasksets and harnesses may define at most one user. A direct callable is wrapped
as `User(fn=...)`. `User` supports `scope`, `bindings`, and lazy `objects`,
mirroring tool object lifetimes. Users may also request scoped sandboxes, using
the same lifecycle rules as toolset sandboxes.

`transcript` is a default binding. It resolves to the current prompt plus
completion when available, falling back to trajectory-derived completion.

### Signals

Metrics and rewards are signal functions.

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
```

Function names are signal names. Name collisions hard fail. Metrics and rewards
are added through taskset/harness config fields or constructor args; `scoring`
config only skips or overrides metadata for existing names.

`Rubric` is not part of the v1 authoring path. Compatibility adapters can be
added around the v1 signal runner where needed.

### Cleanup And Teardown

`@vf.cleanup` is the user extension point for in-place state changes after
program execution and scoring. It can target rollout or group stage.

`@vf.teardown` is process/runtime cleanup for long-lived services. There is no
public `render` decorator; framework-owned rendering is part of harness state
sync.

## Current Examples

- `reverse_text_v1`: single-turn default harness plus rollout reward.
- `alphabet_sort_v1`: multi-turn default harness plus taskset user callable.
- `wiki_search_v1`: callable toolset with lazy Chroma object bindings.
- `math_python_v1`: sandboxed callable Python tool with rollout cleanup.
- `mcp_search_v1`: stdio MCP toolset consumed by the default harness.
- `opencode_harbor_v1`: sandboxed command harness plus rollout reward that runs
  Harbor tests before sandbox cleanup.
- `nested_harness_v1`: tool binding that calls another harness.
- `hello_rlm_v1`: command-program harness sketch.
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
- Whether stop conditions should become public v1 decorators, or remain an
  implementation detail of framework-owned loops.
- How much callable-to-MCP adaptation should be automatic for sandboxed
  programs and remote workers.
- Whether cleanup names need sharper semantics as more resources participate in
  rollout and group stages.
