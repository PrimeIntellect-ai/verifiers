# Taskset + Harness Refactor

This document describes the taskset/harness architecture for composing reusable
task collections with reusable rollout harnesses. It records the API shape,
design choices, conceptual anchors, migration path, and remaining open
questions for:

```python
vf.Env(taskset=..., harness=...)
```

Use `vf.Env` for reusable taskset/harness environments. `MultiTurnEnv`,
`ToolEnv`, `StatefulToolEnv`, `SandboxEnv`, `CliAgentEnv`, `HarborEnv`, and
`ComposableEnv` are supported environment stacks and migration sources.

## Goals

The refactor separates environment responsibilities into three objects:

- `Task`: immutable, serializable task row.
- `State`: mutable, serializable rollout record.
- `Resources`: non-serializable resolved runtime object bag.

The core composition is:

```text
Taskset + Harness = Env
```

`Taskset` owns what is being solved. `Harness` owns how rollout interaction
works. `Resources` owns the resolved product of taskset/harness compatibility:
tools, rubrics, sandbox runtime, endpoint server, lifecycle hooks, and custom
objects from custom channels.

The intent is to make tasksets and harnesses independently reusable:

```python
import verifiers as vf
from verifiers.envs.experimental.modules.harnesses import OpenCode
from verifiers.envs.experimental.modules.tasksets import HarborTaskset


def load_taskset(taskset_args=None) -> vf.Taskset:
    return HarborTaskset(**(taskset_args or {}))


def load_harness(harness_args=None) -> vf.Harness:
    return OpenCode(**(harness_args or {}))


def load_environment(taskset_args=None, harness_args=None) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(taskset_args),
        harness=load_harness(harness_args),
    )
```

## Package Boundaries

The implementation lives under `verifiers/envs/experimental`.

Core framework code:

- `verifiers/envs/experimental/env.py`
  - `Env`
- `verifiers/envs/experimental/task.py`
  - `Task`
- `verifiers/envs/experimental/taskset.py`
  - `Taskset`
- `verifiers/envs/experimental/harness.py`
  - `Harness`
- `verifiers/envs/experimental/resources.py`
  - `Resources`
- `verifiers/envs/experimental/channels/`
  - `Channel`
  - `ChannelContext`
  - `ResourcePatch`
  - `LifecycleHooks`
  - default `*_channel.py` files
  - `DEFAULT_CHANNELS`

Package-shaped code for eventual top-level `tasksets` and `harnesses`
packages:

- `verifiers/envs/experimental/modules/tasksets/*`
  - `HarborTaskset`
  - `HarborRubric`
  - `DatasetTaskset`
- `verifiers/envs/experimental/modules/harnesses/*`
  - `EndpointHarness`
  - `CliHarness`
  - `OpenCode`
  - `RLMHarness`

The channels submodule is core `verifiers` framework code. Tasksets and
harnesses can declare channels, but channel resolution and resource
materialization are framework responsibilities.

## Public API Surface

The core symbols are exported through `verifiers`:

- `vf.Env`
- `vf.Task`
- `vf.Taskset`
- `vf.Harness`
- `vf.Resources`
- `vf.Channel`
- `vf.ChannelConfig`
- `vf.ChannelContext`
- `vf.ChannelMap`
- `vf.CallableTool`
- `vf.ToolInjector`
- `vf.ToolHandle`
- `vf.ToolRegistry`
- `vf.MCPServerSpec`
- `vf.SandboxSpec`
- `vf.SandboxSeed`
- `vf.SandboxTimeouts`
- `vf.User`

Concrete tasksets and harnesses live under experimental module paths. They are
not top-level `vf.*` exports.

## Dataset And Taskset API

`Taskset` uses `source` and `eval_source` as the dataset-shaped interface:

```python
taskset = vf.Taskset(
    source=lambda: load_dataset("my/dataset", split="train"),
    eval_source=lambda: load_dataset("my/dataset", split="test"),
    rubric=my_rubric,
    tools=[search, read],
)
```

`source` and `eval_source` can be:

- callables returning a dataset-like object
- already materialized dataset-like objects
- iterable row mappings
- `None`

`Taskset.get_dataset()` and `Taskset.get_eval_dataset()` cache internally after
first load. Lazy loading is explicit and direct through `source` /
`eval_source`. Subclasses use the same path by passing a bound loader:

```python
class HarborTaskset(vf.Taskset):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__(source=self._discover, rubric=HarborRubric())
```

Dataset rows are converted to `Task` through `Taskset.to_task(...)`. The task
stores:

- `prompt`
- `answer`
- `info`
- `inputs`
- `channels`
- `example_id`

In this API, `task` means the immutable `vf.Task` object. Environment routing
metadata belongs under `task.info`.

## Env

`Env` is parallel to `MultiTurnEnv`, not a subclass of it. It maps taskset and
harness pieces into the existing `Environment` runtime:

```python
class Env(Environment):
    def __init__(self, taskset: Taskset, harness: Harness):
        self.resources = Resources(taskset, harness)
        super().__init__(
            dataset=self.resources.dataset,
            eval_dataset=self.resources.eval_dataset,
            parser=self.resources.rubric.parser,
            rubric=self.resources.rubric,
            env_id=self.resources.env_id,
        )
```

The low-level `Environment` run APIs are the integration surface for
trainers/evals. New taskset/harness functionality uses the harness request path
instead of building around `Environment.generate`.

`Env.rollout(...)` converts the input row to a `Task`, activates
`Resources.rollout(...)`, and delegates to:

```python
await harness.run(task, resources)
```

`Env` accepts only `taskset` and `harness`. Channel behavior lives in tasksets,
harnesses, and channel definitions.

## Resources

`Resources` is a structured but extensible runtime object bag. It owns:

- environment-scope resolved objects
- rollout-local resolved objects
- active model client/model/sampling args through context-local state
- rollout runtime scratch data for non-serializable internals
- lifecycle hooks
- taskset and harness references

Runtime code reads resolved objects from `Resources` by name. Common framework
objects have convenience properties; channel-specific objects keep the names
defined by their channel:

```python
resources.tools
resources.rubric
resources.client
resources.model
resources.sampling_args
resources.runtime
resources.sandbox_runtime
resources.sandbox_request
```

`resources.get(...)` and `resources.require(...)` are lower-level helpers for
generic code whose resource name is dynamic. The channel definition is the
source of truth for which object names exist and what type each one resolves
to.

The fundamental boundary is:

```text
state = serializable rollout record
task = serializable immutable input
resources = non-serializable runtime object bag
```

Harness logic should read immutable row data from `task` and write rollout
progress to `state`. `state["input"]` is the serialized task snapshot used for
outputs, scoring, and trainer integration; it is not the primary source of
taskset metadata during rollout control flow.

`Env` states should not contain live clients, sandbox clients, tool servers, or
other heavy objects. State stores serializable IDs and outputs, such as
`sandbox_id`, stdout/stderr, metrics, errors, and trajectory steps.

Error state for taskset/harness envs is serialized as a compact mapping instead
of exception objects.

## Canonical Task Direction

The target shape is that `Task` is the canonical owner of all row-level input
data throughout rollout execution:

- `task.prompt`
- `task.answer`
- `task.info`
- `task.inputs`
- `task.channels`
- `task.example_id`

`State` then contains only rollout progress and observations:

- trajectory steps
- active context or harness-specific progress fields
- tool/user/sandbox outputs
- metrics, reward, error, usage, and timing

In that shape, state does not copy or forward `prompt`, `answer`, `info`, or
`example_id`. If runtime code needs row metadata, it reads from `task`. If it
needs to record something learned during rollout, it writes a distinct
state-owned field. A harness that rewrites or compacts context should store
that as rollout context, not mutate task input fields.

The compatibility surface can be expressed as views rather than stored copies:

- rubric scoring builds `prompt`, `answer`, `info`, and `task` score objects
  from the immutable task
- output serialization combines `task.to_input()` with the final state
- display code renders task input plus state trajectory
- trainer-facing datasets receive the same serialized columns even though
  state is not the owner of those columns

Rubrics are the main place where unpacking remains useful. Reward functions can
accept `task` directly when they need structured row data, or accept
`prompt`/`answer`/`info` when that is clearer. Those names should be task views,
not independent state values.

Staged rollout:

1. Establish the convention in taskset/harness envs: hook signatures receive
   `task`, `state`, and `resources`; control flow reads row metadata from
   `task`; state records rollout progress.
2. Make rubric scoring task-aware at the environment boundary. `Env` can pass
   the task object or a task-derived score object into rubric calls, while
   preserving existing reward function argument names.
3. Centralize output construction so serialized rollout outputs are produced
   from `(task, state)`. This keeps `prompt`, `answer`, `info`, and
   `example_id` available to saved results without storing them as mutable state
   fields.
4. Move harness prompt assembly off `state["prompt"]`. The first prompt can be
   derived from `task.prompt` plus resolved resources; any transformed context
   belongs in an explicitly named state field.
5. Remove implicit state forwarding for task input fields after trainers,
   displays, and scoring use task-derived views at their boundaries.
6. Treat task-level channel overrides as immutable task data. Per-rollout
   channel resolution may read `task.channels`, but should not mutate the task.

## Channels

Channels are the compatibility and resource-resolution abstraction. They are
inspired by declarative resource tooling: channel config is spec, resources are
resolved runtime/status, and lifecycle hooks act like finalizers.

The default channels are:

- `system_prompt`
- `tools`
- `rubric`
- `skills`
- `sandbox`
- `endpoint`
- `user`
- `stop`
- `cleanup`
- `teardown`

Each channel owns its own shorthand/defaulting/validation/merge behavior. There
is no global fallback merge strategy.

The default channel type:

```python
Channel(
    name="tools",
    outputs={"tools": ToolRegistry},
    extends="rubric",
    always_resolve=True,
    resolve_fn=resolve_tools,
)
```

`extends` is the channel relationship implemented here. It has two meanings:

- order the extending channel before the target channel
- permit the extending channel to submit follow-on config to that target

The field accepts either:

```python
extends="rubric"
extends=("rubric", "cleanup")
```

A target channel accepts after-resolution extension only when it provides
`extend_fn`. Rubric provides one; scalar channels such as `system_prompt` do
not. Normal contributions to `system_prompt` are accepted, but more than one
contribution hard-fails.

`ResourcePatch` is the output of channel resolution:

```python
ResourcePatch(
    objects={"tools": registry},
    hooks=LifecycleHooks(teardown=(registry.teardown,)),
    contributions={"rubric": (ToolMonitorRubric(...),)},
)
```

This lets channels submit to other channels without those target channels
knowing about the source. For example, `tools` extends `rubric` by contributing
`ToolMonitorRubric`; `rubric` does not inspect `tools`. `skills` extends
`sandbox` by contributing a named upload; `sandbox` does not inspect `skills`.

Unknown channel names hard-fail unless a taskset or harness registers a custom
`Channel` through `channel_definitions()`.

## Default Channel Semantics

`system_prompt`

- Accepts zero or one string contribution.
- Rejects multiple normal contributions.
- Has no `extend_fn`.

`tools`

- Accepts Python callables, `CallableTool`, schema-only `Tool` definitions,
  `ToolRegistry`, `ToolInjector`, and `MCPServerSpec`.
- Resolves to `ToolRegistry`.
- Registers registry teardown through lifecycle hooks.
- Extends `rubric` with automatic tool-monitor metrics when tools exist.

`rubric`

- Resolves to existing `Rubric` / `RubricGroup`.
- Uses `NoOpRubric` when nothing is contributed.
- Canonicalizes shorthand into:
  - `rewards`: reward functions, default weight `1.0`
  - `metrics`: metric functions, default weight `0.0`
  - `rubrics`: existing `Rubric` objects/classes
- Provides `extend_fn`, so follow-on rubric contributions compose naturally.

`skills`

- Accepts one skills directory.
- Resolves to `skills`.
- Extends `sandbox` by contributing that directory as the `skills` upload.
- CLI harnesses choose the destination path by setting `skills_path`.

`sandbox`

- Accepts nested sandbox config.
- Harnesses generally contribute `spec` and `runtime`.
- Tasksets generally contribute per-task `SandboxSeed`, scoring needs, and
  uploads.
- Resolves named resource objects such as:
  - `sandbox_request`
  - `sandbox_runtime`
  - `sandbox_scoring`
  - `sandbox_uploads`

`endpoint`

- Materializes an OpenAI-compatible interception endpoint.
- Owns endpoint server/tunnel teardown.
- Currently supports `openai_chat_completions`.

`user`

- Resolves a `User` responder object.
- Used by the base harness when there are no tool calls and a user turn may be
  needed.

`stop`, `cleanup`, `teardown`

- Lifecycle channels.
- Accept handler functions.
- Dispatch through the same lifecycle path as decorated methods.

## Harness

`Harness` owns the shared rollout loop. It is the only base harness exported
from top-level `verifiers`.

Important hooks:

- `setup_state(task, resources)`
- `get_prompt_messages(task, state, resources)`
- `get_env_messages(task, state, resources)`
- `get_model_request(task, state, resources)`
- `get_model_response(prompt, task, state, resources, ...)`
- `normalize_model_response(response, task, state, resources)`
- `add_model_response(prompt, response, task, state, resources, ...)`
- `submit_model_request(prompt, task, state, resources, ...)`
- `finalize_state(task, state, resources)`

The loop schedules model requests through a shared request path. Default
turn-based harness behavior waits for a request before preparing the next turn.
Endpoint-style harnesses can leave multiple requests in flight. Trajectory
steps are treated as a bag for training; ordering is display/debug metadata.

The default interaction pattern is:

1. Build prompt messages.
2. Submit a model request.
3. Append a `TrajectoryStep`.
4. If the assistant called tools, dispatch through `resources.tools`.
5. If there are no tool calls and a `user` resource exists, request a user turn.
6. If there are no tool calls and no user turn, mark the state completed.

Lifecycle decorators remain central:

- `@vf.stop`
- `@vf.cleanup`
- `@vf.teardown`

Tasksets and harnesses can also contribute lifecycle handlers through lifecycle
channels without subclassing.

There is no startup decorator. Rollout startup belongs in `setup_state`; global
runtime setup belongs in channel/resource resolution.

## Tools

Tools are resolved by the `tools` channel into a `ToolRegistry`.

Supported forms:

- plain Python callables
- `CallableTool`
- `Tool` schema definitions
- `ToolRegistry`
- `ToolInjector`
- `MCPServerSpec`

Harnesses see light handles:

```python
resources.tools["search"]
resources.tools.defs()
await resources.tools.call("search", resources, {"query": "..."}, task=task, state=state)
```

The registry owns:

- tool naming
- schema generation
- schema hiding for injected args
- callable dispatch
- MCP runtime connections
- hidden argument injection
- tool channel contributions
- teardown

Hidden tool arguments are injected through `ToolInjector`, not hard-coded call
branches. Built-in injectors cover:

- `resources`
- `state`
- `task`
- `client`
- `model`
- `sampling_args`
- `tools`

Custom routing/stateful arguments can be added through the tools channel:

```python
vf.ToolInjector(
    "sandbox",
    lambda context: context.resources.require("sandbox_runtime"),
)
```

For StatefulToolEnv-style migrations, this provides hidden argument injection
while keeping state serializable. Resources own runtime handles.

Tools can also declare channel requirements directly through `tool.channels()`.
The `tools` channel resolves those declarations before dependent channels, so a
tool can request sandbox runtime, cleanup handlers, or task-specific resource
configuration without requiring a custom harness.

Sandbox-backed tools use this path:

```python
vf.Harness(
    tools=[
        vf.SandboxPythonTool(
            sandbox=vf.SandboxSpec(image="python:3.11-slim"),
            sandbox_use="auto",
        )
    ]
)
```

Sandbox tool modes:

- `auto`: reuse `state["sandbox_id"]` when present, otherwise create a sandbox
  on first use.
- `existing`: require `state["sandbox_id"]`.
- `new`: create a tool-owned sandbox on first use.

Created sandbox IDs are serializable state fields. The sandbox runtime client
stays in resources. Tool-owned sandboxes are cleaned up through lifecycle
channels after taskset cleanup/scoring handlers run.

## Endpoint Harness

`EndpointHarness` is the pattern for arbitrary Python rollout logic that calls
a managed LLM endpoint.

It:

- contributes the `endpoint` channel
- starts/registers an endpoint per rollout
- provides an OpenAI-compatible client to `execute(...)`
- intercepts endpoint requests
- converts endpoint message/tool payloads into vf messages/tools
- forwards model calls through the shared harness request path
- delivers normal/streaming responses back to the endpoint caller

The overridable hook is:

```python
async def execute(self, task, state, resources, client) -> object:
    ...
```

This is where DSPy, LangChain, direct OpenAI-compatible calls, or Agents SDK
logic belongs. The first pass supports `openai_chat_completions`.

## CLI Harness

`CliHarness` builds on `EndpointHarness` for sandboxed CLI agents.

It:

- contributes `endpoint`
- contributes `sandbox`
- creates the sandbox from harness defaults plus taskset seeds
- uploads task assets and configured uploads
- writes instruction/system prompt files
- wires OpenAI-compatible env vars into the sandbox
- runs the CLI command as a background job
- captures stdout/stderr/logs/metrics into serializable state
- keeps the sandbox alive for scoring when requested
- cleans up sandbox resources through lifecycle hooks

`OpenCode` is the reference concrete CLI harness for coding agents.
`RLMHarness` is the concrete CLI harness for the RLM agent. It owns the
host-side RLM checkout cache, RLM install command, sandbox run command, RLM
environment variables, disabled-git shim, tool metric names, session metrics
collection, and `/task/rlm-skills` upload mapping.

## Harbor Taskset

`HarborTaskset` is the reference structured taskset.

It:

- discovers Harbor task directories
- reads `task.toml`
- reads `instruction.md`
- creates dataset rows
- declares per-task sandbox seeds
- declares sandbox scoring
- manages Harbor task metadata under `task.info`
- uses `HarborRubric`

`HarborRubric`:

- runs verifier scripts against the rollout sandbox
- uploads verifier-only assets immediately before scoring
- reads `/logs/verifier/reward.txt`
- falls back to `/logs/verifier/reward.json`
- deletes retained scoring sandboxes after scoring cleanup

Harbor feature parity is intended to mirror the existing `HarborEnv` /
OpenCode Harbor flows:

- task assets uploaded before rollout
- verifier assets hidden until scoring
- framework-managed network MCP servers when possible
- task env vars such as `HARBOR_TASK_NAME`
- same-sandbox scoring

## Current Reference TH Environments

Eight reference migrations exist:

- `environments/reverse_text_th`
- `environments/wiki_search_th`
- `environments/alphabet_sort_th`
- `environments/mcp_search_th`
- `environments/hello_endpoint_th`
- `environments/opencode_harbor_th`
- `environments/deepdive_th`
- `environments/mini_swe_agent_plus_th`

They cover:

- simple dataset + rubric + base harness
- tool-using taskset + base harness
- multi-turn user-channel interaction + base harness
- MCP server discovery/calls through the tools channel
- endpoint-only rollout logic through `EndpointHarness`
- Harbor structured taskset + OpenCode CLI harness
- web-research tools, hidden state injection, and judge rubric composition
- sandbox-lifetime tools, per-task sandbox images, and cleanup-time scoring

These are the examples to keep current while the API stabilizes.

## Research Environment Coverage

The research-environments repo has four important migration references for
this refactor:

- `deepdive`
- `mini_swe_agent_plus`
- `mini_swe_agent_plus_rlm`
- `rlm_swe`

`deepdive_th` ports the `deepdive` pattern into `Taskset + Harness`:

- dataset loading and split logic live in `Taskset(source=..., eval_source=...)`
- Serper-backed web tools are taskset tools
- hidden `state` injection records tool metrics without exposing state in tool
  schemas
- optional `finish` is a normal tool that marks state complete
- judge scoring and tool metrics compose through the rubric channel

`mini_swe_agent_plus_th` ports the Mini SWE tool pattern into
`Taskset + Harness`:

- the taskset owns dataset loading and per-row sandbox image selection
- the taskset owns SWE setup commands, stop conditions, and cleanup-time tests
- the base harness owns the rollout loop and tool dispatch
- bash and string-replacement are reusable sandbox-backed tools
- sandbox IDs and command/test output are serializable state fields
- sandbox clients and retry policy stay in resources

`math_python_th` uses the same tool pattern with `SandboxPythonTool`, showing
how a normal `Harness` can gain sandbox-lifetime Python execution without
becoming a CLI harness.

`rlm_swe` is the primary concrete reference for `RLMHarness`:

- `RLMHarness` owns the RLM checkout cache, install/run commands, RLM env vars,
  disabled-git shim, tool metric names, and RLM session metric extraction
- the SWE taskset should own dataset loading, per-row sandbox images,
  repository setup, skills, and scoring
- taskset-provided skills flow through the `skills` channel, which contributes
  a sandbox upload when a CLI harness declares `skills_path`

`mini_swe_agent_plus_rlm` and the other RLM envs define the next layer of RLM
support above the CLI harness. The shared harness request scheduler should
remain the model-submission path; env-specific RLM behavior should become
taskset channels, tool views, or small concrete harness options. The expected
capabilities are:

- root execution that can call model endpoints through the shared request path
- sub-request fanout with multiple in-flight model calls
- root/sub tool registries or tool views resolved through channels
- sandbox execution as a channel-backed resource, not a state object
- optional inclusion or exclusion of sub-request trajectory steps
- RLM metrics as rubric/channel contributions

This should land after the base request path, endpoint forwarding, and
sandbox-backed tool lifetimes have enough usage to make the harness contract
obvious.

## Task Naming And EnvGroup Routing

Taskset/harness envs use:

- `Task` for taskset rows
- `example_id` for row identity
- `info["env_id"]` only for EnvGroup routing metadata

`env_id` is not a normal dataset column for `Env`/`Taskset`. EnvGroup can inject
and consume routing data under `info["env_id"]` during preprocessing. Users
should not manually manage an `env_id` column for taskset/harness envs.

Schema behavior:

- `RolloutInput` has no `task` routing field.
- `RolloutOutput` has no `task` alias.
- dataset formatting strips top-level `env_id`
- `state_to_output` does not backfill `task`
- eval utility naming moves from task outputs to env outputs
- Harbor task names live under `info["task_name"]`

Saved files that contain only `task` routing labels may need migration before
use with taskset/harness readers.

## Conceptual Anchors

This design borrows a few useful patterns from resource config systems such as
Kubernetes without importing their full object model.

Spec/status split:

- channel config is declarative spec
- resources are resolved runtime/status
- state stores serializable observed rollout data

Defaulting before validation:

- each channel canonicalizes shorthand locally
- no global defaults or fallback compatibility branches

Custom resources:

- external channels use the same `Channel` / `ResourcePatch` machinery as
  built-in channels
- built-in and custom channels are not separate categories

Finalizers:

- channel-created runtime objects register teardown through lifecycle hooks
- endpoint servers, MCP runtimes, sandbox runtimes, caches, and registries all
  use the same resource lifecycle path

Out of scope:

- API versions on every config object
- generic patch strategies
- owner references
- reconcilers/controllers as user-facing concepts
- labels/selectors without a concrete selection use case

## Key Design Choices

`Env` does not subclass `MultiTurnEnv`.

- `Env` keeps the low-level `Environment` APIs stable while letting the new
  rollout model be cleaner.

`Harness.run` is final/shared.

- Model submission and trajectory construction should not be reimplemented per
  harness pattern.
- Endpoint and CLI harnesses forward requests into the same loop.

There are three harness patterns.

- `Harness` covers turn-based message/tool/user interaction.
- `EndpointHarness` covers arbitrary Python code that talks to a managed model
  endpoint.
- `CliHarness` covers sandboxed CLI processes that talk to that endpoint.

`Resources` is the runtime object bag.

- `task`, `state`, and `resources` are the harness contract.
- Active client/model/sampling/runtime are context-local internals on
  `Resources`.

Configuration belongs to tasksets, harnesses, and channels.

- Channel behavior should live in taskset/harness channel declarations.
- Harness constructor arguments should line up with channel-shaped config where
  possible. For example, sandbox runtime knobs should feed the `sandbox`
  channel; tools should feed the `tools` channel; endpoint settings should feed
  the `endpoint` channel. Ordinary behavioral knobs such as `max_turns` remain
  harness fields.
- Logging is a candidate channel once multiple harnesses need shared log
  routing, log collection, or runtime log sinks. Today examples use normal
  Python loggers or harness-specific log paths.

Channels are config-shaped and functional.

- Channel definitions own canonicalization and resolution.
- Target-channel merge semantics are channel-owned.
- `extends` is added only when a concrete cross-channel need emerges. Current
  examples are tools contributing monitor rubrics and skills contributing
  sandbox uploads.

`Rubric` itself is not refactored.

- Rubric composition is a channel concern.
- The existing `Rubric` / `RubricGroup` model remains intact.

## Migration Roadmap

### From Simple `MultiTurnEnv`

Old shape:

```python
class MyEnv(vf.MultiTurnEnv):
    async def setup_state(...): ...
    async def env_response(...): ...
```

New shape:

- dataset/rubric move to `Taskset(source=..., rubric=...)`
- prompt initialization moves to task rows and `Harness.setup_state`
- `env_response` logic moves to `Harness.get_env_messages`
- custom stop/cleanup/teardown decorators move to the harness or taskset
- non-serializable handles move to resources via channels

Use `MultiTurnEnv` directly when the env is tightly integrated and not intended
to share a reusable taskset or harness yet. Use `Env` when the task collection
or interaction pattern should compose with other pieces.

### From `ToolEnv`

Old shape:

```python
vf.ToolEnv(dataset=dataset, tools=[...], rubric=rubric)
```

New shape:

```python
vf.Env(
    taskset=vf.Taskset(source=lambda: dataset, tools=[...], rubric=rubric),
    harness=vf.Harness(max_turns=...),
)
```

Tool-call dispatch is handled by base `Harness` through `resources.tools`.
Automatic tool monitor metrics come from `tools` extending `rubric`.

### From `StatefulToolEnv`

Old shape:

- `setup_state` creates stateful resources or IDs
- `update_tool_args` injects hidden tool args

New shape:

- serializable IDs stay in state
- live objects live in resources
- hidden args use `ToolInjector`
- cleanup uses `@cleanup` or lifecycle channel hooks
- sandbox-backed tools use `SandboxBashTool`, `SandboxPythonTool`, or another
  `SandboxTool` subclass when the stateful resource is a sandbox lifetime

Stateful tools should accept explicit hidden parameters whose schemas are hidden
by the registry:

```python
async def run_query(query: str, db, state):
    ...
```

Then register injectors for `db` or other runtime objects.

### From `ComposableEnv`

Old split:

- `TaskSet`
- composable harness config
- `ComposableEnv` as the glue

New split:

- `Taskset`
- `Harness`
- `Env`
- channel resolution as the glue

Migration mapping:

- `TaskSet.get_dataset()` → `Taskset(source=...)`
- `TaskSet.get_sandbox_spec(...)` → `sandbox` channel / `SandboxSeed`
- `TaskSet.get_upload_dirs(...)` → `sandbox` channel `uploads`
- task workdir/instruction/config → `task.info`
- task-specific env vars → sandbox seed env vars
- harness install/run command → `CliHarness` / concrete harness constructor
- harness system prompt → harness `system_prompt`
- harness metrics path → `CliHarness(metrics_path=...)`
- upload mapping → `CliHarness(upload_mapping=...)`
- scoring rubric → taskset `rubric` / `rubric` channel

`ComposableEnv` is a correctness reference for sandbox upload, install,
metrics, and OpenCode flow behavior.

RLM-style composable envs migrate to `RLMHarness`:

```python
from verifiers.envs.experimental.modules.harnesses import RLMHarness

vf.Env(
    taskset=swe_taskset,
    harness=RLMHarness(
        workdir="/testbed",
        gh_token=gh_token,
        rlm_tools=["ipython", "summarize", "edit"],
        keep_sandbox_for_scoring=True,
    ),
)
```

The current `rlm_swe` reference env maps directly to this split: the SWE
taskset owns dataset/image/setup/scoring/skills, and `RLMHarness` owns the RLM
CLI runtime.

### From `CliAgentEnv`

Old shape:

- subclass `CliAgentEnv`
- override sandbox setup/env vars/command hooks

New shape:

- use `CliHarness` for generic sandboxed CLI execution
- subclass `CliHarness` only when the CLI has reusable behavior
- put task-specific sandbox/image/setup/upload data in taskset channels
- keep endpoint setup in `EndpointHarness`/`endpoint` channel

Concrete OpenCode behavior lives in `OpenCode`.

### From `HarborEnv` / OpenCode Harbor

Old shape:

- `HarborEnv` subclass owns Harbor loading and CLI agent behavior together

New shape:

```python
vf.Env(
    taskset=HarborTaskset(path),
    harness=OpenCode(...),
)
```

The taskset owns Harbor discovery, sandbox seeds, and scoring. The harness owns
OpenCode install/config/run behavior. Same-sandbox scoring is coordinated
through sandbox resources and cleanup lifecycle hooks.

### From Endpoint/Intercept Experiments

Old shape:

- interception logic often lived inside CLI env subclasses

New shape:

- use `EndpointHarness`
- implement `execute(task, state, resources, client)`
- let intercepted model calls flow through shared harness request submission

This is the migration path for DSPy, LangChain, Agents SDK, or direct
OpenAI-compatible endpoint callers.

## Verification State

Verification has focused on implementation sanity, not a full test campaign.

Recently exercised:

- targeted `ruff`
- targeted `ty`
- basic `Env(Taskset, Harness)` rollout with fake client
- hidden tool injection smoke
- unknown channel hard-fail smoke
- tools extending rubric smoke
- duplicate `system_prompt` hard-fail smoke
- circular channel dependency hard-fail smoke
- research-env reference TH ports have been created for `deepdive` and
  `mini_swe_agent_plus`
- targeted `ruff`, `ty`, and syntax checks for `deepdive_th` and
  `mini_swe_agent_plus_th`
- load-environment smoke checks for `deepdive_th` and `mini_swe_agent_plus_th`

Still needed:

- full `prime eval run` coverage for `reverse_text_th`
- full `prime eval run` coverage for `wiki_search_th`
- full `prime eval run` coverage for `alphabet_sort_th`
- full `prime eval run` coverage for `mcp_search_th`
- full `prime eval run` coverage for `hello_endpoint_th`
- full `prime eval run` coverage for `opencode_harbor_th`
- full `prime eval run` coverage for `deepdive_th`
- sandbox-backed smoke coverage for `mini_swe_agent_plus_th`
- focused unit tests for channel resolution
- focused unit tests for `Taskset.to_task`
- focused unit tests for `ToolRegistry` injection/schema hiding
- regression checks for existing `MultiTurnEnv`, `ToolEnv`,
  `StatefulToolEnv`, `SandboxEnv`, and `EnvGroup`

## Open Questions And TODOs

Channel relationship vocabulary:

- `extends` is the channel relationship in this implementation.
- Do not add `after` / `requires` until a concrete channel needs ordering
  without contribution.

Channel extension:

- `extend_fn` is the target-channel capability marker.
- It may be removable if all useful extensions can be ordered before target
  resolution.

Tool channel organization:

- `tools_channel.py` is dense.
- It owns callables, MCP runtime, handles, injectors, schema generation, and
  monitor rubric contribution.
- This is load-bearing, and is the biggest cleanup candidate.

Endpoint protocols:

- only OpenAI chat completions are implemented.
- Anthropic messages and Responses API support should be added only when a
  concrete harness needs them.

MCP lifecycle:

- environment-scope MCP servers work through the tool registry.
- task-scoped MCP declarations need more design before becoming recommended.

Sandbox/resource naming:

- current names are `sandbox_request`, `sandbox_runtime`, `sandbox_scoring`,
  and `sandbox_uploads`.
- These are conventions of the sandbox channel, not fields on `Resources`.

Rubric object namespace:

- resources are attached to rubrics through class objects.
- task/taskset-aware reward functions are supported, but the long-term object
  namespace may need more examples.

Package split:

- extractable modules should stay package-shaped.
- do not move them until Harbor/OpenCode and the simple examples have settled.

Docs:

- user-facing docs should cover both existing env stacks and the new
  taskset/harness path, with taskset/harness as the recommended path for new
  reusable envs.
