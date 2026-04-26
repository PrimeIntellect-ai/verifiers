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
- `vf.Binding`
- `vf.BindingContext`
- `vf.Channel`
- `vf.ChannelConfig`
- `vf.ChannelContext`
- `vf.ChannelMap`
- `vf.RunConfig`
- `vf.EndpointConfig`
- `vf.CliConfig`
- `vf.CliPaths`
- `vf.CliMetrics`
- `vf.SandboxConfig`
- `vf.SandboxSetup`
- `vf.SandboxRuntime`
- `vf.SandboxScoring`
- `vf.CallableTool`
- `vf.ToolInjector`
- `vf.ToolHandle`
- `vf.ToolRegistry`
- `vf.Toolset`
- `vf.StateBinding`
- `vf.TaskBinding`
- `vf.ResourceBinding`
- `vf.MCPTool`
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
    rubric=lambda: load_rubric(...),
    tools=lambda: load_toolset(...),
)
```

`source`, `eval_source`, `rubric`, and `tools` all support lazy loaders. Dataset
sources cache after first load; rubric and tool loaders resolve through channel
resolution inside the env worker.

`source` and `eval_source` can be:

- callables returning a dataset-like object
- already materialized dataset-like objects
- iterable row mappings
- `None`

`tools` can be:

- a `Toolset`
- a list of tool objects/functions
- a zero-arg callable returning either

Bare single-tool callables are not accepted as the `tools` argument. Wrap one
tool in a list.

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

`resources.scoring` controls whether scoring happens inside `Harness.run` or is
left to the environment/runner:

- `"run"` scores at the end of `Harness.run` after harness cleanup. This mode
  rejects group reward functions.
- `"group"` defers scoring to `Environment.run_rollout` /
  `Environment.run_group`. `Env` uses this mode by default because it has a
  taskset.
- `"none"` skips scoring.

`Resources(harness=...)` defaults to `"run"`. `Resources(taskset, harness)`
defaults to `"group"`. Harness cleanup still always runs at the end of
`Harness.run`; rubric cleanup runs after rubric scoring.

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

- Accepts `Toolset`, lists of Python callables, `CallableTool`, schema-only
  `Tool` definitions, `ToolRegistry`, `ToolInjector`, and `MCPTool`.
- Resolves to `ToolRegistry`.
- Registers registry teardown through lifecycle hooks.
- Recursively flattens nested `Toolset` objects.
- Extends `rubric` with automatic tool-monitor metrics when tools exist.
- Extends lifecycle/resource channels with tool and toolset contributions.

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
- `add_model_response(prompt, response, task, state, resources, ...)`
- `submit_model_request(prompt, task, state, resources, ...)`
- `render_timing(task, state, resources)`
- `render_completion(task, state, resources)`

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
6. If there are no tool calls and no user turn, stop via the default stop
   condition.
7. On exit, run harness cleanup. The base `render_state` cleanup records timing
   and completion before lower-priority cleanup handlers run.
8. If `resources.scoring == "run"`, run single-rollout scoring and rubric
   cleanup.

Completion is still owned by stop handlers. `state["is_completed"]` and
`state["stop_condition"]` are framework-rendered fields written by
`is_completed`. Tools and users that want to end the rollout set
`state["done"] = True`; the built-in `done` stop handler then exits through the
normal decorator path and records `stop_condition = "done"`. Finish-tool
patterns should write their domain output, such as a final answer, and set
`done` rather than writing `is_completed` directly.

Lifecycle decorators remain central:

- `@vf.stop`
- `@vf.cleanup`
- `@vf.teardown`

Tasksets and harnesses can also contribute lifecycle handlers through lifecycle
channels without subclassing.

There is no startup decorator. Rollout startup belongs in `setup_state`; global
runtime setup belongs in channel/resource resolution.

Lifecycle timing is localized to the owner:

- Harness cleanup runs at the end of `Harness.run`. It finalizes serializable
  rollout state and releases rollout-local resources that are unrelated to
  scoring before the serialization boundary. The built-in `render_state`
  cleanup records timing and completion in this phase.
- Rubric cleanup is scoring cleanup. It deletes or finalizes resources that
  scoring needed, and runs after `score_rollout` or `score_group`.
- `Environment.cleanup` follows the same rollout-finalization semantics for
  `MultiTurnEnv`-style stacks. In taskset/harness `Env`, this layer is mostly
  a compatibility surface because `Harness.run` already owns rollout cleanup.
- Channel `teardown` is environment-worker teardown. In `Env`, environment
  teardown calls `rubric.teardown()` first, then `resources.teardown()`, which
  dispatches channel teardown handlers.

Injected cleanup handlers target either harness or rubric cleanup through
channel config. Shorthand targets harness cleanup:

```python
channels={"cleanup": cleanup_state}
channels={"cleanup": {"harness": cleanup_state, "rubric": cleanup_after_score}}
```

## Tools

Tools are resolved by the `tools` channel into a `ToolRegistry`.

Supported forms:

- `Toolset`
- plain Python callables
- `CallableTool`
- `Tool` schema definitions
- `ToolRegistry`
- `ToolInjector`
- `MCPTool`

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

`Toolset` is the canonical way to package related tools with hidden bindings and
setup requirements:

```python
def load_toolset():
    wiki = {
        "search_index": lambda: ...,
        "read_page": lambda page_id: ...,
    }

    async def search_pages(query: str, wiki):
        ...

    @vf.cleanup(priority=50)
    async def cleanup_wiki(task, state, resources):
        ...

    return vf.Toolset(
        bindings={"wiki": wiki},
        channels={"cleanup": {"harness": cleanup_wiki}},
        name="wiki_search",
        tools=[search_pages],
    )
```

Toolsets can contain other toolsets. The tools channel flattens them into one
`ToolRegistry`, merges bindings/injectors, and carries toolset-level channel
contributions forward. This is the grouped tool pattern; there is no separate
`ToolsetGroup`.

The `wiki_search_th` reference uses this pattern for Chroma-backed retrieval.
The toolset lazily creates a `chromadb.PersistentClient` binding and contributes
decorated teardown that releases the client reference when the environment
worker exits. Decorators are optional for channel-passed lifecycle functions,
but priority metadata is honored by the same handler ordering path.

Python-facing toolset APIs stay object-native: callers pass `Toolset`,
callables, bindings, and channel contributions directly. TOML-facing toolset
config should name a loader and provide plain args, then load into a normal
`Toolset` before channel resolution.

Bindings are plain `dict[str, object]` values. If a binding value is a zero-arg
callable, it is evaluated lazily and cached by the resolved tool registry. A
tool parameter whose name matches a binding is hidden from the model-visible
schema and injected at call time.

Bindings can also resolve values from call context:

```python
vf.Toolset(
    bindings={
        "sandbox_id": vf.StateBinding("sandbox_id"),
        "answer": vf.TaskBinding("answer"),
        "allow_web": vf.ResourceBinding("allow_web"),
        "tenant": vf.TaskBinding(lambda task: task["tenant"]),
        "api_key": vf.StateBinding(lambda state: state["credentials"]["api_key"]),
        "custom": vf.Binding(lambda ctx: ctx.state["custom"]),
    },
    tools=[run_command],
)
```

Contextual bindings are resolved each tool call. `StateBinding`, `TaskBinding`,
and `ResourceBinding` accept either a direct key/name or a callable over their
scope for nested access. String task bindings read a direct task key or
attribute; nested task conventions should use a callable. If the referenced
state, task, or resource key is missing, tool execution fails.

Hidden tool arguments are injected through `Toolset` bindings or `ToolInjector`,
not hard-coded call branches. Built-in injectors cover:

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
channels. If scoring does not need them, cleanup targets the harness phase; if
scoring does need them, cleanup targets the rubric phase.

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

Endpoint configuration is grouped:

```python
EndpointHarness(
    endpoint=vf.EndpointConfig(port=..., url=..., secret=...),
    run=vf.RunConfig(max_turns=..., parallel_model_requests=True),
    channels={"system_prompt": "..."},
)
```

## CLI Harness

`CliHarness` builds on `EndpointHarness` for sandboxed CLI processes.

It:

- contributes `endpoint`
- contributes `sandbox`
- creates the sandbox from harness defaults plus taskset seeds
- uploads task assets and configured uploads
- writes instruction/system prompt files
- wires OpenAI-compatible env vars into the sandbox
- runs the CLI command as a background job
- captures stdout/stderr/logs/metrics into serializable state
- keeps the sandbox alive when rubric scoring needs it
- cleans up non-scoring sandbox resources during harness cleanup and
  scoring-retained sandbox resources during rubric cleanup

`OpenCode` is the reference concrete CLI harness for coding workflows.
`RLMHarness` is the concrete CLI harness for RLM. It owns the
host-side RLM checkout cache, RLM install command, sandbox run command, RLM
environment variables, disabled-git shim, tool metric names, session metrics
collection, and `/task/rlm-skills` upload mapping.

CLI configuration is intentionally grouped so the top-level constructor does
not grow with every sandbox/runtime knob:

```python
CliHarness(
    cli=vf.CliConfig(
        command="agent run ...",
        workdir="/workspace",
        paths=vf.CliPaths(
            instruction="/task/instruction.md",
            system_prompt="/task/system_prompt.md",
            log="/logs/cli.log",
        ),
        metrics=vf.CliMetrics(path="{workdir}/metrics.json"),
    ),
    sandbox=vf.SandboxConfig(
        spec=vf.SandboxSpec(image="python:3.11-slim"),
        setup=vf.SandboxSetup(
            uploads={"checkout": lambda: "/tmp/checkout"},
            upload_mapping={"checkout": "/workspace/checkout"},
            install_command="pip install -e /workspace/checkout",
        ),
        runtime=vf.SandboxRuntime(client_max_workers=64),
        scoring=vf.SandboxScoring(retain=True),
    ),
    endpoint=vf.EndpointConfig(),
    run=vf.RunConfig(max_turns=-1, parallel_model_requests=True),
)
```

These config objects are pydantic models. Python callers can pass model
instances or plain nested dicts. TOML-facing paths should use plain values and
loader references rather than embedding live Python callables directly.

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

Nine reference migrations exist:

- `environments/reverse_text_th`
- `environments/wiki_search_th`
- `environments/alphabet_sort_th`
- `environments/mcp_search_th`
- `environments/hello_endpoint_th`
- `environments/math_python_th`
- `environments/opencode_harbor_th`
- `environments/deepdive_th`
- `environments/mini_swe_agent_plus_th`

They cover:

- simple dataset + rubric + base harness
- tool-using taskset + base harness
- multi-turn user-channel interaction + base harness
- MCP server discovery/calls through the tools channel
- endpoint-only rollout logic through `EndpointHarness`
- sandbox-backed Python tools and cleanup-time state rendering
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
becoming a CLI harness. Its toolset contributes cleanup that records Python REPL
state before the sandbox tool deletes its owned sandbox.

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
  the `endpoint` channel. Ordinary behavioral knobs such as `max_turns` live
  under `RunConfig`.
- Logging is a candidate channel once multiple harnesses need shared log
  routing, log collection, or runtime log sinks. Today examples use normal
  Python loggers or harness-specific log paths.

Typed config models are the TOML bridge, not a replacement for Python objects.

- Runtime Python APIs can pass callables, `Toolset`, `Rubric`, and channel
  objects directly.
- TOML-friendly configs should refer to named loaders plus plain args, for
  example a future `ToolsetConfig(loader="pkg.env:load_toolset", args={...})`.
- Resolution is responsible for turning loader refs into objects before channel
  resolution.

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
    harness=vf.Harness(run=vf.RunConfig(max_turns=...)),
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
- `TaskSet.get_skills_dir()` → `skills` channel or `sandbox` upload named
  `"skills"`
- `TaskSet.setup(...)` → sandbox seed files/uploads/setup commands, or a
  taskset-contributed sandbox setup channel when arbitrary Python setup is
  needed
- `TaskSet.validate_instance(...)` → taskset validation helper using the same
  sandbox channel objects and setup path as rollouts
- task workdir/instruction/config → `task.info`
- task-specific env vars → sandbox seed env vars
- harness install/run command → `CliHarness` / concrete harness constructor
- harness system prompt → harness `system_prompt`
- harness metrics path → `CliConfig(metrics=CliMetrics(...))`
- upload mapping → `SandboxConfig(setup=SandboxSetup(upload_mapping=...))`
- harness-owned upload dirs → `SandboxSetup(uploads=...)`
- harness post-install uploads/scripts → `SandboxSetup(post_install_uploads=...,
  post_install_command=...)`
- harness tool names → `CliMetrics(tool_names=...)` or automatic tool monitor
  metrics from the `tools` channel
- scoring rubric → taskset `rubric` / `rubric` channel

`ComposableEnv` is a correctness reference for sandbox upload, install,
metrics, and OpenCode flow behavior.

Coverage status:

- CLI install/run, instruction upload, system prompt upload, log collection,
  metrics collection, upload mapping, skills upload, post-install files, and
  same-sandbox scoring are represented in `CliHarness`, `SandboxConfig`,
  `skills`, `sandbox`, and rubric cleanup.
- Task-to-harness tool composition is materially stronger than in
  `ComposableEnv`: tasksets and harnesses can both contribute tools and
  toolsets through the `tools` channel.
- Sandbox clients and other live handles move from state to resources. Existing
  rubrics that read `state["sandbox_client"]` should accept `resources` and use
  `resources.require("sandbox_runtime").client` while keeping only
  `state["sandbox_id"]` in state.
- The main parity gaps are taskset validation ergonomics, arbitrary Python
  sandbox setup hooks, and explicit runtime override syntax for sandbox specs.
  These should be added as taskset/channel APIs rather than as new `Env`
  constructor options.

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
- full pre-push hook pass: `ruff`, `ruff format`, AGENTS sync, and `ty`
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
- load-environment smoke checks for `wiki_search_th`, `math_python_th`,
  `mcp_search_th`, and `opencode_harbor_th`
- focused tests for decorator ordering, environment stop rendering, and
  OpenCode Harbor loading

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

## PR Review Follow-Ups

Standalone harness resources:

- `Env` has a clean golden path: `Resources(taskset, harness)` then
  `harness.run(task, resources)`.
- Standalone harness use is not settled. The intended shape is still that a
  harness can be tested independently with resolved resources, client/model,
  and scoring mode without constructing a taskset or env.
- The next hardening pass should decide whether this is
  `Resources(harness=..., client=..., model=...)`, a harness helper, or a small
  runner utility. Avoid adding extra `run(...)` arguments.

Config loader references:

- Pydantic config models now group harness/runtime settings and provide a TOML
  bridge for plain values.
- Loader-reference config is still conceptual. A future `ToolsetConfig` should
  name a loader plus plain args, then materialize a normal `Toolset` before
  channel resolution.
- The same pattern may apply to rubric, taskset, and harness configs, but
  Python object APIs should continue to accept live callables and objects.

Tool transport resolution:

- Callable tools, `CallableTool`, schema-only tools, `MCPTool`, injectors, and
  `Toolset` groups all resolve through the tools channel.
- Automatic transport conversion is not done yet. The main open case is a
  callable tool that should be presented to a CLI or sandboxed process through
  MCP, while preserving hidden args, sandbox routing, and cleanup behavior.
- Practical boundaries to define: shared vs new sandbox use, task-scoped MCP
  servers, auth/secrets, and which conversions are framework-owned versus
  harness-owned.

Tools channel density:

- `tools_channel.py` is the largest and most complex new file.
- It owns registry behavior, MCP runtime, handles, schema filtering, hidden
  injection, monitor rubric contribution, and toolset channel propagation.
- This is acceptable for the first pass because it keeps one golden path, but
  it is the clearest candidate for a careful split after tests lock behavior.

Lifecycle and scoring cleanup:

- Harness cleanup, rubric cleanup, and teardown now have distinct timing.
- The important behavior to harden is cleanup target selection:
  harness cleanup prepares serializable state after `Harness.run`; rubric
  cleanup runs after scoring; teardown is process lifetime.
- Add tests for cleanup ordering, cleanup functions that ask for
  `task/state/resources` by name, and sandbox deletion timing in both
  `resources.scoring == "run"` and `"group"` modes.

Sandbox and CLI parity:

- `CliHarness` now uses grouped config, but it still maps config values back
  into many legacy-shaped attributes.
- Sandbox setup is split across the sandbox channel, `CliHarness`, and
  sandbox-backed tools.
- The next consolidation point is deciding which pieces of upload/setup/retain
  behavior belong entirely to the sandbox channel versus concrete CLI
  harnesses.

Parallel request semantics:

- The shared harness request path supports multiple in-flight model requests,
  and endpoint harnesses forward intercepted requests into that path.
- More RLM-style coverage is needed for sub-agent fanout, trajectory inclusion
  policy, max-turn accounting, request cancellation, and metric propagation
  across nested harness workflows.

Task/state boundary:

- The desired direction is that immutable row data lives on `Task` and mutable
  rollout progress lives on `State`.
- Current states still carry compatibility fields such as prompt/input/timing
  and serialized task snapshots.
- Centralize task-derived score/output views before deleting those state
  copies.

Verification gaps:

- The PR has static/type/format coverage and targeted smoke tests, but still
  needs real `prime eval run` coverage across all `*_th` examples.
- Sandbox-heavy examples need more attention: `mini_swe_agent_plus_th`,
  `opencode_harbor_th`, and RLM-style harness flows.
- Existing stacks need regression coverage so `MultiTurnEnv`, `ToolEnv`,
  `StatefulToolEnv`, `SandboxEnv`, `CliAgentEnv`, `ComposableEnv`, and
  `EnvGroup` remain valid migration sources.

Package boundary:

- Core primitives that should stay in `verifiers` are now under experimental
  root and channels/resources.
- Package-shaped tasksets and harnesses live under
  `experimental/modules/...`.
- Before extraction, confirm names and exports for `Taskset`, `Harness`,
  config models, `Toolset`, and channel primitives so eventual top-level
  packages do not collide with core framework modules.

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
