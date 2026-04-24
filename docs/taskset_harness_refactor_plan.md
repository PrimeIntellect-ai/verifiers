# Taskset + Harness Refactor Plan

Status: implementation branch
Date: 2026-04-23

This document records the current first-pass plan for the composable
`vf.Env(taskset=..., harness=...)` refactor. It intentionally
tracks the implementation direction in this branch, not the older design-doc
sketch literally.

## Branch Direction

Build from current `main`, not `codex/composable-packages-pr` / PR #1235.

PR #1235 remains useful reference material, especially for Harbor parsing,
OpenCode-style command builders, endpoint forwarding details, and sandbox
lifecycle edge cases. It should not be the base branch because it is far from current
`main`, blocked on merge, and centered on the older `ComposableEnv` package
split rather than the `Env` / `Taskset` / `Harness` API.

The golden path for new work is `vf.Env`. Existing `MultiTurnEnv` and current
environment subclasses stay supported, but are not the architectural target.

## Target Public Shape

Environment packages should be able to look like this:

```python
import verifiers as vf
from harnesses import OpenCode
from tasksets import HarborTaskset


def load_taskset(taskset_args=None) -> vf.Taskset:
    return HarborTaskset(**(taskset_args or {}))


def load_harness(harness_args=None) -> vf.Harness:
    return OpenCode(**(harness_args or {}))


def load_environment(taskset_args=None, harness_args=None) -> vf.Environment:
    taskset = load_taskset(taskset_args)
    harness = load_harness(harness_args)
    return vf.Env(taskset=taskset, harness=harness)
```

Core abstractions are exported through `verifiers`:

- `vf.Env`
- `vf.Task`
- `vf.Taskset`
- `vf.Harness`
- `vf.Channel`
- `vf.ChannelConfig`
- `vf.ChannelContext`
- `vf.ChannelMap`
- `vf.Resources`

Concrete tasksets and harnesses start under experimental module paths:

- `verifiers.envs.experimental.modules.tasksets`
- `verifiers.envs.experimental.modules.harnesses`

Those module folders are deliberately package-shaped so they can later move to
top-level `tasksets` and `harnesses`. Do not use `contrib`; it makes the
eventual package boundary less clear.

Only the `Harness` base is exported from top-level `verifiers`. It owns the
default in-process turn loop. Endpoint and CLI agent patterns, plus concrete
agents such as `OpenCode`, live in the harnesses module path.

## Code Ownership

The first pass lives mostly under `verifiers/envs/experimental`.

Likely permanent framework surface, currently staged in experimental:

- `verifiers/envs/experimental/env.py`
  - `Env`
- `verifiers/envs/experimental/task.py`
  - `Task`
- `verifiers/envs/experimental/taskset.py`
  - `Taskset`
- `verifiers/envs/experimental/harness.py`
  - `Harness`
- `verifiers/envs/experimental/channels/`
  - `Channel` and canonicalize/resolve primitives
  - one file per default channel; `DEFAULT_CHANNELS` is assembled in `__init__`
  - tool, sandbox, and user declaration/support code
  - no-op rubric fallback and rubric/resource attachment utilities
- `verifiers/envs/experimental/resources.py`
  - `Resources`, the resolved object bag and scoped runtime access point
  - context-local model client/model/sampling/runtime handles for active
    rollouts

The `channels` submodule is framework code that should remain in `verifiers`.
It is not part of the future `tasksets` / `harnesses` package split. The
package-shaped taskset and harness modules contribute channel declarations, but
channel application and runtime object materialization stay owned by the core
environment layer.

Extractable package code:

- `verifiers/envs/experimental/modules/tasksets/*`
  - `dataset_taskset.py`
  - `harbor_taskset.py`
  - future SWE, Lean, CP, math, HF adapters
- `verifiers/envs/experimental/modules/harnesses/*`
  - `endpoint_harness.py`
  - `cli_harness.py`
  - CLI sandbox / agent monitor rubrics live beside `CliHarness`
  - `opencode_harness.py`
  - future Codex, Claude Code, mini-SWE-agent, RLM, Terminus adapters

The older `verifiers.envs.experimental.composable` tree, `CliAgentEnv`,
`HarborEnv`, and `OpenCodeHarborEnv` remain correctness references. They are
not the new golden path, but the first-pass taskset/harness rollout behavior
should mirror their feature support closely enough that migration is direct.

## Core Model

The new code enforces three objects:

- `Task`: immutable, Pydantic-serializable taskset row.
- `State`: mutable, serializable rollout record.
- `Resources`: non-serializable global objects plus scoped runtime access.

`Taskset + Harness = Env`.

`Harness.run(task, resources)` owns rollout control flow. `Env` only resolves
resources, converts dataset rows to `Task`, and delegates.

`Taskset(source=..., eval_source=...)` is the public dataset-shaped interface.
The source values can be lazy callables or already materialized rows. Internally,
`Taskset.get_dataset()` and `Taskset.get_eval_dataset()` are the cached access
paths; concrete tasksets may still override those methods when discovery logic
is clearer as class behavior. `Env` passes the cached accessors into the
existing `Environment` runtime without calling them during construction.

New model calls go through the harness request scheduler. A harness produces a
`ModelRequest`; the shared loop submits it through the active model client on
`Resources` and appends a `TrajectoryStep` when the request completes. This path
can run requests concurrently, which is required for endpoint-driven callers
that spawn parallel sub-agents. Treat trajectory steps as an unordered bag;
ordering is only for display/debugging.

Do not build new functionality around `Environment.generate`; it is legacy
plumbing.

`Env` states should not contain live clients, resource managers, or other
non-serializable runtime handles. Those stay in `Resources`; state carries only
serializable rollout data and references such as sandbox IDs. New harness error
state is stored as a small serializable mapping with error type/message/repr
rather than the exception object.

CLI/sandbox harness behavior is intentionally aligned with the existing envs:

- Prime Tunnel by default for remote sandbox access to the managed endpoint.
- Sandbox creation throttling, retry policy, and threaded sandbox client sizing
  knobs from the current `SandboxMixin` path.
- Pinned OpenCode install flow, config generation, disabled-tool support, and
  log collection.
- Max-turn and timeout termination for forwarded endpoint calls.
- Endpoint tool-definition normalization and per-rollout schema caching.
- Sandbox and CLI-agent monitor metrics.
- Install, post-install, upload-directory, system-prompt, instruction-file, and
  harness-metrics hooks from `ComposableEnv`.
- Task-selected working directories via `AGENT_WORKDIR`; OpenCode runs from the
  env var so structured tasksets can set `/app` without rebuilding the command.
- Taskset-provided env vars override harness env vars except for protected
  runtime keys such as OpenAI routing and agent file paths.

Channels resolve the resources needed for taskset/harness interaction. Current
channel asks include:

- `system_prompt`: accepts zero or one string contribution.
- `tools`: callable tools, `CallableTool` declarations, schema-only tool defs,
  and `MCPServerSpec` declarations.
- `rubric`: shorthand/canonical rubric config materialized into the existing
  `Rubric` / `RubricGroup` scoring model.
- `sandbox`: nested sandbox coordination ask. Harnesses can contribute
  `spec/runtime`; tasksets can contribute scoring requirements or per-task
  sandbox seeds. Resolution materializes named resource objects such as
  `sandbox_request`, `sandbox_runtime`, and `sandbox_scoring`.
- `endpoint`: managed LLM endpoint ask. Endpoint-style harnesses contribute
  this when they require an interception endpoint; resolution materializes the
  endpoint server/tunnel handle and teardown handler before rollout.
- `stop`, `cleanup`, `teardown`: lifecycle decorator channels. They accept
  handler functions and dispatch through the same lifecycle path as class
  decorators.

`Resources` is a generic object bag, not a fixed schema of every possible
resource the ecosystem might learn about. It owns the resolved products of
channels under named objects; harnesses and tasksets only contribute
declarations. The rollout loop does not inspect channel maps. Environment-scope
and per-task channel asks are resolved before `harness.run(...)`, and harness
code consumes resolved objects such as
`resources.tools`, `resources.require("sandbox_runtime")`,
`resources.get("sandbox_request")`, and `resources.require("sandbox_uploads")`.
Default resource names are conventions of the built-in channels, while custom
channels can materialize their own names without changing the resource classes.

Channel names describe taskset/harness coordination domains, not necessarily
one-to-one resource attributes. A single channel can materialize several
resource fields.

Channel configs are dict-shaped user-facing declarations. A `Channel` defines
`outputs`, optional `output_types`, optional `requires`, optional
`contributes_to`, `canonicalize(config)`, and `resolve(configs, context)`. The
canonicalize step accepts ergonomic shorthand and normalizes it into one strict
internal shape; resolution turns that canonical shape into a `ResourcePatch`
containing named resource objects, lifecycle hooks, and optional follow-on
channel contributions. `output_types` is the single touchpoint for automatic
`Resources.require(...)` validation; adding a channel should not require adding
a matching property to `Resources`.

Channels can declare ordering through `requires` and `contributes_to`; resolution
orders channels accordingly and only hard-fails on circular dependencies. Some
channels can also opt into `extendable=True`, allowing later contributions after
the channel has resolved. Rubric is extendable, so the tools channel can
contribute tool-monitor rubric entries without the rubric channel knowing about
tools. Scalar channels such as `system_prompt` remain non-extendable. Unknown
channel names hard-fail unless a custom `Channel` is registered by the taskset
or harness.

Lifecycle additions are ordinary `stop`, `cleanup`, and `teardown` channels, so
tasksets and harnesses can add decorator handlers without subclassing.

Resource assembly starts from `DEFAULT_CHANNELS`, assembled in
`verifiers.envs.experimental.channels.__init__` from the per-channel modules,
and overlays any `channel_definitions()` supplied by the taskset or harness.
Those channels are resolved through the same path; a channel from an external
package is not a different category of thing.

`Env` does not have a separate config object in this pass. Runtime options
continue to flow through the existing low-level `Environment` run APIs, while
channel behavior lives in channel definitions. Tasksets and harnesses
contribute config-shaped asks plus named functional objects via
`channel_objects()`.

Per-task channel contributions go through the same merge path as environment
channels. `channels(task=None)` resolves taskset + harness environment asks;
`channels(task)` resolves the same pair with task row overrides included.
Duplicate scalar asks still reject unless the channel definition explicitly
supports combining them.

## Legacy `task` Removal

The old rollout identifier named `task` is hard-removed. Normal rollout inputs,
states, outputs, tasks, and rubrics do not carry an environment identifier.
`env_id` remains in eval CLI/config/metadata paths as the environment package
selector, not as per-rollout data.

The name `task` is now reserved for the immutable `vf.Task` object. New `Env`
rollouts pass that object to the harness and rubrics directly rather than
storing it in state. Existing legacy envs should not rely on `state["task"]` as
an environment label.

Impact already reflected in the branch:

- `RolloutInput` has no `task` or `env_id` routing field.
- `RolloutOutput` has no `env_id` or `task` alias.
- `Environment` dataset formatting creates `example_id` and `prompt`, not an
  `env_id` column.
- Experimental `Taskset` formatting also strips any legacy top-level `env_id`
  column.
- `EnvGroup` injects routing metadata under `info["env_id"]` during dataset
  preprocessing and delegates to the routed child env. The consumed route is
  stripped before the child env sees the input; nested EnvGroups receive only
  the remaining route.
- `state_to_output` no longer backfills `task`.
- `eval_utils.get_task_outputs` is removed; use `get_env_outputs`.
- `State.get(...)` forwards input fields the same way `State.__getitem__` does,
  so `answer`, `info`, `prompt`, and `example_id` are consistently available to
  rubrics and serializers.
- Harbor task names live under `info["task_name"]`.

Older saved result files with only `task` may need migration before use in
newer readers. That is acceptable for this refactor.

## Rubrics

Do not change the `Rubric` / `RubricGroup` scoring model just to match the
design sketch.

Current plan:

- Use existing `Rubric` and `RubricGroup` behavior.
- Treat scoring as the `rubric` channel.
- `rubric_channel` canonicalizes shorthand into:
  - `rewards`: reward functions, default weight `1.0`
  - `metrics`: metric functions, default weight `0.0`
  - `rubrics`: existing `Rubric` instances/classes
- Shorthand examples:
  - `"rubric": "my_rubric"` means a named `Rubric` object/class.
  - `"rubric": ["exact_match"]` means reward functions.
  - `"rubric": {"rewards": ["exact_match"], "metrics": ["num_tool_calls"]}` uses
    bucket defaults.
- Use a no-op rubric when no rubric is contributed.
- Inject `resources` as the only named rubric class object from this path.
- Use `Rubric.score_objects(state)` and `Rubric.group_score_objects(states)` as
  the extensible object namespace for reward functions. Additional names are
  registered through score object providers instead of being stored in state.
- Let reward functions accept `task` for the new `vf.Task` when a taskset-backed
  env injects one. Legacy env states no longer carry the retired
  environment-label field.

This keeps the old scoring entrypoints (`score_rollout`, `score_group`) intact
while giving new taskset-backed rubrics access to resolved resources. `Env`
registers task/task-list score object providers on the resolved rubric, derived
from each state's rollout input, without storing those task objects in state.

## Rollout Identity

Do not add a public `rollout_id` yet.

The existing `trajectory_id` remains available for trainer/logging surfaces.
Interception and sandbox code can generate private request keys or sandbox
names internally, but those are implementation details, not a `RolloutInput`
contract.

## First-Pass Build Sequence

### 1. Experimental Core

Add the kernel under `verifiers/envs/experimental`:

- `Env`
- `Task` / `Taskset`
- `Resources`
- `ToolRegistry`
- generic `Channel` primitives in `experimental/channels/channel.py`
- concrete `*_channel.py` files in `experimental/channels`
- `Harness`

Expose the core symbols through `verifiers.__init__` as `vf.*`.

### 2. Shared Harness Loop

Implement the final rollout loop on `Harness`:

- `setup_state(task, resources)`
- `get_prompt_messages(task, state, resources)`
- `get_env_messages(task, state, resources)`
- `get_model_request(task, state, resources)`
- `get_model_response(prompt, task, state, resources)`
- `add_model_response(state, prompt, response, resources)`
- `submit_model_request(prompt, task, state, resources, ...)`
- `finalize_state(task, state, resources)`

`Harness.add_model_response` writes the same `TrajectoryStep` shape that
trainers already consume. In-process turn-based environments, `EndpointHarness`,
and `CliHarness` all use this shared path instead of implementing separate
request/trajectory loops.

`Harness.run` schedules model requests rather than forcing one blocking request
at a time. The default turn-based path waits for each request before preparing
the next prompt; endpoint-style harnesses can leave multiple requests in flight
and append steps as each completes. A small per-rollout lock guards trajectory
mutation so concurrent completions do not race.

`finalize_state` renders the serializable end state. `@cleanup` handlers release
local runtime objects and run after finalization or cancellation. Additional
stop/cleanup/teardown behavior is contributed through lifecycle channels and
dispatched through the same path as class decorators. There is no separate
startup hook; harness-specific rollout startup belongs in `setup_state`, while
environment-scope setup belongs in resources.

Default `Harness` behavior dispatches OpenAI-style model tool calls through
`resources.tools`. If the assistant makes no tool calls, `Harness` asks
`resources.user` for another user turn when present; if no user turn is
available, it marks `state["is_completed"] = True`, which is picked up by the
shared stop-condition path.

Tool support should cover:

- plain Python callables
- `CallableTool` declarations for explicit names/descriptions/injected args
- MCP stdio servers via `MCPServerSpec`
- stateful tools whose hidden arguments are injected by registered
  `ToolInjector` entries. Built-in injectors cover `resources`, `state`,
  `task`, `client`, `model`, `sampling_args`, and `tools`; custom injectors
  can be contributed through the tools channel.

Tool schemas hide injected routing args from the model. Harnesses see light
tool handles through `resources.tools[...]` and use `resources.tools.call(...)`
for dispatch, while the registry owns actual calls, hidden argument injection,
MCP runtime state, and teardown.

### 3. Endpoint Harness Pattern

Implement `EndpointHarness` around the existing OpenAI-compatible endpoint
server:

- Start a managed endpoint for external rollout code that speaks a standard
  LLM API.
- Register one internal request key per harness run.
- Launch `execute(task, state, resources, client)` as the overridable endpoint
  hook. This is where DSPy, LangChain, direct OpenAI-compatible calls, or other
  endpoint-facing Python code lives.
- Convert endpoint request messages/tools into vf messages/tools.
- Forward each request through `Harness.submit_model_request`.
- Deliver normal and streaming responses back to the endpoint caller.
- Append forwarded model calls as trajectory steps via the shared harness path.

The first pass implements `api_client_type="openai_chat_completions"` because
that matches current CLI usage. Endpoint runtime setup is channel-owned:
`EndpointHarness` contributes an `endpoint` channel ask, and resolution must
materialize a valid endpoint before the harness can run.

### 4. CLI Harness Pattern

Implement `CliHarness` as the sandboxed CLI pattern on top of
`EndpointHarness`:

- Create a sandbox from harness defaults plus taskset per-task sandbox seeds.
- Upload task-provided files/directories.
- Wire `OPENAI_BASE_URL`, model, timeout, and auth env vars.
- Run the CLI command as a background job.
- Capture stdout/stderr/logs into serializable state.
- Keep the sandbox alive for scoring only when the taskset requests it.
- Publish the endpoint server through Prime Tunnel by default so sandboxed CLI
  processes can reach it without an explicit `endpoint_url`.
- Keep endpoint server/tunnel handles on `Resources`, prepared by endpoint
  channel resolution, so state stays serializable.

Reference implementation: `OpenCode`.

### 5. Tasksets

Add package-shaped taskset adapters under
`verifiers/envs/experimental/modules/tasksets`:

- `DatasetTaskset`
- `HarborTaskset`
- `HarborRubric`

`HarborTaskset` is the reference structured taskset. It should parse
`task.toml`, read `instruction.md`, declare per-task sandbox seeds, start
declared network MCP servers when possible, health-check them before the agent
runs, and provide same-sandbox verifier scoring.
The CLI sandbox should receive only rollout-visible task assets before rollout;
verifier assets such as `tests/` are uploaded by the rubric immediately before
scoring.

Harbor compatibility mirrors `HarborEnv` / composable Harbor:

- Upload `instruction.md` and `task.toml` to `/task` before agent execution.
- Create `/task`, `/app`, `/logs/verifier`, `/tests`, and `/oracle`.
- Publish `HARBOR_TASK_NAME`, `HARBOR_TASK_DIR`,
  `HARBOR_INSTRUCTION_PATH`, and network MCP URL env vars.
- Rewrite framework-managed network MCP URLs to localhost inside the sandbox,
  patch `/etc/hosts` for declared hostnames, and wait for TCP readiness.
- Upload `tests/` to `/tests` and `solution/` to `/oracle` only during
  scoring.
- Run `bash test.sh` from `/tests`.
- Prefer `/logs/verifier/reward.txt`, then fall back to
  `/logs/verifier/reward.json`.

### 6. Package Split

Only after the experimental contracts and real Harbor/OpenCode flow are solid:

- Move extractable code into top-level package repos / packages.
- Keep `vf.Taskset`, `vf.Harness`, and `vf.Env` as the stable core imports.
- Add user-facing `tasksets` and `harnesses` imports once boundaries are
  stable.

## Validation Plan

No test campaign is part of the initial implementation push. Keep building
until the first broad scope is present, then add tests.

Later validation should cover:

- `Task` immutability and serialization.
- Dataset row to `Task` conversion.
- absence of per-rollout `env_id` plus EnvGroup routing via `info["env_id"]`.
- Channel merge determinism and tool collision errors.
- `Resources.rollout(...)` context-local client/model/sampling/runtime
  isolation.
- shared harness request scheduling against a mock model client.
- default `Harness` trajectory construction, tool dispatch, and no-tool/user
  completion behavior.
- `EndpointHarness` parallel multi-call trajectory capture.
- `CliHarness` sandbox creation, command execution, cleanup, and retained
  sandbox scoring.
- `HarborTaskset + OpenCode` structured flow.
- Existing `MultiTurnEnv`, `ToolEnv`, `StatefulToolEnv`, `SandboxEnv`, and
  `EnvGroup` regression paths.

## Review Flags

Minor choices made in this branch that deserve review after implementation:

- Concrete taskset/harness adapters are not exported as top-level `vf.*`
  symbols yet. They live under experimental module package paths.
- `task`, `state`, and `resources` remain the only rollout objects in the
  harness contract. Per-rollout client/model/sampling/runtime values are
  context-local internals on `Resources`, not a fourth object.
- Sandbox CLI endpoint forwarding now uses Prime Tunnel by default through the
  endpoint channel.
- MCP server support in `HarborTaskset` currently lives in sandbox setup
  commands. A later tool/spec channel can replace that once the core lifecycle
  is proven.
- Per-task MCP tool declarations can be represented by task channels, but the
  clean lifetime boundary for task-scoped MCP resources should be reviewed
  before making that a recommended pattern.
- Channel shorthand is accepted only inside each channel's canonicalize step.
