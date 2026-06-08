# v1 vs vf-nano

This document describes the current v1 refactor checkpoint relative to
`~/dev/vf-nano`.

## Shared Philosophy

Both systems separate serializable specification from live execution:

- Task data is immutable and serializable.
- Rollout output is a strict Pydantic record.
- Live model clients, runtimes, MCP sessions, and file handles stay in live
  runtime objects, not task/state records.
- Runtime backends sit behind one execution contract.
- Toolset/User implementations are loader-backed owner objects rather than
  Python callables passed through config.
- Decorated Python methods are allowed only on loaded owner objects.

## Naming Map

| vf-nano | v1 checkpoint | Notes |
|---|---|---|
| `Trace` | `State` | v1 keeps `State` as the canonical rollout record. |
| `trajectory` | `transcript` | v1 hard-renames the live/output field to transcript. |
| `Rollout` | `Harness.run(...)` lifecycle | v1 keeps the authoring name `Harness`; it is the agent. |
| `Agent.run(...)` | `Harness.run_with_context(...)` | v1 subclasses branch at the live context boundary. |
| `RolloutContext` | `Context` | v1 context includes task, state, model client, runtime, tools, user, and parent. |
| `Runtime` | `Runtime` | v1 now uses Runtime for the live backend handle. |
| `RuntimeConfig` | `RuntimeConfig` | Serializable runtime spec. |
| `ToolServer` | `Toolset` / `User` | nano exposes generic tool servers; v1 keeps Verifiers authoring names and uses loader-backed configs. |

## v1 Additions Over nano

### Taskset/Harness Authoring

v1 preserves the Verifiers authoring split:

- `Taskset` owns task data, prompts, users, toolsets, metrics, rewards, and
  task lifecycle.
- `Env` owns the selected group advantage function.
- `Harness` owns agent execution, runtime use, protocol interception, and
  reusable execution mechanisms.
- `Env` is a thin adapter for eval/training loaders.

Nano calls the execution object an `Agent`. v1 keeps `Harness` because that is
the existing Verifiers design language, but the object now behaves like nano's
agent.

### State Extras Schema

v1 has `state.extras` as the only user-owned mutable rollout surface. Taskset
and harness configs may each provide an `Extras` schema/default object; v1
realizes a combined schema and rejects duplicate keys.

Nano's `Trace` can be subclassed to add typed fields. v1 keeps a single `State`
type and realizes a merged `Extras` schema from taskset and harness config, so
shared rollout mutation has one predictable target.

### Tool Data Flow

v1 has explicit tool/user data-flow metadata on the decorated method:

```python
@vf.tool(
    args={"case": "state.metadata.case"},
    extends={"events": "state.extras.events"},
    sets={"last": "state.extras.last"},
)
def search(self, query: str, case: str) -> dict: ...
```

Bound args are hidden from model-visible schemas. `sets` replaces one state path;
`extends` appends a returned list. Same-path extends in a parallel batch are
allowed; overlapping sets conflict.

Nano is simpler: a taskset returns `ToolServer` objects and rollout/tool
placement is controlled by `taskset.tools`. Servers are script, command, or URL
MCP endpoints, and the agent receives a map of MCP URLs. Nano does not model
hidden state reads or bound state writes at the framework layer.

### Users As MCP Servers

v1 treats users as MCP servers with a hidden `respond` tool. User and tool
servers both return `ServerResponse` data with `messages`. Toolsets write state
through `@vf.tool` data-flow metadata; users use the sibling `@vf.user`
decorator.

`ToolsetConfig` and `UserConfig` are sibling `ServerConfig` specializations
with the same server contract:

```python
class SearchToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "rollout"


class DialogueUserConfig(vf.UserConfig):
    pass
```

The template convention is one folder per configured server, for example
`servers/search/config.py` plus `servers/search/toolset.py`, and
`servers/user/config.py` plus `servers/user/user.py`. Authors write `Toolset`
and `User` classes; v1 owns the MCP lifecycle behind that contract.

`TasksetConfig.toolsets` is a mapping from toolset name to config. The key is
the model-visible tool prefix. Taskset-defined keys can be overridden in TOML
without `source`; new keys require `source` pointing to a `ToolsetConfig` class.
The tool implementation is derived from that config class.

Nano examples use MCP servers for tools. User simulation is not yet the same
first-class server contract in nano.

### Group Rewards And Advantages

v1 has group scoring and v1-only advantage functions (`rl`, `grpo`, `rloo`,
`reinforce`, `sft`) that mutate token advantages in place. `State` does not
store a scalar advantage. The env default is `advantage="rl"`.

Nano's `Episode` supports cross-rollout group rewards, but deliberately leaves
advantage computation to trainers above the environment layer.

## Runtime And Context Differences

Nano has explicit `Rollout` and `Episode` objects:

- `Rollout` creates the runtime, runs the agent, and scores per-rollout
  rewards/metrics while that runtime is still live.
- `Episode` runs all rollouts for one task, guarantees runtime teardown, and
  then runs `Taskset.score_group(...)` over the resulting traces.

v1 currently has:

- `Harness.run(...)` creates the live `Context`.
- `Context` carries task, state, model client, teacher, runtime, tools, user,
  parent, and scoring flags.
- `RuntimeProvider.create_runtime()` materializes the live `Runtime`.
- `Task.image` and `Task.resources` are serializable task-level runtime
  requests. Harness runtime config wins when it explicitly sets the same field.
- `Env.score_group(...)` scores after rollout runtimes are closed.

This is the largest remaining architectural difference. v1 kept the Verifiers
`State`/`Env.score_group` contract and did not introduce a public `Episode`
object. Nano is now cleaner about ownership because rollout lifetime and
cross-rollout scoring are separate objects. Supporting runtime-backed group
scoring in either design would require either:

- adding an internal episode-like owner for grouped rollout lifetimes, or
- making `Env.score_group(...)` accept live contexts/runtimes and own their
  teardown.

The first option is closer to nano and cleaner for resource ownership. The
second keeps the v1 public surface smaller but makes group runtime lifetime more
implicit.

### Latest nano runtime/Harbor checkpoint

The latest nano Harbor example clarified three runtime contracts that v1 now
matches:

- Harbor tasks with `environment/Dockerfile` but no pullable
  `[environment].docker_image` fail during task loading. Verifiers does not
  build Harbor Dockerfiles, and silently falling back to the default image
  scores the wrong environment.
- Prime runtime file writes use the sandbox upload gateway and resolve relative
  paths against the runtime workdir. Large Harbor test bundles must not be
  base64-inlined into an exec command.
- Prime runtime `public_url(...)` exposes sandbox ports natively, which lets
  runtime-placed MCP servers be reached from other runtimes without a host-local
  URL leak.

Nano also strengthened `run_uv_script(...)` for bare task images. v1 does not
mirror that helper as a runtime contract. Script execution is just one caller
pattern built from `write(...)` plus `run(...)`; keeping it out of `Runtime`
avoids a parallel framework path for dependency and program launch semantics.

Nano's current eval runner added durable incremental `results.jsonl`, resolved
`config.toml`, deterministic shuffle, and a rich dashboard that reads live
rollout phase/runtime descriptors. Those are eval/UI improvements rather than
core v1 authoring contracts. They are worth adopting in the eval layer, but not
as additional v1 state fields or helpers.

## Scoring Contract

v1 direct harness runs default to generation only:

```python
state = await harness.run(task="hello", model="openai/gpt-5", score=False)
```

`Env.run_rollout(...)` calls `Harness.run(..., score=True)`. Nested judge or
self-check calls should pass the parent `Context` and keep `score=False`.
Requesting `score=True` inside an active scoring context raises.

Rewards are taskset-owned. The selected advantage function is env-owned and
mutates turn token advantages in place. Harnesses may define metrics for
execution telemetry, but harness rewards/advantages are rejected because scoring
semantics belong to the task.

Nano scores inside `Rollout.run()` and group-scores inside `Episode.run()`. v1
keeps scoring optional on direct `Harness.run(...)` because reward functions may
call a harness recursively for judging.

## Branching And Compaction

Nano records a flat `trajectory` but exposes `trace.branches` and
`trace.num_branches`. Branches are inferred from token prefixes when token data
is available, otherwise message prefixes. The message fallback ignores
`reasoning_content` and treats `None` and empty assistant content as equal, so
tool-call-only assistant messages do not split branches accidentally.

v1 currently records `state.transcript` as a flat list of `Turn`s and leaves
branch/compaction interpretation to the harness or downstream consumer. If v1
wants compacting agents or subagent transcript visualization as a first-class
contract, nano's branch view is the better reference than reintroducing
trajectory aliases.

## Server Response Differences

v1 server calls produce:

```python
class ServerResponse(vf.Config):
    content: vf.MessageContent | None = None
    messages: vf.Messages = []
```

The default harness converts a single text tool response into a protocol
`ToolMessage`. Explicit multi-message responses are appended after tool results,
which supports compaction and richer server-controlled conversation updates.
Hidden tools are excluded from model-visible schemas and from public tool-call
dispatch; the harness calls them through the hidden-call path.

Nano's tool serving is oriented around exposing MCP URLs to the agent program.
It does not currently model bound state writes or multi-message server
responses as a framework-level contract.

## Compatibility Position

v1 is not trying to preserve old v1 APIs. The compatibility boundary is:

- v0 remains top-level `verifiers`.
- v1 code imports `verifiers.v1 as vf`.
- `MultiTurnEnv` remains on the legacy path.
- v1 examples/packages should teach only the current v1 pattern.

## Current Tensions

- Whether v1 should introduce an internal `Episode` to own grouped rollout
  lifetimes and make `Env.score_group(...)` a smaller adapter.
- Whether group-scoped toolsets should become first-class, or whether
  `state.group_id` plus env-scoped toolsets is enough for the first v1 release.
- Whether `ServerResponse.messages` should become the only visible server return
  channel, or whether advanced tool-call protocols need a stricter message
  contract.
- Whether `Context` should be passed into every handler/signal by default or
  remain opt-in by parameter name.
- Whether v1 advantages belong in Verifiers long term or should move up into
  trainers like nano suggests.
- Whether v1 should adopt a branch view over `transcript` for compaction and
  subagent runs, without renaming back to `trajectory`.
