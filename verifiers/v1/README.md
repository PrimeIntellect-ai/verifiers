# Verifiers v1

v1 is an active-development rewrite of the Taskset/Harness stack. Breaking
changes are expected before release. Import it as:

```python
import verifiers.v1 as vf
```

The top-level `verifiers` package is the v0 surface. v1 code should not import
v1 classes from top-level `verifiers`, and v0 code should not rely on
`verifiers.v1` internals.

## Model

- `Taskset` owns tasks, task prompts, task tools, user simulation, metrics,
  rewards, and task-specific lifecycle.
- `Env` owns the selected group advantage function. The default is `"rl"`;
  pass `advantage=None` to disable environment-provided token advantages.
- `EnvRun` owns one environment execution: env-scope toolsets/users,
  per-rollout runtime creation, and grouped rollout coordination. Eval creates
  one `EnvRun` for the evaluation. Direct `Env.run_rollout(...)` is a one-shot
  convenience around `EnvRun`.
- `Group` owns the tasks and states for one grouped example and calls
  `env.score_group(...)` after its member rollouts finish.
- `Harness` is the agent. Its `run(...)` method starts a standalone `EnvRun`
  when no parent `Context` is supplied; nested calls reuse the parent
  `Context`. Direct `Harness.run(...)` defaults to `score=False`;
  `Env.run_rollout(...)` opts into rollout scoring.
- `Context` is the live per-harness execution record: task, state, runtime,
  model/teacher clients, toolsets, user, parent context, and scoring flags.
- `Env` is the thin adapter that pairs taskset and harness, opens `EnvRun`
  contexts, scores groups, and serializes output.
- `State` is the canonical rollout record. It is a strict Pydantic model with
  `transcript: list[Turn]`; there is no live `trajectory` alias.
- `state.messages` is a convenience rendering of the latest conversation
  prompt plus completion. `state.transcript` remains the canonical record for
  per-request history.
- `state.extras` is the user-owned mutable rollout data surface. Taskset and
  harness configs may provide typed `vf.Extras` defaults; v1 realizes one schema
  from both and rejects duplicate keys.

Runtime handles, model clients, MCP sessions, and server connections are never
stored in `Task` or `State`.

## Runtime And Protocols

Runtime providers expose one live `Runtime` contract: `start`, `stop`, `expose`,
`run`, `read`, and `write`. The built-in configs are `subprocess`, `docker`,
and `prime`; `modal` and `daytona` are reserved provider stubs.
Task rows may set `image` and `resources` for serializable per-task runtime
selection. Runtime config wins over task resources when the config field is set
away from its provider default. Live runtimes stay owned by the harness
lifecycle.

Harnesses that run external agents start an `InterceptionServer` and expose it
through the active runtime. Built-in endpoint protocols cover OpenAI
chat completions, OpenAI completions, OpenAI responses, and Anthropic messages.
Custom protocols are harness-side adapters: override `Harness.load_protocols()`
and return `EndpointProtocol` objects with `routes`, `env(...)`, `parse(...)`,
and `serialize(...)`. Protocols may execute Python on the loaded harness side,
but they must exchange only JSON request/response data and must not write live
handles into `Task` or `State`.

## Tools And Users

Toolsets are declared with `ToolsetConfig` and implemented as `Toolset`
subclasses:

```python
class SearchToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "rollout"


class SearchToolset(vf.Toolset):
    @vf.tool(
        args={"query_context": "state.extras.query_context"},
        extends={"events": "state.extras.search_events"},
    )
    def search(self, query: str, query_context: str) -> dict:
        ...


class SearchTasksetConfig(vf.TasksetConfig):
    toolsets: vf.ToolsetConfigs = {"wiki": SearchToolsetConfig()}
```

The `toolsets` key is the model-visible tool prefix. Config may override a
taskset-defined toolset by key without repeating its source, and may add a new
toolset by pointing `source` at a `ToolsetConfig` class.

One `Toolset` may expose multiple tools. Supported scopes are:

- `rollout`: started for one rollout and cleaned up afterward.
- `env`: started once for an `EnvRun` and reused by all rollouts in that run.

Supported placements are:

- `dedicated`: start the toolset/user in its own runtime.
- `colocated`: start the toolset/user in the owning rollout runtime.
- `remote`: connect to an existing URL.

`@vf.tool(args=..., sets=..., extends=...)` is the only framework wiring path
for hidden args and state writes. Bound args are hidden from the model and
injected from serialized `task.*`, `state.*`, `extras.*`, and server-local
`resources.*` paths. `sets` replaces one `state.*` or `extras.*` path;
`extends` appends a returned list to one `state.*` or `extras.*` list path.
Multiple same-path extends in one tool-call batch are allowed, with no ordering
guarantee.

Users use the sibling `UserConfig` / `User` path over the same server base. A
user exposes a hidden `respond` tool and returns `messages`. Toolsets use the
same response shape; the default harness converts single text tool responses
into protocol `tool` messages and appends explicit multi-message responses
after tool results. Hidden tools are callable only by the harness through the
hidden-call path, not by model-visible tool calls.

## Authoring Pattern

The default v1 package layout is component-first:

```text
my_env/
  my_env/
    taskset.py
    harness.py        # optional
    servers/
      search/
        config.py
        toolset.py
      user/
        config.py
        user.py
```

`vf.load_environment("my-env")` imports the package, discovers `taskset.py` and
optional `harness.py`, and constructs `vf.Env` internally.

```python
import verifiers.v1 as vf
from pydantic import BaseModel


class MyTask(vf.Task):
    answer: str


class MyDetails(BaseModel, extra="forbid"):
    source: str


class MyTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Say exactly what is requested."


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    task_type = MyTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return [{"prompt": [{"role": "user", "content": "Say ok."}], "answer": "ok"}]

    @vf.reward
    async def exact(self, task: MyTask, state: vf.State) -> float:
        message = state.completion[-1]
        return float(str(message.content).strip() == task.answer)


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)
```

Config is serializable policy. Live Python functions are allowed as decorated
methods on loaded `Taskset`/`Harness` objects, not as config values, task fields,
state fields, tool definitions, or runtime specs.

Use ordinary Pydantic models for strict nested task/config records. The v1
library keeps its own types to framework contracts; example-specific nesting is
userspace schema.

## Current Tensions

- Group rewards and token-level advantages are first-class, and v1 envs default
  to the built-in `"rl"` advantage. Group scoring
  currently runs after per-rollout runtimes close. Supporting runtime-backed
  group scoring would require an explicit group runtime lifetime.
- Env-scope toolsets are first-class. Group-specific resources should use
  `state.group_id` plus env-scope toolset state rather than a third tool scope.
- The base harness is model-loop native. Command/program agents should be
  implemented as `Harness` subclasses that use `Runtime`, not as generic
  callable config.
