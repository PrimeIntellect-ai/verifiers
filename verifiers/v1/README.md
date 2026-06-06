# Verifiers v1

v1 is an active-development rewrite of the Taskset/Harness stack. Import it as:

```python
import verifiers.v1 as vf
```

The top-level `verifiers` package is the v0 surface. v1 code should not import
v1 classes from top-level `verifiers`, and v0 code should not rely on
`verifiers.v1` internals.

## Model

- `Taskset` owns tasks, task prompts, task tools, user simulation, metrics,
  rewards, advantages, and task-specific lifecycle.
- `Harness` is the agent. Its `run(...)` method owns the rollout lifecycle:
  runtime session, MCP connections, setup/generation/update/scoring/cleanup, and
  finalization.
- `Env` is the thin adapter that creates `State`, delegates rollouts to the
  harness, scores groups, and serializes output.
- `State` is the canonical rollout record. It is a strict Pydantic model with
  `transcript: list[Turn]`; there is no live `trajectory` alias.
- `state.scratch` is the only user-owned mutable rollout scratchpad.

Runtime handles, model clients, MCP sessions, and server connections are never
stored in `Task` or `State`.

## Runtime And Protocols

Runtime providers expose one session contract: `start`, `stop`, `expose`, `run`,
`read`, and `write`. The built-in configs are `local`, `docker`, and `prime`.
Task rows may set `image` for serializable per-task runtime image selection;
live sessions stay owned by the harness lifecycle.

Harnesses that run external agents start an `InterceptionServer` and expose it
through the active runtime session. Built-in endpoint protocols cover OpenAI
chat completions, OpenAI completions, OpenAI responses, and Anthropic messages.
Custom protocols are harness-side adapters: override `Harness.load_protocols()`
and return `EndpointProtocol` objects with `routes`, `env(...)`, `parse(...)`,
and `serialize(...)`. Protocols may execute Python on the loaded harness side,
but they must exchange only JSON request/response data and must not write live
handles into `Task` or `State`.

## Tools And Users

Toolsets are MCP servers:

```python
vf.Toolset(
    name="wiki",
    server=vf.MCPServerSpec(command=["python", "-m", "my_env.wiki_server"]),
    scope="rollout",
)
```

One `Toolset` is one server that may expose multiple tools. Supported scopes are:

- `rollout`: started for one rollout and cleaned up afterward.
- `env`: started once and reused across rollout lifetime.

Users are MCP servers too. A user server exposes a hidden `respond` tool and
returns a serialized `vf.User.TurnResult`; the harness calls it
between assistant turns when present.

## Authoring Pattern

The default v1 package layout is component-first:

```text
my_env/
  my_env/
    taskset.py
    harness.py        # optional
    servers/
      user.py         # optional
      tools.py        # optional
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

- Group rewards and advantages are first-class, but group scoring currently runs
  after per-rollout runtime sessions close. Supporting runtime-backed group
  scoring would require an explicit group runtime lifetime.
- Env-scope MCP servers are first-class. Group-specific resources should use
  `state.group_id` plus env-scope server state rather than a third tool scope.
- The base harness is model-loop native. Command/program agents should be
  implemented as `Harness` subclasses that use `RuntimeSession`, not as generic
  callable config.
