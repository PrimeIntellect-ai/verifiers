# Verifiers v1

`verifiers.v1` is the next-pass environment stack. It keeps the public authoring
model close to functions and config, while the framework owns endpoint
forwarding, trajectory capture, tool resolution, sandbox lifetime, and
rollout/group scoring.

The stable boundary is:

- `Task`: immutable JSON-serializable dataset row.
- `State`: mutable JSON-serializable rollout record.
- `Taskset`: task source plus task-owned tools, user behavior, metrics, rewards,
  stop conditions, and cleanup.
- `Harness`: endpoint-backed runner plus harness-owned tools, user behavior,
  stop conditions, metrics, rewards, cleanup, and teardown.
- `Env`: `vf.Environment` adapter for eval/training workers.

## Opinionated Choices

- Use `Taskset + Harness -> Env` for eval/training. Do not subclass `Env` for
  normal environment authoring.
- Use `harness.run(task, state=None)` for standalone experiments. A harness
  should be runnable without a taskset.
- Put immutable per-example data in `Task`; put mutable rollout data in
  `State`. Both must remain JSON-serializable at API boundaries.
- Put pluggable logic in functions: programs, tools, users, metrics, rewards,
  cleanup, and teardown.
- Use configs for selection and tuning. Code may pass functions directly; TOML
  uses `"module:object"` refs.
- Subclass `Taskset` or `Harness` only to define a new config surface or
  behavior that cannot be represented by functions/config.
- Package tools as `Toolset`. A callable tool can be wrapped implicitly, but
  stateful tools, sandboxed tools, MCP tools, and lifecycle hooks belong in a
  toolset.
- Keep parser-like helpers as ordinary Python objects. Pass them through
  closures, toolset objects, user objects, or config refs.
- Treat metrics and rewards as signals. Rollout signals run inside
  `harness.run`; group signals run from `Env` group scoring.
- Treat `done` / `is_completed` as the generic finish-tool escape hatch. Custom
  stop functions should normally be contributed by the taskset, harness, or a
  toolset.
- Use cleanup for user-extensible end-of-rollout or end-of-group state/resource
  finalization. Framework-owned timing, completion rendering, trajectory sync,
  and sandbox destruction are automatic.
- Treat channel-like behavior as promoted config sections. `toolsets`, `user`,
  `sandbox`, `metrics`, `rewards`, `scoring`, `cleanup`, and `teardown` are the
  supported compatibility/resolution surfaces. New cross-cutting surfaces start
  as fields on a `TasksetConfig` or `HarnessConfig` subclass.

## Minimal Shape

```python
import verifiers.v1 as vf


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state["answer"] == task["answer"])


def source():
    yield {
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
    }


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[exact_answer], config=config)


def load_harness(config=None):
    return vf.Harness(config=config)


def load_environment(taskset_config=None, harness_config=None):
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=load_harness(harness_config),
)
```

`Taskset.get_dataset()` returns rows compatible with the current
`vf.Environment` worker schema. The full task row is stored as a JSON string in
`info["task"]`, and `Taskset.to_task(...)` accepts a `Task`, mapping, or JSON
string.

## Harness Programs

`Harness.run(task, state=None)` is the primary local entrypoint. It initializes
state when needed, runs the program through the interception endpoint, syncs
trajectory-derived state fields, runs rollout metrics/rewards, runs rollout
cleanup, and validates that the returned state is serializable.

The supported program forms are:

- `program=None`: default tool loop using the selected client protocol.
- `program=callable`: Python function called through the interception endpoint.
- `program={"command": [...], ...}`: local or sandboxed command.
- `program={"base": true, ...}`: explicit default loop, useful when the default
  loop needs program-level options such as sandbox execution.

Mapping programs must specify exactly one kind: `base=true`, `entrypoint`, or
`command`. If a mapping only contains program options such as `sandbox`, `files`,
`dirs`, `setup`, `env`, or `artifacts`, it resolves to the base loop. Use
`program=None` for the ordinary local base loop.

Program placement is explicit. A program runs locally/in-process unless its
mapping contains `sandbox = true` or an inline sandbox mapping. `Harness.sandbox`
is the default primary sandbox config; it does not by itself move the program.
This lets harnesses use sandbox-backed tools while keeping the main program
local.

```python
# Local command.
vf.Harness(program={"command": ["python", "run.py"]})

# Sandboxed command using the harness sandbox config.
vf.Harness(
    sandbox={"image": "python:3.11-slim"},
    program={"sandbox": True, "command": ["python", "run.py"]},
)

# Sandboxed default loop.
vf.Harness(
    sandbox={"image": "python:3.11-slim"},
    program={"sandbox": True},
)

# Sandboxed importable Python program.
vf.Harness(
    program={
        "entrypoint": "my_env.program:run",
        "sandbox": {"image": "python:3.11-slim"},
    }
)
```

Sandboxed programs support `env`, `files`, `dirs`, `setup`, and `artifacts`.
`files` and `dirs` map sandbox paths to literal values, `task.*` / `state.*`
paths, runtime paths, or callables. `setup` runs after uploads and before the
main program. `artifacts` reads text or JSON from sandbox paths or globs into
`state["artifacts"]`.

The sandboxed base loop uses the rollout endpoint for model calls and a sibling
tool proxy for resolved callable/MCP tools, so the tool registry still lives in
the host runtime while the loop itself runs in the sandbox.

The default loop is controlled by `HarnessConfig.max_turns`.

The selected client protocol is carried in runtime state. OpenAI Chat
Completions is the default; Anthropic Messages and OpenAI Responses are selected
through `ClientConfig.client_type` / endpoint registry `type`.

Third-party Python programs should configure their own libraries against the
interception endpoint passed in state. For example, a DSPy entrypoint can build
`dspy.LM(..., api_base=state["endpoint_base_url"], api_key="intercepted")` and
use `dspy.context(lm=...)` inside the program call. Avoid global library
configuration in async entrypoints.

Standalone harnesses can receive model controls directly:

```python
harness = vf.Harness(
    client=vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url="https://api.openai.com/v1",
        api_key_var="OPENAI_API_KEY",
    ),
    model="gpt-5.4-mini",
)

state = await harness.run(vf.Task({"prompt": [{"role": "user", "content": "2+2?"}]}))
```

When a harness is used through `Env`, the eval/training worker supplies the
client, model, and sampling args for each rollout/group invocation.

## Tools

Tools are packaged with `Toolset`:

```python
async def search(query: str, index) -> str:
    ...


toolset = vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)
```

`Toolset.tools` accepts callable tools, nested `Toolset` objects, and
`vf.MCPTool(command=..., args=[...])` stdio servers. Toolsets expose all tools
by default; `show=[...]` or `hide=[...]` can narrow the exposed surface.
In config/TOML, MCP tools can be written as command specs inside `tools`.

Hidden bindings can read from `task`, `state`, runtime `objects`, or other
tools. Programs that need callable tools use:

```python
from verifiers.v1.utils.tool_utils import load_tools_from_state


async def program(task, state):
    tools = load_tools_from_state(state)
    result = await tools["search"](query=task["question"])
    state["answer"] = result
    return state
```

Toolsets can declare `sandbox={...}` for tools that need isolated execution.
Sandbox scope can be `rollout`, `group`, or `global`. A toolset can also use
`sandbox="program"` when its tools should operate against the primary program
sandbox for the current rollout/group.

Lazy `objects` are scoped as well. Read-only toolsets default to global objects;
`write=True` toolsets default to rollout-scoped objects unless `scope` is set.

Tasks can narrow the exposed tools for a rollout by setting
`task["runtime"]["tools"] = ["tool_name", ...]`. They can also use
`{"show": [...]}` or `{"hide": [...]}`. Unknown tool names hard fail.

Programs running through the interception endpoint can discover and call the
same resolved tools over HTTP:

- `GET {state["endpoint_root_url"]}/vf/tools` returns provider-agnostic tool
  schemas.
- `GET {state["endpoint_root_url"]}/vf/tools?protocol=openai_chat_completions`
  returns OpenAI Chat Completions tool payloads. `openai_responses` and
  `anthropic_messages` are also supported.
- `POST {state["endpoint_root_url"]}/vf/tools/{name}` with
  `{"arguments": {...}}` calls the tool.
- command and sandbox programs receive `VF_TOOLS_JSON`, `VF_TOOL_BASE_URL`,
  `VF_TOOL_API_KEY`, and `VF_ENDPOINT_API_KEY`. `VF_TOOLS_JSON` matches the
  selected client protocol; `VF_TOOL_DEFS_JSON` preserves the provider-agnostic
  schema.
- harness builders choose how programs accept tools with `tool_protocol`.
  `tool_protocol="callable"` is the default for Python programs and the base
  tool loop. `tool_protocol="mcp"` exposes the same resolved tools through an
  official stdio MCP proxy, with `VF_MCP_TOOL_COMMAND_JSON` and
  `VF_MCP_TOOL_COMMAND` available to command programs.

MCP tools are normalized into callable tool handles before programs see them.
Callable tools can also be presented as MCP tools when the harness selects MCP.

## Nested Harnesses

A tool or program can launch a child harness through the active runtime:

```python
async def ask_child(prompt: str, harness, state):
    task = vf.Task({"prompt": prompt}).freeze()
    child_state = await vf.current_runtime().run_harness(
        harness,
        task,
        parent_state=state,
    )
    return child_state["answer"]
```

The child rollout receives its own `trajectory_id` and rollout-local state. It
inherits the parent group key, model, sampling args, and model client unless the
child state overrides them. The parent should store only serializable child
outputs in its own state.

## Users

Tasksets and harnesses may define a `User`, or pass a user callable directly.
The callable receives `task`, `state`, and configured bindings. `transcript` is
available by default and resolves to the current observable conversation.

```python
async def user(task, state, transcript):
    if len([m for m in transcript if m["role"] == "assistant"]) >= 2:
        return []
    return [{"role": "user", "content": "Try one more time."}]


taskset = vf.Taskset(source=source, user=user)
```

User objects can use `scope="rollout"`, `"group"`, or `"global"` for stateful
helpers created through `User(objects={...})`.

Users can also request sandboxes with the same scope vocabulary as tools:

```python
async def user(task, state, sandbox, transcript):
    result = await sandbox.execute("python /tmp/check.py")
    if result.exit_code:
        return [{"role": "user", "content": "Check your work again."}]
    return []


taskset = vf.Taskset(
    user=vf.User(
        user,
        scope="group",
        sandbox={"image": "python:3.11-slim", "scope": "group"},
    )
)
```

## Signals And Cleanup

Metrics and rewards are signal functions.

```python
@vf.metric
async def turns(task, state) -> float: ...


@vf.reward(weight=0.5, priority=10)
async def format_reward(task, state) -> float: ...


@vf.reward(stage="group")
async def best_of_n(tasks, states) -> list[float]: ...


@vf.advantage
async def center_advantages(tasks, states) -> list[float]: ...
```

Rollout signals accept exactly `task, state`. Group signals opt in with
`stage="group"` and accept exactly `tasks, states`. `@vf.advantage` is always a
group signal. If no advantage signal is configured, group scoring uses
`reward - mean(group_reward)` and writes the value to each state and trajectory
step with an unset advantage.

Cleanup functions use `@vf.cleanup`; teardown functions use `@vf.teardown`.
There is no public render decorator. State rendering that the framework owns,
such as timing, completion, and trajectory sync, is part of the harness run
contract.

Stop functions use `@vf.stop` or `stop=[...]` on tasksets, harnesses, or
toolsets:

```python
@vf.stop
async def submitted(task, state) -> bool:
    return bool(state.get("submitted"))


toolset = vf.Toolset(tools=[submit], stop=[submitted])
```

## Config

Tasksets and harnesses expose their constructor extension surface through
Pydantic configs. Code can pass callables and objects directly; TOML passes
`"module:object"` refs.

```python
taskset = vf.Taskset(
    config={
        "source": "my_env.data:load_rows",
        "eval_source": "my_env.data:load_eval_rows",
        "rewards": ["my_env.signals:exact_answer"],
        "toolsets": [{"tools": ["my_env.tools:search"]}],
    }
)

harness = vf.Harness(
    config={
        "program": "my_env.program:run",
        "max_turns": 20,
        "cleanup": ["my_env.cleanup:summarize_rollout"],
    }
)
```

Direct constructor items extend list-like config items. Scalar constructor args
such as `source`, `program`, `sandbox`, `user`, and `max_turns` override config.

`scoring` tunes named signal functions:

```toml
[env.taskset.scoring.exact_answer]
weight = 0.5

[env.harness.scoring.turns]
skip = true
```

Subclass only when an environment needs a new config surface or new behavior.
Set `config_type` to a `TasksetConfig` or `HarnessConfig` subclass:

```python
class MyHarnessConfig(vf.HarnessConfig):
    cache_dir: str | None = None


class MyHarness(vf.Harness):
    config_type = MyHarnessConfig
```

Dicts are accepted by constructors and validated through the same models. To
inspect the active shape:

```python
print(vf.TasksetConfig.schema_text())
print(vf.HarnessConfig.schema_text())
print(MyHarness.config_schema())
```

### Custom Config Surfaces

Use a config subclass for taskset/harness-specific surfaces that do not belong
in the promoted v1 fields yet:

```python
class WikiTasksetConfig(vf.TasksetConfig):
    db_path: str


class WikiTaskset(vf.Taskset):
    config_type = WikiTasksetConfig

    def __init__(self, config):
        config = self.config_type.model_validate(config)
        super().__init__(
            source=load_wiki_rows,
            toolsets=[
                vf.Toolset(
                    tools=[search],
                    objects={"db": lambda: open_wiki_db(config.db_path)},
                    bindings={"search.db": "objects.db"},
                )
            ],
            config=config,
        )
```

If a taskset and harness both need to negotiate a new kind of object, prefer an
explicit promoted field once there are multiple concrete users. Until then,
keep it local to the config subclass and use `objects`/`bindings` to route the
resolved object into functions.

There is no public generic channel registry in v1. A custom `db` surface, for
example, should be a typed config field owned by the taskset or harness that
needs it:

```python
class DbTasksetConfig(vf.TasksetConfig):
    db_path: str


class DbTaskset(vf.Taskset):
    config_type = DbTasksetConfig

    def __init__(self, config):
        config = self.config_type.model_validate(config)
        super().__init__(
            source=load_rows,
            toolsets=[
                vf.Toolset(
                    tools=[search],
                    objects={"db": lambda: open_db(config.db_path)},
                    bindings={"search.db": "objects.db"},
                )
            ],
            config=config,
        )
```

If several tasksets and harnesses start repeating the same `db` shape, promote
that shape into a first-class config field rather than adding an ad hoc
parallel resolver.

## Mini Tutorials

### Add A Reward In TOML

Define the function in Python:

```python
@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))
```

Expose it from config:

```toml
[env.taskset]
rewards = ["my_env.signals:exact_answer"]

[env.taskset.scoring.exact_answer]
weight = 0.5
```

### Add A Tool With Hidden Runtime State

```python
async def search(query: str, index) -> str:
    return index.search(query)


toolset = vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)
```

The model only sees `query`; the runtime injects `index`.

### Limit Tools Per Task

```python
{
    "prompt": [{"role": "user", "content": "Use only read access."}],
    "runtime": {"tools": ["read_file"]},
}
```

The runtime hard-fails if a task requests an unknown tool.

### Use A Sandboxed Tool

```python
async def python(code: str, sandbox) -> str:
    result = await sandbox.execute(f"python - <<'PY'\n{code}\nPY")
    return result.stdout


toolset = vf.Toolset(
    tools=[python],
    write=True,
    sandbox={
        "image": "python:3.11-slim",
        "scope": "group",
        "packages": ["numpy"],
    },
)
```

The sandbox handle is hidden from the model and recorded in
`state["runtime"]["sandboxes"]`.

### Use The Program Sandbox From A Tool

```python
async def inspect_workspace(command: str, sandbox) -> str:
    result = await sandbox.execute(command)
    return result.stdout


harness = vf.Harness(
    sandbox={"image": "python:3.11-slim", "scope": "group"},
    program={"sandbox": True},
    toolsets=[
        vf.Toolset(
            tools=[inspect_workspace],
            sandbox="program",
            write=True,
        )
    ],
)
```

`sandbox="program"` hard-fails if the program is not currently running with a
primary sandbox. This keeps same-sandbox coupling explicit.

### Configure A Harness Program

```toml
[env.harness]
program = "my_env.programs:run"
max_turns = 20
```

```python
async def run(task, state):
    state["answer"] = "..."
    return state
```

`program=None` uses the default endpoint-backed tool loop.

### Call A Nested Harness

```python
async def solve_subtask(prompt: str, state):
    child = vf.Harness(program="my_env.child:run")
    task = vf.Task({"prompt": [{"role": "user", "content": prompt}]}).freeze()
    child_state = await vf.current_runtime().run_harness(
        child,
        task,
        parent_state=state,
    )
    return child_state["answer"]
```

The child rollout gets its own `trajectory_id` and rollout-scoped resources. It
inherits the parent `group_key`, model, sampling args, and model client unless
the child state overrides them.

## FAQ

### Where Do Model And Client Settings Live?

Eval/training runners pass `client`, `model`, and `sampling_args` into the
`Env` boundary. v1 stores the serializable parts in `state["runtime"]` and keeps
live clients inside the harness runtime.

Standalone harnesses may pass `client=...`, `model=...`, and
`sampling_args=...` at construction. State runtime values take precedence for a
specific rollout.

### How Do I Use Multiple Models?

Use runtime state to select a client key/model for a particular rollout, and
bind the corresponding client through the harness runtime. Calls that should not
be optimized can be tagged in state or trajectory extras by the program that
submits them.

### When Should A Signal Be Group-Scoped?

Use rollout signals for anything computable from one `task, state`. Use
`stage="group"` only when the function needs multiple attempts at once, such as
best-of-N preference, relative advantage, or a group-level judge.

### When Should A Sandbox Be Rollout, Group, Or Global?

Use `rollout` when each attempt must be isolated. Use `group` when scoring or
comparison may need artifacts after rollout completion. Use `global` for
read-only or reusable services that are safe to share for the harness lifetime.
Primary program sandboxes, tool sandboxes, and user sandboxes all use the same
scoped lease cleanup path.

### How Do MCP Tools Fit?

Use `vf.MCPTool(command=..., args=[...])` inside a `Toolset`. The runtime turns
MCP server tools into normal tool handles for the harness. Callable tools and
MCP tools share the same visibility and per-task tool filtering rules.

```toml
[[env.harness.toolsets]]
tools = [
  { command = "uvx", args = ["mcp-server-fetch"] },
]
```

### How Do I Keep A Harness Lightweight?

Prefer `Harness(program=..., toolsets=..., user=..., metrics=..., rewards=...)`
and config refs. Subclass only when the harness needs a named reusable config
surface or a different `setup_state`/program behavior.

### How Do I Share Expensive Objects?

Use lazy `objects` on `Toolset` or `User`. Read-only objects default to global
toolset scope; writable toolsets default to rollout scope. Set `scope`
explicitly when group or global lifetime is intended.

## Examples

The active v1 ports live beside their existing environments:

- `environments/reverse_text/reverse_text_v1.py`
- `environments/alphabet_sort/alphabet_sort_v1.py`
- `environments/wiki_search/wiki_search_v1.py`
- `environments/math_python/math_python_v1.py`
- `environments/mcp_search_env/mcp_search_v1.py`
- `environments/opencode_harbor/opencode_harbor_v1.py`
- `environments/tau2_bench/tau2_bench.py`
