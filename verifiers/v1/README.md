# Verifiers v1

`verifiers.v1` is the composable environment API for building eval and training
environments from two primary objects:

- a `Taskset`, which defines what is being attempted;
- a `Harness`, which defines how a model attempts it.

`vf.Env(taskset, harness)` adapts those objects to the existing
`vf.Environment` worker API used by evals and trainers. For local experiments,
`Harness` is runnable on its own with `await harness.run(task)`.

The programming model is intentionally small: tasks and state are serializable
data; everything else is a function, config value, or runtime-managed handle.

## Mental Model

The v1 boundary is data-first:

- `Task` is immutable input data.
- `State` is mutable output data.
- `Taskset` packages tasks and task-owned behavior.
- `Harness` packages rollout behavior.
- `Env` adapts a taskset/harness pair to eval and training workers.

Runtime objects are deliberately hidden from user-facing signatures. Tools,
MCP sessions, sandboxes, model clients, and local service handles are resolved
behind the scenes and reached through state helpers while a rollout is active.

Extension is function-and-config based. Add a metric, reward, cleanup, toolset,
or user by passing a function/object directly, referencing it from config, or
defining a shallow subclass when a package needs a typed config surface.

## Core Objects

### `Task`

A `Task` is an immutable, JSON-serializable dataset row. It is the canonical
place for per-example data such as prompts, answers, metadata, tool filters, and
sandbox overrides. A task is frozen before rollout code sees it.

```python
task = vf.Task(
    {
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
    }
).freeze()
```

### `State`

A `State` is a mutable, JSON-serializable rollout record. It starts from a task,
then accumulates trajectory, completion, metrics, reward, timing, artifacts,
errors, and any extra fields the environment author chooses to expose.

```python
state = vf.State.for_task(task)
state["answer"] = "cba"
state.assert_serializable()
```

During an active rollout, state may also carry runtime metadata used to find
live in-process handles. Those runtime handles are stripped before state is
returned from a standalone rollout or completed `Env` group.

### `Taskset`

A `Taskset` provides task rows and task-owned logic:

- source and eval source;
- task-owned tools/toolsets;
- user behavior;
- stop conditions;
- metrics, rewards, advantages;
- cleanup.

Taskset rows are lazy-loaded and cached after first access.

### `Harness`

A `Harness` runs a task. It owns runtime resolution for model endpoints, tools,
MCP sessions, sandboxes, nested harness calls, trajectory capture, and lifecycle
execution.

The default harness is endpoint-backed: model calls go through a local
interception endpoint so trajectory capture and tool forwarding use one path
across local Python programs, command programs, and sandboxed programs.

### `Env`

`Env` is the eval/training adapter:

```python
env = vf.Env(taskset=taskset, harness=harness)
```

Normal v1 environment packages should not subclass `Env`. Define or configure a
taskset and harness, then compose them.

## Minimal Environment

```python
import verifiers.v1 as vf


def source():
    yield {
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
    }


@vf.reward(weight=1.0)
async def contains_answer(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[contains_answer], config=config)


def load_harness(config=None):
    return vf.Harness(config=config)


def load_environment(taskset_config=None, harness_config=None):
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=load_harness(harness_config),
    )
```

Standalone harness use is the same runner without the `Env` adapter:

```python
from verifiers.types import ClientConfig

harness = vf.Harness(
    client=ClientConfig(
        client_type="openai_chat_completions",
        api_base_url="https://api.openai.com/v1",
        api_key_var="OPENAI_API_KEY",
    ),
    model="gpt-5.4-mini",
)

state = await harness.run(
    vf.Task({"prompt": [{"role": "user", "content": "What is 2+2?"}]}).freeze()
)
```

## Tasksets And Datasets

`Taskset(source=...)` accepts an iterable of rows or a zero-argument loader
function. Use a loader for datasets that should not be loaded at import time.

```python
from datasets import load_dataset


def source():
    dataset = load_dataset("gsm8k", "main", split="train")
    for index, row in enumerate(dataset):
        yield {
            "example_id": index,
            "prompt": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"],
        }


taskset = vf.Taskset(source=source)
```

`eval_source` is optional. If it is omitted, `get_eval_dataset()` uses the same
rows as `get_dataset()`.

Every task receives:

- `taskset_id`: the taskset identifier, defaulting to the class name;
- `task_id`: `task_id`, `id`, `example_id`, or a generated UUID.

For compatibility with the current `vf.Environment` worker schema,
`Taskset.get_dataset()` emits worker rows and stores the full canonical task as
a JSON string in `info["task"]`. `Taskset.to_task(...)` accepts a `Task`,
mapping, or that JSON payload.

### Group Setup

`Taskset.init_group(task, num_rollouts)` customizes group-consistent task/state
setup. The default duplicates the task and creates one fresh state per rollout.

Use this for prompt randomization that should be shared by all attempts in a
group, group-scoped seeds, or task variants that must be generated before
rollout dispatch.

```python
class MyTaskset(vf.Taskset):
    async def init_group(self, task, num_rollouts):
        tasks = []
        states = []
        shared_suffix = sample_suffix(task)
        for _ in range(num_rollouts):
            group_task = vf.Task({**task, "suffix": shared_suffix}).freeze()
            tasks.append(group_task)
            states.append(vf.State.for_task(group_task))
        return tasks, states
```

## Harness Programs

`Harness.run(task, state=None)` owns the rollout lifecycle:

1. initialize state if needed;
2. merge task runtime metadata into state;
3. start endpoint/tool/MCP/sandbox resources needed by the run;
4. run the configured program;
5. collect artifacts;
6. sync trajectory-derived prompt/completion fields;
7. record timing;
8. run rollout metrics and rewards;
9. run rollout cleanup and release rollout-scoped resources;
10. if no group boundary exists, run group cleanup and release group-scoped
    resources;
11. strip runtime handles and validate JSON serializability.

Handled `vf.Error` instances are serialized into `state["error"]` and still
flow through scoring and cleanup. Other exceptions raise after cleanup.

### Program Forms

`Harness.program` can be:

| Form | Meaning |
| --- | --- |
| `None` | default endpoint-backed tool loop |
| callable | Python program called in-process through the interception endpoint |
| `{"base": True, ...}` | explicit default loop, usually to set sandbox options |
| `{"entrypoint": "pkg.module:run", ...}` | importable Python program |
| `{"command": ["cmd", "arg"], ...}` | local or sandboxed command |

Mapping programs must specify exactly one of `base=true`, `entrypoint`, or
`command`. An option-only mapping such as `{"sandbox": True}` resolves to the
base loop; option-only mappings without sandbox placement hard fail because the
options would be inert.

The preferred Python program signature is:

```python
async def program(task, state):
    state["answer"] = "..."
    return state
```

If a third-party library needs an OpenAI/Anthropic-compatible client, add a
`client` parameter:

```python
async def program(task, state, client):
    ...
```

The client points at the v1 interception endpoint, not directly at the upstream
model provider.

### Default Tool Loop

The default loop reads `state["prompt"]`, sends it to the model with the
resolved tool definitions, executes tool calls, appends tool results, and
continues until one of these happens:

- a stop condition returns `True`;
- the model returns no tool calls and no user response is available;
- `max_turns` is reached.

If a taskset/harness supplies a `User`, the default loop calls it when the model
does not call a tool. If the user returns messages, the loop continues; if the
user returns no messages, the rollout stops with `stop_condition="no_tools"`.

### Program Placement

A program runs locally unless its mapping asks for sandbox placement.
`Harness.sandbox` is only the default primary sandbox config; it does not move
the program by itself.

```python
# Local command.
vf.Harness(program={"command": ["python", "run.py"]})

# Sandboxed command using Harness.sandbox.
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

Sandboxed programs support:

- `env`: environment variables for the program process;
- `files`: remote path -> literal string, `task.*` / `state.*` path, or
  callable value;
- `dirs`: remote path -> local path or importlib resource directory;
- `setup`: commands run after uploads and before the program;
- `artifacts`: text/JSON files read back into `state["artifacts"]`.

The sandboxed base loop uses the same interception endpoint as local programs.
The loop runs in the sandbox; tool execution and model forwarding remain owned
by the host runtime.

Common sandbox config keys:

- `image`: Docker image, defaulting to `python:3.11-slim`;
- `scope`: `rollout`, `group`, or `global`, defaulting to `rollout`;
- `start_command`: process used to keep the sandbox alive;
- `cpu_cores`, `memory_gb`, `disk_size_gb`, `gpu_count`;
- `network_access`, `timeout_minutes`;
- `packages`: Python packages installed before setup;
- `setup_commands`: shell commands run once after package install;
- `workdir`: working directory for the program command;
- `command_timeout`, `install_timeout`, `setup_timeout`.

### Model Controls

Standalone harnesses can receive `client`, `model`, and `sampling_args` at
construction. `Env` receives those controls from eval/training workers and
writes the serializable pieces into state for each rollout or group.

State runtime values take precedence for a specific rollout. This is how nested
or specialized runs can override model controls without changing the method
signature.

## Runtime State Helpers

User code should not accept a `runtime` argument. Runtime access is mediated
through `State` while the rollout is active:

```python
async def program(task, state):
    tools = state.tools()
    result = await tools["search"](query=task["question"])
    state["answer"] = result
    return state
```

Available helpers:

- `state.runtime()`: load the active process-local runtime;
- `state.tools()`: load callable tool handles for the current task/state;
- `state.run_harness(harness, task, state=None)`: run a child harness.

These helpers use a process-local runtime registry keyed by
`state["runtime"]["runtime_id"]`. The registry is not a persistence mechanism:
runtime IDs are valid only inside the process and only while the runtime is
alive. Runtime IDs, client keys, endpoint URLs, and sandbox lease keys are
stripped before returned state crosses the rollout/group boundary.

## Toolsets

Tools are packaged as `Toolset` objects. A bare callable passed through
`toolsets=[fn]` is wrapped in a one-tool toolset, but stateful tools, sandboxed
tools, MCP tools, bindings, stop conditions, cleanup, and teardown should be
declared on an explicit `Toolset`.

```python
async def search(query: str, index) -> str:
    return index.search(query)


toolset = vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)
```

`Toolset.tools` accepts:

- callable tools;
- nested `Toolset` objects;
- `vf.MCPTool(command=..., args=[...])` stdio servers;
- MCP command specs in config/TOML.

Tools are exposed by function/object name. Name collisions hard fail. Toolsets
show all tools by default; use `show=[...]` or `hide=[...]` to whitelist or
blacklist a nested tool surface.

### Hidden Bindings

Bindings inject arguments that the model does not see:

```python
vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)
```

Binding roots:

- `task.*`: read from the immutable task;
- `state.*`: read from mutable state;
- `runtime.*`: read serializable runtime metadata from state;
- `objects.*`: resolve a lazy toolset object;
- `tools.*`: call another resolved tool.

Arguments named `task`, `state`, and `runtime` are reserved. `task` and `state`
are injected automatically when a callable asks for them; runtime access goes
through state helpers. `sandbox` is reserved for tools owned by a sandboxed
toolset.

Tasks can select tools for one rollout:

```python
{
    "prompt": [{"role": "user", "content": "Use read-only tools."}],
    "runtime": {"tools": {"show": ["read_file"]}},
}
```

`runtime.tools` may be a list of tool names, `{"show": [...]}`, or
`{"hide": [...]}`. Unknown tool names hard fail.

### Tool Lifetimes

Lazy `objects` are scoped:

- read-only toolsets default to global object scope;
- `write=True` toolsets default to rollout object scope;
- `scope="rollout" | "group" | "global"` overrides the default.

Use global scope for reusable read-only clients or indexes, group scope when
scoring may need the object after rollout completion, and rollout scope for
mutable per-attempt state.

### Sandboxed Tools

Toolsets can request their own sandbox:

```python
async def python(code: str, sandbox) -> str:
    result = await sandbox.execute(f"python - <<'PY'\n{code}\nPY")
    return result.stdout


python_tools = vf.Toolset(
    tools=[python],
    write=True,
    sandbox={
        "image": "python:3.11-slim",
        "scope": "group",
        "packages": ["numpy"],
    },
)
```

The sandbox handle is hidden from the model. Sandbox refs and command records
are written to serializable state fields; live sandbox clients remain inside
the runtime.

A toolset can also use the primary program sandbox:

```python
vf.Toolset(tools=[inspect_workspace], sandbox="program", write=True)
```

`sandbox="program"` hard-fails unless a primary program sandbox is active.

### MCP Tools

Use `vf.MCPTool` for stdio MCP servers:

```python
fetch_tools = vf.Toolset(
    tools=[vf.MCPTool(command="uvx", args=["mcp-server-fetch"])]
)
```

In TOML/config:

```toml
[[env.harness.toolsets]]
tools = [
  { command = "uvx", args = ["mcp-server-fetch"] },
]
```

The runtime normalizes MCP tools into callable handles for Python programs and
can also present callable tools as an MCP proxy when `tool_protocol="mcp"`.

Programs can discover and call resolved tools through the interception endpoint:

- `GET {state["endpoint_root_url"]}/vf/tools`;
- `GET {state["endpoint_root_url"]}/vf/tools?protocol=openai_chat_completions`;
- `GET ...?protocol=openai_responses`;
- `GET ...?protocol=anthropic_messages`;
- `POST {state["endpoint_root_url"]}/vf/tools/{name}`.

Command and sandbox programs receive `VF_TOOLS_JSON`, `VF_TOOL_DEFS_JSON`,
`VF_TOOL_BASE_URL`, `VF_TOOL_API_KEY`, and `VF_ENDPOINT_API_KEY`.

## Users

A `User` is a callable that can return environment/user messages during the
default loop. Tasksets and harnesses may define at most one user.

```python
async def user(task, state, transcript):
    if len([m for m in transcript if m["role"] == "assistant"]) >= 2:
        return []
    return [{"role": "user", "content": "Try one more time."}]


taskset = vf.Taskset(source=source, user=user)
```

Direct callables are wrapped as `vf.User(fn=...)`. Use `vf.User(...)` when the
user needs bindings, lazy objects, scope, or a sandbox:

```python
taskset = vf.Taskset(
    user=vf.User(
        fn=user,
        scope="group",
        objects={"profile_db": load_profile_db},
        bindings={"profile_db": "objects.profile_db"},
        sandbox={"image": "python:3.11-slim", "scope": "group"},
    )
)
```

`transcript` is a default binding. It resolves to the current observable
conversation.

## Signals, Stop, Cleanup, Teardown

Signals are numeric functions. Function names are signal names. Collisions hard
fail.

### Rollout Signals

Rollout metrics and rewards run inside `harness.run` after program execution
and before rollout cleanup.

```python
@vf.metric
async def turns(task, state) -> float:
    return float(len(state["trajectory"]))


@vf.reward(weight=0.5, priority=10)
async def format_reward(task, state) -> float:
    return float("<answer>" in str(state.get("completion")))
```

Rollout signals must accept exactly `task, state`.

### Group Signals

Group signals run from `Env` group scoring after all rollouts in the group have
finished and after rollout cleanup has run.

```python
@vf.reward(stage="group")
async def best_of_n(tasks, states) -> list[float]:
    ...


@vf.advantage
async def centered(tasks, states) -> list[float]:
    ...
```

Group metrics/rewards/advantages must accept exactly `tasks, states` and return
one float per state. If no advantage signal is configured, v1 writes
`reward - mean(group_reward)` as the default advantage.

### Stop

Stop handlers can be contributed by tasksets, harnesses, and toolsets:

```python
async def submit(answer: str, state):
    state["answer"] = answer
    state["done"] = True
    return "submitted"


@vf.stop
async def submitted(task, state) -> bool:
    return bool(state.get("done"))


toolset = vf.Toolset(tools=[submit], stop=[submitted])
```

The built-in stop condition treats `state["done"]` and
`state["is_completed"]` as generic finish signals.

### Cleanup And Teardown

`@vf.cleanup` runs after scoring for its stage. Rollout cleanup receives
`task, state`; group cleanup receives `tasks, states`. Cleanup is the user
extension point for final state mutation and resource-related cleanup.

```python
@vf.cleanup(stage="group")
async def summarize_group(tasks, states):
    ...
```

`@vf.teardown` has no task/state arguments and runs when the harness runtime is
destroyed. Use teardown for global services and `atexit`-style cleanup.

There is no public render decorator. Framework-owned rendering such as timing,
completion sync, trajectory sync, and runtime-handle stripping is part of the
harness contract.

## Config And TOML

`TasksetConfig` and `HarnessConfig` are Pydantic models. Constructors accept
dicts, config objects, and direct Python objects. TOML/config strings resolve as
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
```

List-like fields are additive: constructor items and config items both
contribute. Scalar constructor arguments such as `source`, `program`,
`sandbox`, `user`, and `max_turns` override config values.

`scoring` tunes existing signal names:

```toml
[env.taskset.scoring.exact_answer]
weight = 0.5

[env.harness.scoring.turns]
skip = true
```

Config does not create a signal by name inside `scoring`; the function must
already be present through a constructor arg, config list, or decorated method.

Environment packages decide how to route their top-level config into
`load_taskset` and `load_harness`. A simple pattern is:

```python
def load_environment(config=None):
    config = config or {}
    return vf.Env(
        taskset=load_taskset(config.get("taskset")),
        harness=load_harness(config.get("harness")),
    )
```

### Custom Config Surfaces

Subclass `Taskset` or `Harness` when a package needs a reusable typed config
surface or a different method implementation. Keep subclasses shallow and
specific.

```python
class WikiTasksetConfig(vf.TasksetConfig):
    db_path: str


class WikiTaskset(vf.Taskset):
    config_type = WikiTasksetConfig

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

To inspect the active config shape:

```python
print(vf.TasksetConfig.schema_text())
print(vf.HarnessConfig.schema_text())
print(WikiTaskset.config_schema())
```

There is no public generic channel registry in v1. Stable cross-cutting
surfaces are promoted config fields. Local or package-specific surfaces should
live on a config subclass until they appear in enough tasksets/harnesses to
deserve promotion.

## Nested Harnesses

Use `state.run_harness(...)` to launch a child harness from an active rollout:

```python
async def ask_child(prompt: str, harness, state):
    task = vf.Task({"prompt": [{"role": "user", "content": prompt}]}).freeze()
    child_state = await state.run_harness(harness, task)
    return child_state["answer"]
```

The child receives a fresh `trajectory_id` and its own rollout-local state. It
inherits the parent group key, model, sampling args, and model client unless
the child harness or child state overrides them.

Each child call is recorded under `state["child_rollouts"]` as a serializable
`{"task": ..., "state": ...}` record. Child metrics are not merged into parent
metrics by default.

## Third-Party Python Programs

Third-party agent libraries should be configured against the v1 interception
endpoint inside the program call, not through global module state.

For OpenAI-compatible libraries:

```python
from verifiers.v1.utils.endpoint_utils import openai_endpoint_config


async def program(task, state, client):
    cfg = openai_endpoint_config(state, client)
    ...
```

For DSPy-style sync APIs, use the library's async entrypoint when available or
wrap blocking sync calls with `asyncio.to_thread(...)`.

## Examples

Reference implementations live beside their existing environments:

- `environments/reverse_text/reverse_text_v1.py`
- `environments/alphabet_sort/alphabet_sort_v1.py`
- `environments/wiki_search/wiki_search_v1.py`
- `environments/math_python/math_python_v1.py`
- `environments/mcp_search_env/mcp_search_v1.py`
- `environments/opencode_harbor/opencode_harbor_v1.py`
- `environments/tau2_bench/tau2_bench.py`
- `environments/nested_harness_v1/nested_harness_v1.py`
- `environments/hello_subagent_v1/hello_subagent_v1.py`
- `environments/hello_rlm_v1/hello_rlm_v1.py`
- `environments/dspy_flights/dspy_flights.py`
