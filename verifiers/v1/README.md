# Verifiers v1

`verifiers.v1` is the current Taskset/Harness environment API for building eval
and training environments from two primary objects:

- a `Taskset`, which defines what is being attempted;
- a `Harness`, which defines how a model attempts it.

`vf.Env(taskset, harness)` adapts those objects to the existing
`vf.Environment` worker API used by evals and trainers. For local experiments,
`Harness` is runnable on its own with `await harness.run(task)`.

For current environment authoring guidance, start with
[`ENVIRONMENT_BEST_PRACTICES.md`](ENVIRONMENT_BEST_PRACTICES.md).

The programming model is intentionally small: tasks and state are serializable
data; everything else is a function, config value, or runtime-managed handle.

## Mental Model

![Task to Harness to State](../../docs/assets/v1-task-harness-state.svg)

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
`task["prompt"]` must not contain system messages. Use `task["system_prompt"]`
for per-task system instructions, or set `Taskset(system_prompt=...)` /
`Harness(system_prompt=...)` for package-level instructions.
Multiple system prompt sources reject by default; set `system_prompt_merge` in
`HarnessConfig` only when that harness knows how they should combine.
Tasks may also set `max_turns`, `tools`, `toolsets`, and `sandbox` at top level
for per-row runtime specialization. `tools` and `toolsets` use `show`/`hide`
maps; `runtime` remains hidden framework metadata.

```python
task = vf.Task(
    {
        "system_prompt": "Reverse text exactly.",
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
    }
).freeze()
```

### `State`

A `State` is a mutable, JSON-serializable rollout record. It starts from a task,
then accumulates trajectory, completion, metrics, reward, timing, artifacts,
errors, and any extra fields the environment author chooses to expose.
By convention, `task["answer"]` is the reference answer and `state["answer"]`
is the rollout's submitted answer. v1 does not copy `task["answer"]` into
top-level state; rewards and metrics should read reference data from `task`.

`trajectory` is the request-level audit log captured by the interception
endpoint. It contains the prompts, completions, tool calls, provider metadata,
and token-level details needed for evaluation and debugging. `completion` is the
post-prompt message view used by ordinary reward and display code, not
necessarily the last assistant message. When the default loop executes a tool
and stops before another model request, `completion` still includes the final
tool message even though no additional trajectory step exists.

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

If `harness` is omitted, `Env` uses the base endpoint-backed `Harness`:

```python
env = vf.Env(taskset=taskset)
```

Normal v1 environment packages should not subclass `Env`. Define or configure a
taskset and harness, then compose them.

## Minimal Environment

```python
import verifiers as vf


def source():
    yield {
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
        "max_turns": 1,
    }


@vf.reward(weight=1.0)
async def contains_answer(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


def load_taskset(config: vf.TasksetConfig):
    return vf.Taskset(source=source, rewards=[contains_answer], config=config)


def load_environment(config: vf.EnvConfig):
    return vf.Env(taskset=load_taskset(config=config.taskset))
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
function. A direct iterable is useful for tiny examples. A zero-argument loader
is the preferred path for real tasksets because it keeps imports and
constructors cheap.

The source loading contract is:

- source rows are plain JSON-serializable mappings;
- source loaders take no arguments;
- `Taskset` calls the loader on first access and caches the rows;
- config is resolved before source loading, then closed over by the loader;
- trainers and harnesses do not pass runtime values into source.

This keeps dataset loading lazy without reintroducing dynamic loader kwargs.
Use `load_taskset(...)` or a config-backed `Taskset` subclass to resolve names,
splits, paths, credentials, or package-specific options before defining the
loader.

```python
from datasets import load_dataset
import verifiers as vf


class GSM8KTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"
    split: str = "train"


def load_taskset(config: GSM8KTasksetConfig):
    dataset_name = config.dataset_name
    split = config.split

    def source():
        dataset = load_dataset(dataset_name, "main", split=split)
        for index, row in enumerate(dataset):
            yield {
                "example_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": row["answer"],
            }

    return vf.Taskset(source=source, config=config)
```

`eval_source` is optional. If it is omitted, `get_eval_dataset()` uses the same
rows as `get_dataset()`.

Every task receives:

- `taskset_id`: the taskset identifier, defaulting to the class name;
- `task_id`: `task_id`, `id`, `example_id`, or a generated UUID.

### Task Controls

Tasks can request rollout behavior through top-level serializable fields:

- `max_turns`: per-rollout turn limit for the base harness loop;
- `tools`: tool visibility as `{"show": [...]}` or `{"hide": [...]}`;
- `toolsets`: toolset visibility or rollout-local toolsets;
- `sandbox`: per-task primary sandbox overrides;
- `program`: per-task program options contributed by a taskset.

The priority rule is:

```text
explicit state.runtime > task top-level controls > harness defaults
```

`state.runtime` is only present when a caller passes an already-specialized
state into `harness.run(...)`, when `Taskset.init_group(...)` returns customized
states, or when `Env` writes model controls for eval/training. The base harness
default remains `max_turns=10`; a taskset can override it per row:

```python
yield {
    "prompt": [{"role": "user", "content": "Answer briefly."}],
    "max_turns": 3,
}
```

`task.runtime` is not part of the v1 task schema. Task rows should use top-level
controls; runtime metadata belongs on `state`.

`task.program` is the taskset-to-harness merge point for command/program data
that is task-owned but consumed by a harness. It can define only `files`,
`dirs`, `setup`, `env`, `artifacts`, and command `args`. The harness still owns
the program kind (`command`, `fn`, or base loop), sandbox placement, and program
channel. Duplicate file/env/artifact keys across harness and task fail fast.

Advanced callers may create a state for a new task while borrowing selected live
resources from an existing state. The stored state remains serializable: borrowed
resources are represented as runtime handles and stripped before return. The
runtime that created a resource owns cleanup, so a later harness that uses the
handle does not release it.

```python
child_state = state.for_task(
    child_task,
    borrow="model",
    tools="bash",
    transcript="append",
)
child_state = await child_harness.run(child_task, child_state)
```

Supported borrow targets are:

- `model`: reuse a live model client while allowing state/task model controls to
  specialize the request;
- `sandbox`: reuse the active primary program sandbox.

`tools` accepts a tool name or list of tool names. Borrowed tools remain owned by
the source runtime, so rollout-scoped backing resources such as sandboxes, MCP
sessions, and stateful clients are not duplicated.

`transcript="append"` writes the child rollout into the same public trajectory,
marked by the child `trajectory_id`. The parent owns that public log. Appended
child steps affect saved `completion` and `num_model_requests` when they run
before the framework `render_completion` update. The default is a private child
transcript.

For compatibility with the current `vf.Environment` worker schema,
`Taskset.get_dataset()` emits worker rows and stores the full canonical task as
a JSON string in `info["task"]`. `Taskset.to_task(...)` accepts a `Task`,
mapping, or that JSON payload.

Plain string task routes are not part of the v1 rollout schema. If a worker row
contains a top-level `task` field, it must be a JSON object string or mapping;
non-JSON strings fail during state initialization. Use `info["env_id"]` for
environment routing.

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

![v1 composition lifecycle](../../docs/assets/v1-composition-lifecycle.svg)

`Harness.run(task, state=None)` owns the rollout lifecycle:

1. initialize state if needed;
2. resolve task controls into hidden runtime metadata;
3. start endpoint/tool/MCP/sandbox resources needed by the run;
4. run rollout setup functions, including framework-owned program preparation;
5. run the configured program;
6. collect artifacts;
7. run rollout update functions, including the framework-owned late
   `render_completion` update;
8. record timing;
9. run rollout metrics and rewards;
10. run rollout cleanup and release rollout-scoped resources;
11. if no group boundary exists, run group cleanup and release group-scoped
    resources;
12. strip runtime handles and validate JSON serializability.

Handled `vf.Error` instances are serialized into `state["error"]` and still
flow through scoring and cleanup. Other exceptions raise after cleanup.

### Program Forms

`Harness.program` can be:

| Form | Meaning |
| --- | --- |
| `None` | default endpoint-backed tool loop |
| callable | Python program called in-process through the interception endpoint |
| `{"base": True, ...}` | explicit default loop, usually to set sandbox options |
| `{"fn": "pkg.module:run", ...}` | importable Python program |
| `{"command": ["cmd", "arg"], ...}` | local or sandboxed command |

Mapping programs must specify exactly one of `base=true`, `fn`, or
`command`. An option-only mapping such as `{"sandbox": True}` resolves to the
base loop; option-only mappings without sandbox placement hard fail because the
options would be inert.

The preferred Python program signature is:

```python
async def program(task, state):
    state["answer"] = "..."
    return state
```

If a third-party library needs an OpenAI/Anthropic-compatible client,
materialize one from state:

```python
async def program(task, state):
    client = state.get_client(api="chat")
```

The client points at the v1 interception endpoint, not directly at the upstream
model provider. `state.get_client(...)` defaults to async OpenAI chat completions;
pass `sync=True` for sync libraries, or choose another API with
`api="openai_responses"`, `api="openai_completions"`, or
`api="anthropic_messages"`. Short aliases `chat`, `responses`, `completions`,
and `messages` are also accepted.

### LLM-Free Replay And Solvers

Programs do not have to call a model. Offline solution replay, gold-patch
validation, cached completions, and deterministic solvers should be ordinary
programs that read immutable `task`, write serializable `state`, and then let
the normal scoring and cleanup lifecycle run.

```python
async def replay_solution(task, state):
    state["answer"] = task["answer"]
    return state


@vf.reward
async def exact(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))


taskset = vf.Taskset(source=load_rows, rewards=[exact])
harness = vf.Harness(program=replay_solution)
env = vf.Env(taskset=taskset, harness=harness)
```

This is the preferred shape for "solve without inference" flows. Use a custom
`Harness` subclass only when you are packaging reusable behavior with a new
configuration surface; do not subclass `Env` just to replay offline solutions.

### Default Tool Loop

The default loop reads `state["prompt"]`, sends it to the model with the
resolved tool definitions, executes tool calls, appends tool results, and
continues until one of these happens. If `state["system_prompt"]` is present,
the base harness prepends it to the model request without merging it into
`state["prompt"]`; custom programs and external harnesses decide how to consume
the resolved system prompt.

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
        "fn": "my_env.program:run",
        "sandbox": {"image": "python:3.11-slim"},
    }
)
```

Sandboxed programs support:

- `env`: environment variables for the program process;
- `files`: remote path -> literal string, `task.*` / `state.*` path, or
  callable value;
- `dirs`: remote path -> local path or importlib resource directory;
- `setup`: commands contributed to the rollout setup queue;
- `channels`: program-facing tool channels, optionally with late setup commands;
- `artifacts`: text/JSON files read back into `state["artifacts"]`.

Program does not own lifecycle scoring. `updates`, `metrics`, `rewards`,
`advantages`, and `cleanups` stay on the taskset, harness, or toolset.

Artifact specs use `{"path": "...", "format": "text" | "json"}`. Set
`optional=True` for logs or outputs that may not be produced on every rollout;
missing optional artifacts are recorded as `None`.

The sandboxed base loop uses the same interception endpoint as local programs.
The loop runs in the sandbox; tool execution and model forwarding remain owned
by the host runtime.

Tasksets can contribute task-local program options with the same field names:

```python
yield {
    "prompt": [{"role": "user", "content": instruction}],
    "sandbox": {"image": image, "workdir": "/app"},
    "program": {
        "files": {
            "/task/instruction.md": {"task": "instruction"},
            "/task/task.toml": {"task": "task_toml"},
        },
        "env": {"AGENT_WORKDIR": "/app"},
    },
}
```

Use this boundary when the taskset owns task assets, resource sizing, or
per-task environment variables, and the harness owns the reusable CLI behavior.

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

### Packaged CLI Harnesses And Harbor

Reusable CLI programs should be packaged as `Harness` subclasses. Package
implementations live under `verifiers.v1.packages` while the v1 API stabilizes,
and are re-exported from `verifiers.v1` for normal use. `OpenCode`, `Pi`,
`MiniSWEAgent`, `Terminus2`, and `RLM` are bundled `Harness` leaf wrappers for
common coding-agent CLIs.

```python
import verifiers as vf

env = vf.Env(
    taskset=vf.HarborTaskset(),
    harness=vf.OpenCode(),
)
```

`HarborTaskset()` loads Harbor-format task directories from the environment
package's reserved `tasks/` directory. `HarborTaskset(dataset="owner/name")`
fetches a Harbor Hub dataset. Harbor task rows contribute sandbox settings and
`task.program` uploads for `/task/instruction.md` and `/task/task.toml`.
`OpenCode` contributes the OpenCode install/setup, config generation, MCP tool
proxy wiring, and log artifact collection. `Pi` follows the same pattern for
Pi Coding Agent: the harness writes a Pi model config for the intercepted v1
endpoint and, when tools are enabled, installs the Pi MCP adapter and writes a
project `.mcp.json`. Neither side needs to know the other's private fields.
`MiniSWEAgent` owns mini-swe-agent installation, config layering, endpoint env,
and log/trajectory artifacts.
`Terminus2` owns Harbor Terminus agent installation, endpoint env, and log
artifacts.
`RLM` follows the same boundary for recursive LLM runs: `HarborTaskset` owns
the task directory and tests, while `RLM` owns RLM installation, optional skill
upload to `/rlm/skills`, endpoint wiring, and trajectory filtering.
Use `RLMConfig` in `env.harness` for RLM-specific settings such as
`rlm_repo_ref`, `rlm_tools`, `rlm_max_turns`, and `summarize_at_tokens`.
Tasksets can expose package-owned upload directories with `get_upload_dirs()`.
The base `Taskset` discovers a sibling `skills/` directory by default, and
`RLM` uploads that directory to `/rlm/skills` unless `skills=` is passed
explicitly to the harness.

## State Helpers

User code should not accept a `runtime` argument. Runtime access is mediated
through `State` while the rollout is active:

```python
async def program(task, state):
    tools = state.get_tools()
    result = await tools["search"](query=task["question"])
    state["answer"] = result
    return state
```

Available helpers:

- `state.get_model()`: read the resolved model name;
- `state.get_client(...)`: materialize an intercepted SDK client;
- `state.get_endpoint_config(...)`: materialize a config dict for third-party
  model libraries;
- `state.get_max_turns(default)`: resolve a rollout turn limit for harness
  authors;
- `state.get_tools()`: load callable tool handles for the current task/state.

These helpers may use process-local handles while the rollout is active. Handles
are not persistence mechanisms and are stripped before returned state crosses
the rollout/group boundary.

Message helpers are intentionally limited to transcript selection:
`vf.get_messages(messages, role=None)` returns matching typed message objects. It
does not parse answers or define a generic completion-text policy; index or
slice the returned list with ordinary Python, read `message.content` explicitly,
or bind a task-specific extractor on the taskset.

## Singletons And Collections

v1 keeps a sharp distinction between singleton fields and collection fields.

Singletons describe one logical value for a taskset, harness, or rollout:
`source`, `eval_source`, `program`, `user`, `model`, `client`, `system_prompt`,
and the primary program `sandbox`. Singleton runtime resources may be borrowed
across child harness calls when sharing is intentional:

```python
child_state = state.for_task(child_task, borrow="model")
child_state = state.for_task(child_task, borrow=["model", "sandbox"])
```

Collections are merged and extended: `toolsets`, `stops`, `setups`, `updates`,
`metrics`, `rewards`, `advantages`, and `cleanups`. Decorators stay singular
because each decorator marks one function, while constructor/config fields are
plural because they hold many functions.

Named tools can also be passed into a child state. The child sees the selected
tool surface, while calls still execute against the source runtime and its
rollout-scoped resources:

```python
child_state = state.for_task(child_task, borrow="model", tools="bash")
```

Local dependencies belong inside the package that owns the callable. Toolsets
and users can keep private dependency factories and bind those values into
their own callables, but those dependencies are not state fields or top-level
borrow targets.

## Toolsets

Tools are packaged as `Toolset` objects. A bare callable passed through
`toolsets=[fn]` is wrapped in a one-tool toolset, but stateful tools, sandboxed
tools, MCP tools, bindings, stop conditions, cleanup, and teardown should be
declared on an explicit `Toolset`.

```python
async def search(query: str, index) -> str:
    return index.search(query)


def load_index():
    return SearchIndex.open()


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
show all tools by default; use `show=[...]` or `hide=[...]` on a toolset to
whitelist or blacklist that toolset's nested tool surface.

Tasksets and harnesses can pass toolsets as a list or a mapping:

```python
vf.Taskset(
    source=source,
    toolsets={
        "wiki": load_wiki_toolset(),
        "python": vf.Toolset(tools=[python]),
    },
)
```

Mapped toolsets are still active by default, but their keys become task-level
addresses. List toolsets are active defaults but unnamed.

### Hidden Bindings

Bindings inject arguments that the model does not see:

```python
async def search(query: str, index) -> str:
    return index.search(query)


def load_index():
    return SearchIndex.open()


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
- `objects.*`: resolve a private dependency owned by the same callable package;
- `tools.*`: call another resolved tool.

Arguments named `task`, `state`, and `runtime` are reserved. `task` and `state`
are injected automatically when a callable asks for them; runtime access goes
through state helpers. `sandbox` is reserved for tools owned by a sandboxed
toolset.

`objects.*` is intentionally owner-private. Object factories are named zero-arg
loaders for private dependencies owned by the same `Taskset`, `Toolset`, or
`User`. If a hidden argument needs task or state data, bind it with a callable
source instead of an object factory. Framework args such as `task`, `state`,
`completion`, and `prompt` win over bindings when names collide.

String binding sources are always framework paths such as `task.answer` or
`objects.index`. Bind literal strings with a callable source so typos in binding
paths fail early instead of silently becoming constants.

Tasks can select toolsets and tools for one rollout:

```python
{
    "prompt": [{"role": "user", "content": "Use read-only tools."}],
    "toolsets": {"show": ["wiki"]},
    "tools": {"show": ["read_file"]},
}
```

`task.toolsets` accepts `{"show": [...]}` or `{"hide": [...]}` over mapped
toolset keys. `task.tools` accepts `{"show": [...]}` or `{"hide": [...]}` over
the flattened resolved tool names. In each mapping, `show` and `hide` are
mutually exclusive, and unknown names hard fail.

Task rows can also add rollout-local toolsets:

```python
{
    "toolsets": {
        "local_search": {
            "tools": ["my_env:search"],
            "bindings": {"search.index": "task.index_id"},
        }
    }
}
```

Task-local toolsets are resolved during rollout setup and follow the same
tool/binding/lifecycle rules as taskset and harness toolsets.

Callable tool schemas come from the Python callable. Use the function docstring
for the tool description, type annotations for JSON Schema, and either
Google-style `Args:` docstrings or `Annotated[..., pydantic.Field(description=...)]`
for argument descriptions:

```python
from typing import Annotated

from pydantic import Field


async def search(
    query: Annotated[str, Field(description="Search query.")],
) -> list[str]:
    """Search indexed pages."""
    ...
```

Dynamic schema-backed tools should still be callable objects. The object can
provide a `tool_def` when the visible schema is task data rather than a Python
signature:

```python
class DynamicTool:
    def __init__(self, tool_def):
        self.name = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, state, **arguments):
        state.setdefault("tool_calls", []).append({self.name: arguments})
        return "recorded"


def load_task_toolset(task):
    return vf.Toolset(tools=[DynamicTool(vf.Tool(**task["tool_schema"]))])
```

This keeps BFCL-style dynamic tools on the normal callable-tool path: the model
sees the task-specific schema, and the runtime still has an executable tool
surface for recording, validation, or replay.

Tool `**arguments` receive only model-visible arguments. Framework values such
as `task` and `state`, and configured hidden args such as bound clients or
sandboxes, are injected only when the callable declares them by name.

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
Use `prefer="program"` when a toolset should reuse the primary sandbox when one
exists, then fall back to provisioning its own scoped sandbox:

```python
vf.Toolset(
    tools=[python],
    write=True,
    sandbox={
        "prefer": "program",
        "image": "python:3.11-slim",
        "scope": "rollout",
    },
)
```

### MCP Tools

Use `vf.MCPTool` for stdio MCP servers:

```python
fetch_tools = vf.Toolset(
    tools=[vf.MCPTool(command="uvx", args=["mcp-server-fetch"])]
)
```

In TOML/config, toolsets are addressable by key:

```toml
[env.harness.toolsets.fetch]
tools = [
    { command = "uvx", args = ["mcp-server-fetch"] },
]

[env.taskset.toolsets.wiki]
fn = "my_env:load_wiki_toolset"
index = "simplewiki"
```

The runtime normalizes MCP tools into callable handles for Python programs and
can also present the resolved toolsets as an MCP server for sandbox command
programs. `program.channels` selects the program tool channel, not a concrete
tool name: it accepts `"callable"` or `"mcp"`, or a mapping whose keys are those
channels.
Concrete tools such as `bash` belong to a `Toolset` and are then exposed through
one of those channels.

Programs can discover and call resolved tools through the interception endpoint:

- `GET {state["endpoint_root_url"]}/vf/tools`;
- `GET {state["endpoint_root_url"]}/vf/tools?protocol=openai_chat_completions`;
- `GET ...?protocol=openai_responses`;
- `GET ...?protocol=anthropic_messages`;
- `POST {state["endpoint_root_url"]}/vf/tools/{name}`.

Python entrypoint programs should call resolved tools through state:

```python
async def program(task, state):
    tools = state.get_tools()
    result = await tools["search"](query=task["question"])
    state["answer"] = result
    return state
```

Sandboxed base and Python entrypoint programs use the callable channel by
default. Set `program={"sandbox": True, "channels": "callable"}` when the config
should make that channel explicit.
`program.channels` supports only the generic `callable` and `mcp` channels.
Harness-specific tool carriers, such as RLM skill uploads, should live on the
taskset upload directory contract or the harness config.

When a sandboxed `program.fn` ref points at local source, v1 resolves the
package from the module root: single-file modules use `pyproject.toml` in the
same directory as the module file, and package modules use `pyproject.toml`
inside the package directory. v1 uploads that package root and installs it in
the program sandbox before running the entrypoint. Dependencies are normal
package dependencies.

Command programs do not have a universal Python call surface. If
`program.channels` requests `mcp`, v1 materializes an MCP proxy for the resolved
toolsets. Generic CLI programs usually need a setup command to add that MCP
server to their own config, so the golden path is to put that setup under the
`mcp` key.

The channel setup entry is a late rollout setup contribution that runs inside
the program sandbox before the command. It may be a shell string or callable;
callables can request non-`task` / `state` args only through `program.bindings`.

```python
import json


def endpoint_config(state):
    return state.get_endpoint_config(api="chat")


def write_cli_config(endpoint_config):
    return f"""\
mkdir -p ~/.my-cli
cat > ~/.my-cli/model.json <<'EOF'
{json.dumps(endpoint_config)}
EOF
"""


harness = vf.Harness(
    program={
        "command": ["my-cli", "run", "/task/instruction.md"],
        "sandbox": True,
        "bindings": {"write_cli_config.endpoint_config": endpoint_config},
        "channels": {"mcp": write_cli_config},
    },
    sandbox={"image": "python:3.11-slim"},
)
```

`program.setup` is for installing or preparing the process. `program.channels.mcp`
is for registering resolved runtime surfaces such as MCP tools or intercepted
model endpoints. Both participate in the same priority-ordered setup stage as
`@vf.setup` handlers; built-in program uploads run early, `program.setup` runs
before ordinary priority-0 setup handlers, and channel setup runs late.
`program.setup` callables should only request `task` and `state`; use
`program.channels.<channel>` when setup needs bound runtime values such as an
endpoint config.

## Package Dependencies

Environment directories are Python packages. Put environment-specific
dependencies in that environment package's own `pyproject.toml`; do not add
benchmark-only or tool-only dependencies to the root package. This is expected
for packages such as BFCL, DSPy, Harbor/OpenCode, and browser/search
environments.

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
user needs bindings, private dependency factories, scope, or a sandbox:

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

`transcript` is a default binding for user functions. It currently means the
observable message list passed to the user simulator.

## Signals, Stop, Update, Cleanup, Teardown

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

Rollout signals can request framework args such as `task`, `state`,
`completion`, and `prompt`. Extra required arguments are valid when a taskset or
toolset binding supplies them. Group signals can request `tasks`, `states`, and
bound hidden args.

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

Group metrics/rewards/advantages must return one float per state. v1 writes
advantages only when an explicit advantage signal is configured.

`Env.requires_group_rollouts` is true when group-stage updates, signals,
cleanup, or custom group setup are present. `Env.provides_advantages` is true
only when explicit group advantages are present.

### Stop

Stop handlers can be contributed by tasksets, harnesses, and toolsets:

```python
async def submit(answer: str, state):
    state["answer"] = answer
    state.stop("submitted")
    return "submitted"
```

For custom programs and tools, `state.stop("reason")` is the generic finish
signal. A plain `state["done"] = True` also works through the built-in stop
condition when no custom reason is needed.
`state["is_completed"]`, `state["stop_condition"]`, `state["is_truncated"]`,
and `state["error"]` are framework-managed lifecycle fields. They appear on
returned states, but normal state writes cannot set them. Use `done`, `@vf.stop`,
`state.stop(...)`, or raise `vf.Error` instead.

### Update

`@vf.update` runs after program execution and before scoring for its stage.
Update functions prepare serializable state for metrics and rewards: parse
artifacts, copy sandbox outputs, normalize answer fields, or attach summaries.
The framework registers `render_completion` as a rollout update at priority
`-100`; ordinary updates run before it, and lower-priority updates intentionally
bypass the default completion render.

```python
@vf.update
async def parse_answer(task, state):
    state["answer"] = extract_answer(state.get("completion") or "")


@vf.update(stage="group")
async def summarize_attempts(tasks, states):
    ...
```

Rollout updates can request framework args and taskset/toolset-bound hidden
args. Group updates can request `tasks`, `states`, and bound hidden args.

### Cleanup And Teardown

`@vf.cleanup` runs after scoring for its stage. Cleanup functions can request
their stage's framework args and taskset/toolset-bound hidden args. Cleanup is
the user extension point for final state mutation and resource-related cleanup.

```python
@vf.cleanup(stage="group")
async def summarize_group(tasks, states):
    ...
```

`@vf.teardown` has no task/state arguments and runs when the harness runtime is
destroyed. Use teardown for global services and `atexit`-style cleanup.

## Config And TOML

v1 has two config layers:

1. Runner config: eval and RL TOML decide which environment to load, which model
   to use, how many rollouts to run, and what sampling args to pass.
2. v1 object config: `TasksetConfig` and `HarnessConfig` configure the taskset
   and harness inside that environment package.

Environment packages choose how to route runner config into v1 objects. The
recommended loader shape is:

```python
import verifiers as vf


def load_taskset(config: vf.TasksetConfig):
    return vf.Taskset(source=source, rewards=[exact], config=config)


def load_harness(config: vf.HarnessConfig):
    return vf.Harness(config=config)


def load_environment(config: vf.EnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

If the base harness is enough, omit `load_harness`:

```python
def load_environment(config: vf.EnvConfig):
    return vf.Env(taskset=load_taskset(config=config.taskset))
```

With that loader, eval TOML routes v1 config through the `taskset`/`harness`
sections:

```toml
# configs/eval/my-v1-env.toml
model = "openai/gpt-5.4-mini"
num_examples = 5
rollouts_per_example = 3

[[eval]]
env_id = "my-v1-env"
sampling_args = { max_tokens = 4096, reasoning_effort = "medium" }

[eval.args]
split = "test"

[eval.harness]
max_turns = 4

[eval.taskset.scoring.exact_answer]
weight = 0.5
```

For environment-specific settings, define leaf fields on the taskset or harness
config that owns them. An `EnvConfig` subclass only fixes the concrete taskset
and harness config types for the loader.

```python
class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig
    harness: vf.HarnessConfig


def load_taskset(config: MyTasksetConfig):
    ...


def load_harness(config: vf.HarnessConfig):
    ...


def load_environment(config: MyEnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

RL and Hosted Training TOML uses the same split under `env`:

```toml
# configs/rl/my-v1-env.toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 100
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 4096

[[env]]
id = "primeintellect/my-v1-env"

[env.harness]
max_turns = 8

[env.taskset]
split = "train"

[env.taskset.scoring.exact_answer]
weight = 1.0
```

Taskset and harness sections can import a base config with `config` and then
overlay local fields. Collection fields extend the imported config.

```toml
[env.harness]
config = "my_env.configs:load_another_harness_config"

[[env.harness.rewards]]
fn = "my_env.rewards:new_reward_func"
weight = 0
```

The outer runner owns model, endpoint, client, sampling, rollout count, and
training/eval controls. v1 config owns taskset/harness behavior. Only put
`harness.model` or `harness.client` in v1 config for standalone harnesses,
nested harnesses, or explicit auxiliary-model workflows.

`TasksetConfig` and `HarnessConfig` are Pydantic models. Constructors accept
dicts, config objects, and direct Python objects. TOML/config strings resolve as
`"module:object"` refs.

```python
taskset = vf.Taskset(
    config={
        "source": "my_env.data:load_rows",
        "eval_source": "my_env.data:load_eval_rows",
        "rewards": [
            {"fn": "my_env.signals:exact_answer", "weight": 1.0}
        ],
        "toolsets": {
            "search": {"tools": ["my_env.tools:search"]},
        },
    }
)
```

List-like fields are additive: constructor items and config items both
contribute. Scalar constructor arguments such as `source`, `program`,
`sandbox`, `user`, and `max_turns` override config values.

### Callable Config

Callable config fields use one grammar. Function identity is always the Python
function name; TOML does not define custom metric/reward/update names.

```toml
[[env.taskset.setups]]
fn = "my_env.setup:prepare_state"
priority = 20

[[env.taskset.updates]]
fn = "my_env.signals:parse_answer"
priority = 10

[[env.taskset.rewards]]
fn = "my_env.signals:exact_answer"
weight = 1.0
priority = 0

[[env.harness.cleanups]]
fn = "my_env.signals:close_trace"
stage = "group"
```

A bare string is shorthand when no metadata is needed:

```toml
[env.taskset]
rewards = ["my_env.signals:exact_answer"]
```

Group-stage signals and cleanup must opt in with `stage = "group"` and use
plural args (`tasks, states`). Rollout-stage callables are the default and use
`task, state`.

`scoring` tunes existing signal names:

```toml
[env.taskset.scoring.exact_answer]
weight = 0.5

[env.harness.scoring.turns]
skip = true
```

Config does not create a signal by name inside `scoring`; the function must
already be present through a constructor arg, config list, or decorated method.
`skip = true` belongs in `scoring`, not in a callable list entry.

### Sources

Use zero-argument source loaders in config so environment construction stays
cheap and import-safe:

```toml
[env.taskset]
source = "my_env.data:train_rows"
eval_source = "my_env.data:eval_rows"
```

```python
def train_rows():
    for row in load_dataset("my-org/my-dataset", split="train"):
        yield {
            "prompt": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"],
        }
```

`source` and `eval_source` are singleton fields. Passing them directly to
`Taskset(...)` overrides config.

### Toolsets

Toolsets are addressable resource packages, so TOML keys are toolset ids:

```toml
[env.taskset.toolsets]
wiki = "my_env.tools:load_wiki_toolset"

[env.taskset.toolsets.python]
fn = "my_env.tools:load_python_toolset"
packages = ["numpy", "pandas"]

[env.taskset.toolsets.search]
tools = ["my_env.tools:search"]
bindings = { "search.index" = "objects.index" }
```

A string value under `toolsets` or a `fn` table must produce exactly one
`Toolset`. A table without `fn` is an inline `Toolset` config.

Named toolsets can be selected per task:

```python
yield {
    "prompt": [{"role": "user", "content": question}],
    "toolsets": {"show": ["wiki"]},
}
```

Use `show` when a task should opt into a subset and `hide` when a task should
remove a subset. Do not set both.

### Program Config

Python programs can be configured directly:

```toml
[env.harness.program]
fn = "my_env.programs:run_agent"
sandbox = true
channels = "callable"
```

Command programs use `command` and can receive the resolved tools as an MCP
server when they run in a sandbox:

```toml
[env.harness.program]
command = ["bash", "-lc", "my-cli run /task/instruction.md"]
sandbox = true

[env.harness.program.channels]
mcp = true

[env.harness.program.files]
"/task/instruction.md" = "task.instruction"
```

`program.channels` is deliberately limited to `callable` and `mcp`.
Harness-specific tool carriers belong on the harness or taskset contract; for
example, RLM reads `Taskset.get_upload_dirs()["skills"]` and uploads it to
`/rlm/skills`.

`program.setup` prepares the process. `program.channels.mcp` registers resolved
tool or endpoint config after the interception endpoint is live and before the
command runs:

```toml
[env.harness.program]
command = ["my-cli", "run", "--config", "/tmp/my-cli.json"]
sandbox = true

[env.harness.program.channels]
mcp = { fn = "my_env.cli:write_cli_config" }

[env.harness.program.bindings]
"write_cli_config.endpoint_config" = { fn = "my_env.cli:endpoint_config" }
```

```python
import json
import shlex


def endpoint_config(state):
    return state.get_endpoint_config(api="chat")


def write_cli_config(endpoint_config) -> str:
    payload = json.dumps(endpoint_config)
    script = (
        "from pathlib import Path; "
        f"Path('/tmp/my-cli.json').write_text({payload!r})"
    )
    return "python -c " + shlex.quote(script)
```

Program bindings can point at `task.*`, `state.*`, `runtime.*`, `tools.*`, or a
callable `{ fn = "module:object" }` source. They cannot access Toolset
`objects.*`; toolset objects are private to their owning tools.

Command programs usually receive API keys through `program.env`. Values can be
plain strings, `task.*` / `state.*` paths, or callables:

```python
def openai_key(state):
    return state.get_endpoint_config(api="chat")["api_key"]


vf.Harness(
    program={
        "command": ["my-cli", "run"],
        "sandbox": True,
        "env": {"OPENAI_API_KEY": openai_key},
    },
    sandbox={"image": "python:3.11-slim"},
)
```

Tasksets can contribute task-specific program data without changing the harness
command:

```python
yield {
    "prompt": [{"role": "user", "content": instruction}],
    "program": {
        "files": {"/task/instruction.md": "task.instruction"},
        "env": {"TASK_ID": "task.example_id"},
    },
}
```

### Loading Config In Code

Config objects can be loaded directly from a TOML section:

```python
taskset_config = vf.TasksetConfig.from_toml("local.toml", "taskset")
harness_config = vf.HarnessConfig.from_toml("local.toml", "harness")

env = vf.Env(
    taskset=load_taskset(config=taskset_config),
    harness=load_harness(config=harness_config),
)
```

TOML can only express serializable values plus import refs. Direct Python
objects, live clients, and closures belong in code.

### Custom Config Surfaces

Subclass `Taskset` or `Harness` when a package needs a reusable typed config
surface or a different method implementation. Keep subclasses shallow and
specific.

```python
class WikiTasksetConfig(vf.TasksetConfig):
    db_path: str


class WikiTaskset(vf.Taskset):
    def __init__(self, config):
        config = WikiTasksetConfig(config)

        def load_db():
            return open_db(config.db_path)

        super().__init__(
            source=load_rows,
            toolsets=[
                vf.Toolset(
                    tools=[search],
                    objects={"db": load_db},
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
print(WikiTasksetConfig.schema_text())
```

There is no public generic channel registry in v1. Stable cross-cutting
surfaces are promoted config fields. Local or package-specific surfaces should
live on a config subclass until they appear in enough tasksets/harnesses to
deserve promotion.

## Nested Harnesses

Call child harnesses directly. The parent decides what, if anything, to write
back into its state:

```python
async def ask_child(prompt: str, harness, state):
    task = vf.Task({"prompt": [{"role": "user", "content": prompt}]}).freeze()
    child_state = await harness.run(task)
    state.setdefault("child_answers", []).append(child_state["answer"])
    return child_state["answer"]
```

The child receives a fresh `trajectory_id` and its own rollout-local state. It
does not automatically inherit model controls or write records into parent
state. To reuse live resources, construct the child state from the parent state:

```python
async def summarize(task, state):
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": str(state["completion"])}]}
    ).freeze()
    child_state = state.for_task(
        child_task,
        borrow="model",
        transcript="append",
    )
    child_state = await vf.Harness(
        system_prompt="Summarize the rollout in one sentence."
    ).run(child_task, child_state)
    state["summary"] = child_state["completion"]
```

## Third-Party Python Programs

Third-party agent libraries should be configured against the v1 interception
endpoint inside the program call, not through global module state.

For OpenAI-compatible libraries:

```python
async def program(task, state):
    cfg = state.get_endpoint_config(api="chat")
    model = state.get_model()
    ...
```

`api` accepts the same endpoint type names used in `endpoints.toml`
(`openai_chat_completions`, `openai_completions`, `openai_responses`,
`anthropic_messages`) plus the short aliases `chat`, `completions`,
`responses`, and `messages`.

For libraries that want a client object, use `state.get_client(...)`:

```python
async def program(task, state):
    client = state.get_client(api="responses")
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
- `environments/mcp_search_env/mcp_search_env.py`
- `environments/opencode_harbor/opencode_harbor.py`
- `environments/tau2_bench_v1/tau2_bench_v1.py`
- `environments/nested_harness_v1/nested_harness_v1.py`
- `environments/hello_subagent_v1/hello_subagent_v1.py`
- `environments/hello_self_judge_v1/hello_self_judge_v1.py`
- `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py`
- `environments/hello_rlm_v1/hello_rlm_v1.py`
- `environments/dspy_flights/dspy_flights.py`
