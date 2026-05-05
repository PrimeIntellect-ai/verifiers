# BYO Harness

BYO Harness is the `verifiers.v1` authoring path for environments that need a
clean separation between the task being attempted and the way a model attempts
it.

Use this path when you want to bring your own harness: a tool loop, CLI program,
third-party Python program, sandboxed program, user simulator, MCP server, or
nested sub-harness workflow. For simple one-off environments, the core
[Environments](environments.md) guide remains the shortest path.

## Core Shape

v1 environments are composed from:

- `Taskset`: task rows, task-owned tools, user behavior, metrics, rewards, and
  cleanup;
- `Harness`: rollout behavior, model endpoint forwarding, program execution,
  harness-owned tools, sandboxes, and nested harness calls;
- `Env`: adapter that makes a taskset/harness pair usable by eval and training
  workers.

The smallest v1 environment only needs a taskset. If no harness is passed,
`vf.Env` uses the base endpoint-backed harness.

```python
import verifiers.v1 as vf


def source():
    yield {
        "system_prompt": "Reverse text exactly.",
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
        "max_turns": 1,
    }


@vf.reward(weight=1.0)
async def contains_answer(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[contains_answer], config=config)


def load_environment(taskset_config=None):
    return vf.Env(taskset=load_taskset(taskset_config))
```

## Tasksets

`Taskset(source=...)` accepts either a direct iterable of rows or a zero-argument
loader. Direct iterables are fine for tiny examples. Real tasksets should use a
zero-argument loader so imports and constructors stay cheap.

```python
from datasets import load_dataset


def load_taskset(config=None):
    config = config or {}
    dataset_name = config.get("dataset_name", "gsm8k")
    split = config.get("split", "train")

    def source():
        dataset = load_dataset(dataset_name, "main", split=split)
        for index, row in enumerate(dataset):
            yield {
                "example_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": row["answer"],
            }

    return vf.Taskset(source=source)
```

Source rows are JSON-serializable mappings. Config is resolved before source
loading and closed over by the loader; trainers and harnesses do not pass
runtime values into source.

## Task Controls

Tasks can request rollout behavior through top-level serializable fields:

- `max_turns`: per-rollout turn limit for the base harness loop;
- `tools`: tool visibility as `{"show": [...]}` or `{"hide": [...]}`;
- `toolsets`: toolset visibility or rollout-local toolsets;
- `sandbox`: per-task overrides for a sandboxed program.

Priority is:

```text
explicit state.runtime > task top-level controls > harness defaults
```

Keep system instructions out of `prompt`. v1 resolves `system_prompt` from the
task, taskset, and harness as a separate field; the base harness concatenates
the resolved system messages with `prompt` only when it submits a model request.
If more than one source provides a system prompt, resolution fails unless the
harness explicitly sets a merge policy.

`state.runtime` comes from explicit standalone state passing, `Taskset.init_group`
customization, or eval/training model controls. For normal tasksets, use
top-level task controls:

```python
yield {
    "prompt": [{"role": "user", "content": "Use the search tool."}],
    "max_turns": 5,
    "tools": {"show": ["search"]},
}
```

`task.runtime` is not part of the public task schema. Runtime metadata lives on
`state.runtime` and is written by the harness, the taskset group initializer, or
the eval/training worker.

## Toolsets

Tools are packaged as `Toolset` objects. A taskset can own tools directly:

```python
async def search(query: str, index) -> str:
    return index.search(query)


toolset = vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)

taskset = vf.Taskset(source=source, toolsets=[toolset])
```

Bindings inject hidden arguments that the model does not see. Common binding
roots are `task.*`, `state.*`, and `tools.*`. Tool and user callables can also
bind `objects.*` from their own private dependency factories.

MCP servers are also tools:

```python
taskset = vf.Taskset(
    source=source,
    toolsets=[
        vf.Toolset(
            tools=[vf.MCPTool(command="uvx", args=["mcp-server-fetch"])]
        )
    ],
)
```

## Harnesses

Create a harness when rollout behavior is no longer just "call the model with
the resolved taskset tools."

```python
def load_harness(config=None):
    return vf.Harness(
        program={"fn": "my_env.program:run"},
        config=config,
    )


def load_environment(taskset_config=None, harness_config=None):
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=load_harness(harness_config),
    )
```

`Harness.program` can be:

| Form | Meaning |
| --- | --- |
| `None` | default endpoint-backed tool loop |
| callable | Python program called in-process |
| `{"fn": "pkg.module:run"}` | importable Python program |
| `{"command": ["cmd", "arg"]}` | local or sandboxed command |
| `{"sandbox": True}` | sandboxed default loop |

All model calls go through the v1 interception endpoint so trajectory capture,
tool forwarding, and protocol translation share one path.

Sandbox command programs can request the resolved tools as an MCP server with
`program={"command": [...], "sandbox": True, "tools": "mcp"}`. Python programs
receive callable tool handles by default, or can set
`program={"sandbox": True, "tools": "callable"}` when the base loop is moved
into a sandbox.

Programs are also the right shape for LLM-free replay:

```python
async def replay_solution(task, state):
    state["answer"] = task["answer"]
    state.stop("replayed")
    return state


@vf.reward
async def exact(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))


def load_environment():
    taskset = vf.Taskset(source=load_rows, rewards=[exact])
    return vf.Env(taskset=taskset, harness=vf.Harness(program=replay_solution))
```

Use this for cached completions, deterministic solvers, and gold-solution
validation. Subclass `Harness` only when packaging reusable behavior with a new
config surface; do not subclass `Env` just to bypass inference.

## Updates, Signals, And Cleanup

Update functions, metrics, rewards, and advantages are lifecycle functions
around the rollout/group scoring boundary.

```python
@vf.metric
async def turns(task, state) -> float:
    return float(len(state["trajectory"]))


@vf.reward(weight=1.0)
async def correct(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


@vf.reward(stage="group")
async def best_of_n(tasks, states) -> list[float]:
    ...
```

Rollout signals accept `task, state`, plus any Toolset-bound hidden args. Group
signals accept exactly `tasks, states` and return one value per state. Update
functions use `@vf.update` and run before scoring; cleanup functions use
`@vf.cleanup` and run after scoring; teardown functions use `@vf.teardown`.

`env.requires_group_rollouts` is true when group-stage updates, scoring,
cleanup, or group setup are part of the environment contract.
`env.provides_advantages` is true when the environment has explicit advantage
handlers.

## When To Use Which Path

Use the core `SingleTurnEnv`, `ToolEnv`, and `MultiTurnEnv` docs when you want
the shortest path through the established environment classes.

Use BYO Harness when you want reusable tasksets, reusable harnesses, task-owned
or harness-owned toolsets, third-party Python programs, sandboxed programs,
stateful users, MCP tools, or nested harness calls.

The repository also includes a deeper implementation guide at
`verifiers/v1/README.md`.
