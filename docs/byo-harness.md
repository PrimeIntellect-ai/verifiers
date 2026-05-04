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
        "prompt": [{"role": "user", "content": "Reverse abc."}],
        "answer": "cba",
        "runtime": {"max_turns": 1},
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

## Task Runtime

`runtime` is a privileged task field for serializable rollout requests. Current
fields include:

- `max_turns`: per-rollout turn limit for the base harness loop;
- `tools`: tool visibility as a list of names, `{"show": [...]}`, or
  `{"hide": [...]}`.

Priority is:

```text
explicit state.runtime > task.runtime > harness defaults
```

`state.runtime` comes from explicit standalone state passing, `Taskset.init_group`
customization, or eval/training model controls. For normal taskset-only envs,
use task runtime:

```python
yield {
    "prompt": [{"role": "user", "content": "Use the search tool."}],
    "runtime": {"max_turns": 5, "tools": {"show": ["search"]}},
}
```

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
roots are `task.*`, `state.*`, `objects.*`, and `tools.*`.

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
        program={"entrypoint": "my_env.program:run"},
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
| `{"entrypoint": "pkg.module:run"}` | importable Python program |
| `{"command": ["cmd", "arg"]}` | local or sandboxed command |
| `{"sandbox": True}` | sandboxed default loop |

All model calls go through the v1 interception endpoint so trajectory capture,
tool forwarding, and protocol translation share one path.

Programs are also the right shape for LLM-free replay:

```python
async def replay_solution(task, state):
    state["answer"] = task["offline_answer"]
    state["stop_condition"] = "offline_replay"
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

## Signals And Cleanup

Metrics, rewards, and advantages are signal functions.

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

Rollout signals accept exactly `task, state`. Group signals accept exactly
`tasks, states` and return one value per state. Cleanup functions use
`@vf.cleanup`; teardown functions use `@vf.teardown`.

## When To Use Which Path

Use the core `SingleTurnEnv`, `ToolEnv`, and `MultiTurnEnv` docs when you want
the shortest path through the established environment classes.

Use BYO Harness when you want reusable tasksets, reusable harnesses, task-owned
or harness-owned toolsets, third-party Python programs, sandboxed programs,
stateful users, MCP tools, or nested harness calls.

The repository also includes a deeper implementation guide at
`verifiers/v1/README.md`.
