# Taskset/Harness Best Practices

This is the concise guide for writing taskset/harness (`TH`) environments. The
current implementation lives under `verifiers.envs.experimental`; examples use
`import verifiers as vf`.

## Core Shape

Use a taskset for **what** is being attempted, a harness for **how** it is
attempted, and resources for resolved runtime objects.

```python
def load_taskset(taskset_args=None):
    return vf.Taskset(
        source=lambda: [...],
        rubric=lambda: vf.Rubric(funcs=[...]),
        tools=lambda: load_toolset(...),
        name="my-taskset",
    )


def load_harness(harness_args=None):
    return vf.Harness(...)


def load_environment(taskset_args=None, harness_args=None):
    return vf.Env(
        taskset=load_taskset(taskset_args),
        harness=load_harness(harness_args),
    )
```

## Tasks

Treat `Task` as a frozen dataset row. It is dict-shaped,
JSON-serializable, and may contain arbitrary columns.

```python
rows = [
    {
        "prompt": [vf.UserMessage(content="Sort: delta alpha")],
        "answer": "alpha delta",
        "difficulty": "easy",
        "info": {"split": "train"},
    }
]

taskset = vf.Taskset(source=lambda: rows)
```

Use privileged keys when helpful: `prompt`, `answer`, `info`, `example_id`.
Everything else is ordinary row data: `task["repo"]`, `task["difficulty"]`,
`task["seed"]`, and so on.

Do not mutate `task` during rollout. Write rollout progress to `state`.
`Task` is frozen after construction at the top level; nested serializable values
should also be treated as read-only.
Live clients, file handles, sandbox handles, database connections, and other
non-serializable objects belong in `Resources`, usually through channel
resolution or toolset bindings.

```python
class MyHarness(vf.Harness):
    async def setup_state(self, task, resources):
        state = await super().setup_state(task, resources)
        state["attempts"] = []
        return state
```

## Taskset Channels

Channels are usually hidden behind taskset/harness arguments. Subclass a
taskset when row fields need to become resource requirements.

```python
class CodingTaskset(vf.Taskset):
    def channels(self, task=None):
        channels = super().channels(task)
        if task is not None:
            channels["sandbox"] = {
                "spec": vf.SandboxSeed(
                    image=task["image"],
                    files={task["repo_path"]: "/workspace"},
                ),
                "scoring": True,
            }
        return channels
```

Do not put generic channel config directly into task rows as the default user
API. Prefer clear task columns that the taskset translates into channels.

## Harnesses

Most harnesses only override small hooks. Let the base loop own model requests,
trajectory writing, stop checks, rendering, scoring, and cleanup.

```python
class SubmitHarness(vf.Harness):
    async def get_env_messages(self, task, state, resources):
        tool_calls = self.get_tool_calls(state)
        if not tool_calls:
            state["done"] = True
            return []
        messages = []
        for call in tool_calls:
            result = await resources.tools.call(
                call.name,
                resources,
                call.arguments,
                task=task,
                state=state,
            )
            messages.append(vf.ToolMessage(tool_call_id=call.id, content=str(result)))
        return messages
```

Use `state["done"] = True` for finish-tool patterns. The built-in stop handler
records the completion state.

## Signals

Use rollout-stage signals for per-rollout metrics/rewards. Group-stage signals
must opt in explicitly and use plural args.

```python
class ToolMetricsHarness(vf.Harness):
    @vf.metric
    async def num_turns(self, state):
        return len(state["trajectory"])


class PreferenceTaskset(vf.Taskset):
    @vf.reward(stage="group")
    async def best_of_group(self, tasks, states, resources):
        return [...]
```

Use `@vf.render` for pre-scoring state materialization and `@vf.cleanup` for
post-scoring cleanup.

```python
class LogHarness(vf.Harness):
    @vf.render(priority=100)
    async def render_logs(self, task, state, resources):
        state["logs"] = await ...

    @vf.cleanup(stage="group")
    async def delete_scoring_sandbox(self, tasks, states, resources):
        await ...
```

## Tools And Toolsets

Use `Toolset` for anything more than a plain read-only callable. Toolsets package
tools with bindings, lifecycle hooks, and channel requirements.

```python
def load_toolset():
    return vf.Toolset(
        tools=[search, fetch],
        bindings={
            "db": vf.ResourceBinding("wiki_db"),
            "query_id": vf.StateBinding("query_id"),
        },
        channels={
            "teardown": close_db,
        },
    )


async def search(query: str, db, query_id):
    ...
```

Hidden bound args are not shown to the model. Model-visible args remain in the
tool schema.

## Resources

Resources are the non-serializable runtime bag. Store clients, registries,
sandbox runtimes, databases, and endpoint servers there. Keep state
serializable.

```python
async def summarize(task, state, resources):
    judge = resources.require("judge_client")
    state["summary"] = await judge.summarize(state["completion"])
```

Use `resources.get(...)` when optional and `resources.require(...)` when the
object must exist.

## Endpoint Harnesses

Use `EndpointHarness` when external Python code should call a standard LLM
endpoint and have those calls intercepted into the shared trajectory path.

```python
class DSPyHarness(vf.EndpointHarness):
    async def execute(self, task, state, resources, client):
        result = await my_dspy_program(task["question"], client=client)
        state["answer"] = result.answer
        state["done"] = True
```

The external code should talk to the provided endpoint/client, not call the
environment directly.

## CLI/Sandbox Harnesses

Use `CliHarness` for external binaries that run in sandboxes.

```python
class MyCli(vf.CliHarness):
    def __init__(self):
        super().__init__(
            command="my-agent --task {instruction_path}",
            sandbox=vf.SandboxConfig(...),
            run=vf.RunConfig(max_turns=10),
        )
```

Tasksets own task data, images, fixtures, and scoring requirements. Harnesses
own command execution, endpoint wiring, logs, and process behavior.

## Group Setup

Group-consistent randomization should happen before rollout fanout. The taskset
or runner prepares task rows with group context embedded in normal columns.

```python
class PromptVariantTaskset(vf.Taskset):
    def prepare_group(self, rows, group_seed):
        variant = choose_variant(group_seed)
        return [{**row, "prompt_variant": variant} for row in rows]
```

Do not put group setup inside `Harness.setup_state`; that hook is per-rollout.

## Common Rules

- Task rows are read-only during rollout.
- State is serializable rollout progress.
- Resources own live objects.
- Tasksets and harnesses declare what they need; channels resolve compatibility.
- Rubrics can still be passed directly, but metrics/rewards are staged signals
  under the hood.
- Prefer loader callables (`source=lambda: ...`, `tools=lambda: ...`) for heavy
  objects and datasets.
