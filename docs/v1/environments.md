# Building Environments

To scaffold an environment, run the following:
```bash
uv run init <MY_ENV_NAME>
```

There are optional flags:
- `-p`, `--path <dir>` — parent directory, default: `./environments`
- `-T`, `--add-tool` — also scaffold a `vf.Toolset` tool server at `servers/tool.py`
  - Use this option to create custom tools which are installed into the harnesses via MCP. However, not all harnesses support external tools.
- `-U`, `--add-user` — also scaffold a `vf.User` simulator at `servers/user.py`
  - Use this when you need to simulate a user interacting with the main LLM. However, not all harnesses support user simulation.
- `-H`, `--add-harness` — also scaffold a custom `vf.Harness` at `harness.py`, selectable via `--harness.id <name>`
  - In general, you should build your environments so that any of the built-in harnesses work. There are few reasons to build a custom harness.

However, for most environments, building a taskset should be enough.

> For a production-scale catalog of benchmark environments, see the companion [`research-environments`](https://github.com/PrimeIntellect-ai/research-environments) repository.

## A minimal environment

```python
import verifiers.v1 as vf


# The data for a given task
class AdditionData(vf.TaskData):
    answer: int


# A task defines a single problem and is defined as a subclass of vf.Task
class AdditionTask(vf.Task[AdditionData]):
    # @vf.reward denotes the scoring function for the task.
    # It needs the trace, which contains the whole message graph, including function calls, user messages etc.
    # It returns the reward for the single task based on this function.
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


# The taskset defines the tasks and needs a vf.TasksetConfig, which can be empty.
class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(100)
        ]


# Export the Taskset for verifiers to find it when loading
__all__ = ["AdditionTaskset"]
```

At execution time this taskset and the selected harness are lowered to the built-in
single-agent topology. Each invocation therefore produces a one-trace `AgentGraph`, using
the same runner and server path as an explicitly authored topology.

Use `@vf.metric` to record non-scored values. Comparisons across independent graph
invocations belong to the training algorithm; cross-trace environment judgement belongs
on a topology reward.

## Making values configurable

If you want to make certain fields configurable for the user, subclass `vf.TasksetConfig`:

```python
# Allow the user to change the number of tasks
class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100

class AdditionTaskset(vf.Taskset[AdditionTask, AdditionConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(self.config.num_tasks) # <- re-use the value here
        ]
```

Common usages for `vf.TasksetConfig` are load-time settings, such as splits (e.g., train/test), dataset names, etc.

You can also use `vf.TaskConfig` to configure task data, such as scoring parameters:

```python
class AdditionTaskConfig(vf.TaskConfig):
    tolerance: float = 0.0

class AdditionTask(vf.Task[AdditionData, vf.State, AdditionTaskConfig]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        tolerance = self.config.tolerance  # A task-wide confgi
        ...

class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100                             # --taskset.num-tasks
    task: AdditionTaskConfig = AdditionTaskConfig()  # --taskset.task.tolerance
```

The boundary: per-row data (the question, the reference answer) lives on `TaskData` fields;
values uniform across the taskset live on the config — load-time ones directly on the
`TasksetConfig`, task-facing ones under `task`. A task can also be constructed directly —
`AdditionTask(data, config=AdditionTaskConfig(...))` — and an omitted config defaults to
the declared type's defaults, so a standalone task works out of the box. Overriding
`from_trace(trace)` (not implemented by default) opts a task into being derived from a
finished rollout's bare `Trace` — how a multi-agent step spawns a follow-up task. Only the data rides the wire:
`trace.task.data` is the `TaskData` (with `trace.task.type` recording the producing Task
class's name), and behavior re-attaches by constructing the task class around it.

## Lazy and infinite tasksets

`load()` may be a generator instead of returning a list: yield each task as it's built.
Consumers materialize tasks through `Taskset.select`, which pulls only what a run needs —
`eval -n 5` builds 5 tasks, not the whole set — so a generator pays off whenever building
a task is expensive.

A procedural taskset can keep yielding forever. Declare `INFINITE = True` so consumers know
the stream never ends — infinity is inherent to the taskset, not a config knob; how many
tasks a run takes is the run's choice (`-n`), not the taskset's:

```python
import itertools
from collections.abc import Iterator


class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[AdditionTask]:
        for i in itertools.count():
            yield AdditionTask(
                AdditionData(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
```

Two rules follow from infinity: a run over an infinite taskset must be bounded with
`num_tasks` (`-n` on the CLI — omitting it is an error), and `shuffle` is a no-op (warned):
there is no whole set to sample from, and the first `n` generated tasks are already an
arbitrary sample. Generation must be deterministic — env-server pool workers each run
their own `load()` and rely on every worker producing the same sequence, so seed any
randomness with a constant (see `alphabet_sort_v1`, `color_codeword_v1`, or the built-in
`textarena` taskset).

## Adding Tools

Some environments require custom tools, which are bundled as a `vf.Toolset` (similar to how a `vf.Taskset` bundles `vf.Task`).
Tools are exposed as MCP servers to the given harness and thus need a harness which exposes MCP support (via `SUPPORTS_MCP`).

You can create them like this (remember the bootstrapping with `uv run init MY_ENV -T`):
```python
DATABASE = None

class SearchToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        """Search the task corpus."""
        return DATABASE.search(text)

# User-configurable knobs
class SearchConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()

class SearchTaskset(vf.Taskset[vf.Task, SearchConfig]):
    tools = (SearchToolset,)
```

Taskset tools are shared by a worker's rollouts. Tools can also be set per task.

## Using Judges

If your reward is semantic, use an LLM judge.

```python
import verifiers.v1 as vf
from functools import cached_property

class Task(vf.Task):
    answer: str

class CorrectnessJudge(vf.Judge[bool]):
    # The rubric for the judge
    prompt = """Question: {question}
    Answer: {answer}
    Response: {response}
    Correct? Reply yes or no."""

    # Parse the response from the judge
    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return "yes" in response.text


class JudgedData(vf.TaskData):
    answer: str


class JudgedTaskConfig(vf.TaskConfig):
    # The judge inherits base_url and api keys from the client config
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-5-mini")


class JudgedTask(vf.Task[JudgedData, vf.State, JudgedTaskConfig]):
    @vf.reward()
    async def correct(self, trace: vf.Trace) -> float:
        # Keeping judge configuration on TaskConfig makes it overridable from CLI/TOML.
        judge = CorrectnessJudge(self.config.judge)
        result = await judge.evaluate(
            trace=trace,
            question=self.data.prompt_text,
            answer=self.data.answer,
            # give the last assistant message to the judge
            response=trace.last_reply,
        )
        return float(result.parsed)


class SetConfig(vf.TasksetConfig):
    task: JudgedTaskConfig = JudgedTaskConfig()


class JudgeTraceTaskset(vf.Taskset[JudgedTask, SetConfig]):
    def load(self) -> list[JudgedTask]:
        return [
            JudgedTask(
                JudgedData(idx=0, prompt="What is 2+2?", answer="4"),
                self.config.task,
            )
        ]
```

To override the judge model, set `taskset.task.judge.model` in your config (it is a string).
