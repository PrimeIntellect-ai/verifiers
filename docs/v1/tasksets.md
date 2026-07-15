# Building Tasksets

A taskset defines the work to be done, which will be solved by the agent in a _harness_ running in a _runtime_.

You can scaffold a new taskset with the following:

```bash
uv run init addition-v1
```

The generated package has two important files:

```text
environments/addition_v1/addition_v1/
├── __init__.py  # exports the taskset entry point
└── taskset.py   # defines the data, tasks, and taskset
```

The command also supports:

- `-p`, `--path <dir>` — parent directory, default: `./environments`
- `-T`, `--add-tool` — also scaffold a `vf.Toolset` tool server at `servers/tool.py`
  - Use this to create custom tools which are installed into supported harnesses via MCP.
- `-U`, `--add-user` — also scaffold a `vf.User` simulator at `servers/user.py`
  - Use this to simulate a user interacting with the model. Not all harnesses support user simulation.
- `-H`, `--add-harness` — also scaffold a custom `vf.Harness` at `harness.py`, selectable via `--harness.id <name>`
  - Prefer a built-in harness unless the model needs to run inside a custom program.

Most tasksets do not need specific tools, user simulations or custom harnesses.

> For a production-scale catalog of tasksets, see the companion [`research-environments`](https://github.com/PrimeIntellect-ai/research-environments) repository.

## An example taskset

Tasksets are made of the following components:
- The **Taskset** loads the actual **Tasks** from a dataset using the `load()` function. It can be configured with the **TasksetConfig**, to e.g. load a certain split. Configs are exposed to the user and thus should only contain configurable values.
- A **Task** defines the scoring, stop conditions, setup, judging etc. of the task to solve. It also gets the tools or user config. It gets configured by a **TaskConfig**, e.g., to set a specific judge model.
- The **TaskData** is the immutable object that holds the actual data, i.e., the prompts, images, expected outputs etc., as well as other information such as timeouts (if set).

The following taskset generates addition questions and checks whether the model returned the exact answer.

```python
import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    # One immutable row in the dataset, including its reference answer.
    answer: int


class AdditionTask(vf.Task[AdditionData]):
    # @vf.reward denotes the scoring function for the task.
    # It needs the trace, which contains the whole message graph, including function calls, user messages etc.
    # It returns the reward for the single task based on this function.
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


class AdditionConfig(vf.TasksetConfig):
    # Values users can configure for the whole taskset.
    num_tasks: int = 100


# The Taskset itself
class AdditionTaskset(vf.Taskset[AdditionTask, AdditionConfig]):
    # The loading function for the actual tasks
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(self.config.num_tasks)
        ]
```

If a config class is not explicitly created, it means that no configurable, custom values are exposed to the user. In this example, there is no `vf.TaskConfig`, so no task values (like judge models) are configurable.

The scaffold also exports the taskset from `addition_v1/__init__.py`:

```python
from addition_v1.taskset import AdditionTaskset

__all__ = ["AdditionTaskset"]
```

The exported `AdditionTaskset` is what verifiers loads and makes discoverable for evaluation.

## Data and configuration

Keep values on the narrowest object that needs them:

- Put load-time values shared across the dataset, such as its split, name, seed, or size, on `TasksetConfig`.
- Put values used by every task during execution or scoring under `TasksetConfig.task`.

```python
class AdditionTaskConfig(vf.TaskConfig):
    tolerance: float = 0.0

class AdditionTask(vf.Task[AdditionData, vf.State, AdditionTaskConfig]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        error = abs(float(trace.last_reply) - self.data.answer)
        return float(error <= self.config.tolerance)

class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100
    task: AdditionTaskConfig = AdditionTaskConfig()
```

These values can be overridden with `--taskset.num-tasks` and `--taskset.task.tolerance`, or with the equivalent TOML fields.

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
arbitrary sample. The generator runs once, client-side (the eval entrypoint or the
prime-rl orchestrator pulls tasks off it and ships each task's data to the env server),
so nothing needs to re-produce the same sequence across processes; keep `load()`
deterministic only if you want `--resume` to regenerate the same first `n` tasks (see
`alphabet_sort_v1`, `color_codeword_v1`, or the built-in `textarena` taskset).

## Adding Tools

Some tasksets require custom tools, which are bundled as a `vf.Toolset` (similar to how a `vf.Taskset` bundles `vf.Task`).
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
