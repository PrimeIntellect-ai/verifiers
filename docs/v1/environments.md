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

You can also use `@vf.metric` to record non-scored values and `@vf.group_reward` for group rewards, which might be useful for training.

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

## Adding Tools

Some environments require custom tools, which are bundled as a `vf.Toolset` (similar to how a `vf.Taskset` bundles `vf.Task`).
Tools are exposed as MCP servers to the given harness and thus need a harness which exposes MCP support (via `SUPPORTS_MCP`).

You can create them like this (remember the bootstrapping with `uv run init MY_ENV -T`):
```python
DATABASE = None

class SearchToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        """Search the task corpus."""
        return DATABASE.search(text)

# User-configurable knobs
class SearchConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()

class SearchTaskset(vf.Taskset[vf.Task, SearchConfig]):
    # Launch the tools during setup
    def tools(self, task: vf.Task) -> list[vf.Toolset]:
        return [SearchToolset(self.config.tools)]
```

Tools can also be set per task, not just per taskset.

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
            question=self.data.prompt,
            answer=self.data.answer,
            response=trace.last_reply,  # Grade the final assistant answer.
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
