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

However, for most environments, a task subclass plus a thin taskset loader is enough.

> For a production-scale catalog of benchmark environments, see the companion [`research-environments`](https://github.com/PrimeIntellect-ai/research-environments) repository.

## A minimal environment

A `vf.Task` carries a single problem's data *and* its behavior — the `@vf.reward` /
`@vf.metric` scoring methods, lifecycle hooks, and tool/user declarations all live on the
task subclass and read the task's own fields off `self`. The `vf.Taskset` is just the
loader: config in, typed tasks out, one concrete task type per taskset.

```python
import verifiers.v1 as vf


# A task defines a single problem: its data fields plus how it is scored.
class AdditionTask(vf.Task):
    answer: int

    # @vf.reward denotes a scoring function. It receives whatever it declares by
    # parameter name (`trace`, `runtime`); `self` is the task, so your fields are
    # right there. The trace contains the whole message graph, including function
    # calls and user messages.
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.answer))


# The taskset is the loader. It needs a vf.TasksetConfig, which can be empty.
class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load(self) -> list[AdditionTask]:
        # Load the dataset, in this case we build it on the initial load
        return [
            AdditionTask(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i)
            for i in range(100)
        ]


# Export the Taskset for verifiers to find it when loading
__all__ = ["AdditionTaskset"]
```

You can also use `@vf.metric` to record non-scored values, `@vf.group_reward` for group
rewards (comparing a task's rollouts, useful for training), and `@vf.stop` for stop
conditions. Lifecycle hooks (`setup`, `finalize`, `validate`) are methods on the task too.

## Making values configurable

If you want to make certain fields configurable for the user, subclass `vf.TasksetConfig`:

```python
# Allow the user to change the number of tasks
class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100

class AdditionTaskset(vf.Taskset[AdditionTask, AdditionConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i)
            for i in range(self.config.num_tasks) # <- re-use the value here
        ]
```

Common usages for `vf.TasksetConfig` are **load-time** settings: splits (e.g., train/test),
dataset names, sample counts, rng seeds.

Knobs the *task itself* reads — scoring parameters, judge endpoints, server placement — go
on a `vf.TaskConfig`, nested on the taskset config under `task` and stamped onto every
loaded row. The task reads them off `self.config`, typed by parameterizing the task:

```python
class AdditionTaskConfig(vf.TaskConfig):
    tolerance: float = 0.0

class AdditionTask(vf.Task[vf.State, AdditionTaskConfig]):
    answer: int

    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        tolerance = self.config.tolerance  # a config knob, not a per-row field
        ...

class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100                              # load-time: --taskset.num-tasks
    task: AdditionTaskConfig = AdditionTaskConfig()   # task-facing: --taskset.task.tolerance
```

The boundary: per-row data (the question, the reference answer) lives on task fields;
values uniform across the taskset live on the config — load-time ones directly on the
`TasksetConfig`, task-facing ones under `task`. A task can also be constructed directly
with a config (`AdditionTask(idx=0, prompt=..., config=AdditionTaskConfig(...))`) — omitted,
it defaults to the declared type's defaults, so a standalone task works out of the box.

## Adding Tools

Some environments require custom tools, which are bundled as a `vf.Toolset` (similar to how a `vf.Taskset` bundles `vf.Task`).
Tools are exposed as MCP servers to the given harness and thus need a harness which exposes MCP support (via `SUPPORTS_MCP`).

Declare the toolset classes on the task (remember the bootstrapping with `uv run init MY_ENV -T`):

```python
DATABASE = None

class SearchToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        """Search the task corpus."""
        return DATABASE.search(text)

# User-configurable knobs (placement: colocated / shared / own runtime)
class SearchTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()

class SearchTask(vf.Task[vf.State, SearchTaskConfig]):
    tools = (SearchToolset,)

class SearchConfig(vf.TasksetConfig):
    task: SearchTaskConfig = SearchTaskConfig()

class SearchTaskset(vf.Taskset[SearchTask, SearchConfig]):
    ...
```

The framework builds each declared server with the matching config off the task's config —
the field whose type is the server's declared config type (here `SearchTaskConfig.tools`,
i.e. `--taskset.task.tools.*`), falling back to a default-constructed one. Override
`Task.server_config` if you need explicit pairing (e.g. two servers sharing one config
type). User simulators follow the same pattern: `user = MyUser` on the task, a
`vf.UserConfig` field on the task config.

## Using Judges

If your reward is semantic, use an LLM judge.

```python
import verifiers.v1 as vf

class CorrectnessJudge(vf.Judge[bool]):
    # The rubric for the judge
    prompt = """Question: {question}
    Answer: {answer}
    Response: {response}
    Correct? Reply yes or no."""

    # Parse the response from the judge
    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return "yes" in response.text


class Config(vf.TaskConfig):
    # The judge inherits base_url and api keys from the client config (env vars, with the Prime CLI config as a fallback)
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-5-mini")


class JudgedTask(vf.Task[vf.State, Config]):
    answer: str

    @vf.reward()
    async def correct(self, trace: vf.Trace) -> float:
        judge = CorrectnessJudge(self.config.judge)  # config knobs stay CLI-tunable
        result = await judge.evaluate(
            trace=trace,
            question=self.prompt,
            answer=self.answer,
            # give the last assistant message to the judge
            response=trace.last_reply,
        )
        return 1.0 if result.parsed else 0.0


class SetConfig(vf.TasksetConfig):
    task: Config = Config()


class JudgeTraceTaskset(vf.Taskset[JudgedTask, SetConfig]):
    def load(self) -> list[JudgedTask]:
        return [JudgedTask(idx=0, prompt="What is 2+2?", answer="4")]
```

To override the judge model, set `taskset.task.judge.model` in your config (it is a string).
Sampling knobs live under `taskset.task.judge.sampling` — e.g.
`taskset.task.judge.sampling.max_tokens`.

Judges can also be plugged **from config alone** — no judge-calling code: the base
`TaskConfig.judges` list (`--taskset.task.judges`) attaches judge plugins (built-ins like
`reference` / `rubric`, or hub packages) to every task at load, and `Task.score` runs them
after the task's own `@reward`s. Each task records its judges' configs on the wire
(`Task.judges`), so a saved trace shows exactly what judged it.
