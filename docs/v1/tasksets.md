# Building Tasksets

A taskset loads tasks and defines how to score them.

For a complete package with separate agent and grading sandboxes, start with
[building environments](building-environments.md).

Create a package with:

```bash
uv run vf-init addition-v1
```

The generated package has two important files:

```text
environments/addition_v1/addition_v1/
├── __init__.py  # exports the taskset entry point
└── taskset.py   # defines the data, tasks, and taskset
```

The command also supports:

- `-p`, `--path <dir>` — choose the parent directory (default: `./environments`).
- `-T`, `--add-tool` — add a `vf.Toolset` server at `servers/tool.py` for custom MCP tools.
- `-H`, `--add-harness` — add a `vf.Harness` at `harness.py`. Select it with `--env.agent.harness.id <name>`. Use a built-in harness unless you need a custom program.

Install the package in the Python environment running the CLI:

```bash
uv pip install -e environments/addition_v1
uv run vf-eval addition-v1 --dry-run
```

For real examples, see [`prime-envs`](https://github.com/PrimeIntellect-ai/prime-envs): AIME checks math answers, PaperSearchQA uses a judge, MMMU-Pro includes images, and Harbor tasksets run in containers. Check each package's Verifiers dependency before copying its code.

## An example taskset

The main classes are:

| Class | What it holds |
| --- | --- |
| `TaskData` | One task's prompt, reference answer, and runtime requirements. These values cannot be changed after creation. |
| `Task` | Code to set up, run checks on, and score that task. |
| `Taskset` | A `load()` method that creates the tasks. |
| `TaskConfig` | Settings used while running or scoring a task. |
| `TasksetConfig` | Settings used to load tasks, such as the dataset split. |

The following taskset generates addition questions and checks whether the model returned the exact answer.

```python
import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    answer: int


class AdditionTask(vf.Task[AdditionData]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


class AdditionConfig(vf.TasksetConfig):
    num_tasks: int = 100


class AdditionTaskset(vf.Taskset[AdditionTask, AdditionConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(self.config.num_tasks)
        ]
```

Only add a config class when users need custom settings. This example adds
`num_tasks` to the taskset config and uses the base task config.

The scaffold also exports the taskset from `addition_v1/__init__.py`:

```python
from addition_v1.taskset import AdditionTaskset

__all__ = ["AdditionTaskset"]
```

verifiers loads the class listed in `__all__`.

## Data and configuration

Put each setting where it is used:

- Dataset settings, such as split, seed, or size, go on `TasksetConfig`.
- Execution and scoring settings go on `TaskConfig`, stored under `TasksetConfig.task`.

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

These values can be overridden with `--env.taskset.num-tasks` and `--env.taskset.task.tolerance`, or with the equivalent TOML fields (`[env.taskset]`).

## Lifecycle and scoring

An agent run follows this order:

1. Task `setup` prepares files and services.
2. Harness setup prepares the agent program.
3. The agent works on the task.
4. Task `finalize` saves outputs needed for scoring or later inspection.
5. Scoring runs, then the runtime is released.

Task hooks can request `trace` and `runtime` parameters. See the
[runtime guide](runtimes.md) for files, services, and network access. To grade in
a fresh sandbox, use [isolated verification](env.md#isolated-deterministic-verification)
and list the files to transfer in `TaskData.artifacts`.

For values needed only during a run, use a `vf.State` subclass:
`Task[YourData, YourState, YourConfig]`. Access it through `trace.state`.
It is not saved. Put evidence you need after the run in `trace.info`, using values
that can be saved as JSON.

`@vf.metric` records a measurement. `@vf.reward(weight=...)` adds to the total
reward: `trace.reward` is the **sum** of each `score * weight`. A reward with
weight zero is still recorded. Scoring methods return a float or a dictionary of
named scores. They can request `trace`, `runtime`, or `task` by parameter name.
Here, `task` means `TaskData`; `self` is the `Task` object.

Metrics run first, then rewards, then configured judges. Methods in the same
group run concurrently: a higher `priority` does not make one finish before
another. Prepare shared inputs in `finalize`. Raise an exception when a service
or judge fails; returning zero would count the failure as a wrong answer.

Reuse the scoring helpers exported by `verifiers.v1` when they match the benchmark:

| Helper | Purpose |
| --- | --- |
| `extract_boxed_answer(text, strict=False)` | Extract the final balanced `\boxed{...}`; strict mode returns empty when absent |
| `verify_boxed_math_answer(response, answer)` | Compare a boxed answer with the reference using math-verify |
| `read_answer_file_or_last_reply(runtime, path, trace)` | Prefer a nonempty answer file, falling back to the last reply |
| `parse_judge_choice(text, choices=("A", "B", "C"))` | Extract a verdict from a judge response |
| `compare_stdout_results(actual, expected, tolerance=1e-3)` | Compare program outputs with whitespace and numeric tolerance |
| `parse_pytest_outcomes(output)` | Parse pytest short-summary test outcomes |

For coding tasks, call `vf.capture_patch(trace, runtime, base_commit=...)` in
`finalize`, before tests change the repository. Use the dataset's base commit or
save `vf.resolve_head(runtime)` during setup. This also captures changes the agent
committed. To exclude untracked files already in the image, record them with
`verifiers.v1.utils.git.snapshot_untracked` and pass the result as `ignore`.
The patch is saved in `trace.info["patch"]`.

Implement `async def validate(self, runtime) -> bool | None` when a gold answer or solution
can be checked without a model. See [validation and replay](debugging.md) for the
commands and their limits.

### File answers and shared scoring inputs

For long inputs, write the context into the sandbox and point the prompt at its
path. Read the answer once in `finalize` when several scores need it:

```python
class ContextData(vf.TaskData):
    context: str
    answer: str


class ContextTask(vf.Task[ContextData]):
    NEEDS_CONTAINER = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.write("/workspace/context.txt", self.data.context.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.info["answer"] = await vf.read_answer_file_or_last_reply(
            runtime, "/workspace/answer.txt", trace
        )

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        return float(bool(trace.info["answer"]))

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return float(trace.info["answer"] == self.data.answer)
```

This task's prompt should name both paths and say that either the answer file or
the final reply is accepted. Use `runtime.read` directly when the benchmark
requires a file; a fallback would change its scoring rules. Save small parsed
results or judge verdicts in `trace.info` for reuse, with distinct keys for your
own values. Custom judge calls already write request records under `info["judge"]`.

With isolated verification, `finalize` still runs in the solver sandbox. Use it
to capture evidence; keep checks that need private tests or a clean filesystem
in scoring methods, which receive the verifier runtime. Transfer their input
files through `TaskData.artifacts`.

## Multimodal prompts

`TaskData.prompt` accepts text, typed messages, or `None` if the caller will send
the first message. For images, use message content:

```python
from verifiers.v1.utils.image import image_data_url

data = vf.TaskData(
    prompt=[
        vf.UserMessage(
            content=[
                vf.TextContentPart(text="What is shown in this image?"),
                vf.ImageUrlContentPart(
                    image_url=vf.ImageUrlSource(url=image_data_url(image)),
                ),
            ]
        )
    ]
)
```

Here `image` is a PIL image. Keep the benchmark's text and image order unchanged.
`prompt_text` extracts only text, so it loses images. Choose a model and harness
that accept image messages: `null` accepts typed messages; a harness using
`resolve_text_prompt` rejects them.

## Task identity

A task's `idx` is its position in the `load()` stream, assigned automatically.
Use `TaskData.id` for a stable source ID and `TaskData.name` for a readable name.
Both are optional.

A task's `hash` is computed from its data. Its `key` identifies the task across
runs and defaults to the hash. If fields such as `idx` change between runs,
override `Task.key` with a stable ID from the dataset. Keys must be unique within
a taskset. Traces record both values.

## Selecting tasks

In Python, chain methods to select tasks:

```python
taskset.include(idx=["0:100"]).exclude(names=["broken"]).shuffle(seed=0).take(5)
```

`include` keeps matching tasks; `exclude` removes them. Match by `idx`, `ids`,
`keys`, or `names`. A task matches if any entry selects it. Indices accept integers
and Python slices such as `"0:100"`, `"100:"`, or `"::2"`, but no negative values.
An entry that matches nothing logs a warning once the stream has been read.

Eval, debug, validate, and GEPA accept the same `select` config. It applies
`include`, `exclude`, `shuffle`, `skip`, and `limit`, in that order:

```bash
uv run vf-eval gsm8k --select.include.idx 0:100
uv run vf-eval gsm8k -s -n 50
uv run vf-eval gsm8k -s --select.skip 50 -n 50
```

The last two commands take different groups of 50 tasks from the same shuffled
order. `select.seed` controls that order and defaults to zero.

```toml
[select]
include = { idx = ["0:500"] }
exclude = { names = ["broken-task"] }
shuffle = true
limit = 100
```

With the same inputs and seed, raising `limit` extends the selection instead of
drawing a new sample.

## Lazy and infinite tasksets

`load()` can yield tasks one at a time instead of building a list. This is useful
when creating tasks is expensive: `vf-eval -n 5` then loads only five tasks,
unless filtering skips tasks or shuffling needs the full set.

A taskset that generates tasks forever must declare `INFINITE = True`. The run
chooses how many to take with `-n`:

```python
import itertools
from collections.abc import Iterator


class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[AdditionTask]:
        for i in itertools.count():
            yield AdditionTask(
                AdditionData(prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
```

An infinite taskset needs a finite selection: use `-n` or closed index ranges
such as `--select.include.idx 0:1000`. To shuffle it on the CLI, use closed
`include.idx` ranges first, for example `--select.include.idx 0:1000 -s -n 50`.
`limit` alone is too late because it runs after shuffle. In Python, use
`taskset.take(n).shuffle()`.

The client generates tasks once and sends their data to workers. To support
`--resume`, make sure `load()` produces the same first `n` tasks on the next run.
Examples include `alphabet_sort`, `color_codeword`, and `textarena`.

## Adding Tools

Use `vf.Toolset` for tools the task needs beyond those built into the harness.
Tools are served through MCP, so the harness must declare `SUPPORTS_MCP`.

Start with `uv run vf-init MY_ENV -T`. For example:

```python
DATABASE = None


class SearchToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        """Search the task corpus."""
        return DATABASE.search(text)


class SearchConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()


class SearchTaskset(vf.Taskset[vf.Task, SearchConfig]):
    @classmethod
    def toolsets(cls, config: SearchConfig) -> list[vf.Toolset]:
        return [SearchToolset(config.tools)]
```

`Taskset.toolsets` creates servers shared by a worker's rollouts. For one server
per rollout, use `Task.toolsets(config)` and put its `vf.ToolsetConfig` on the task
config. Choose where the server runs:

- Default tool runtime: a host subprocess, with its own filesystem access.
- `colocated = true`: a task tool runs inside the harness runtime.
- `runtime = {type = "docker"}`: a separate tool container.
- `url = "https://.../mcp"`: connect to an existing streamable-HTTP MCP service.

A local server module must call `SearchToolset.run()` under
`if __name__ == "__main__":`; `vf-init -T` includes this. Use `setup()` to prepare
the server and `setup_task(task)` to receive task data in a server created per task.

`Toolset[Config, State]` gives tools access to rollout state through `self.state`.
Each update replaces the whole state, so concurrent writes can overwrite each
other. Coordinate them when needed. To share the agent's files, set
`colocated = true`; connecting through MCP alone does not share files.

### Stateful tools

Use the same state class on the task and its toolset. Decorated tool calls read
and update that rollout's state; task scoring reads it through `trace.state`:

```python
class CounterState(vf.State):
    calls: int = 0


class CounterToolset(vf.Toolset[vf.ToolsetConfig, CounterState]):
    @vf.tool
    def count(self) -> int:
        """Count this tool call and return the total."""
        self.state.calls += 1
        return self.state.calls


class CounterTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class CounterTask(vf.Task[vf.TaskData, CounterState, CounterTaskConfig]):
    @classmethod
    def toolsets(cls, config: CounterTaskConfig) -> list[vf.Toolset]:
        return [CounterToolset(config.tools)]

    @vf.metric
    async def tool_calls(self, trace: vf.Trace) -> float:
        return float(trace.state.calls)
```

Put the toolset and its `run()` entry point in the scaffold's server module.
Initialize per-rollout values in the task's `setup` through `trace.state`.
Server `setup_task(task)` receives `TaskData`, not a `Task` or `Trace`; use it
for fixed inputs such as a source page or service address. State synchronization
wraps tool calls, so changing `self.state` in server setup does not initialize
the rollout's state.

Server startup runs `setup()`, then `setup_task(task)` for per-task servers,
then `register(mcp)`. Shared servers skip `setup_task`; keep shared indexes and
clients on the server, and per-rollout progress in `self.state`. If a task needs
a service, start it and wait for readiness in task `setup` before tools use it.
Override `register(mcp)` only for dynamic tool schemas that `@vf.tool` cannot
describe; the default implementation registers and synchronizes decorated tools.

## Using Judges

Use a judge model when code alone cannot check the answer. Start with a built-in
judge. The `reference` judge compares the response with a field on `TaskData`:

```toml
[[env.taskset.task.judges]]
id = "reference"
name = "correct"
answer_field = "answer"
view = "last_reply"
```

To make this the package default, set
`judges: vf.Judges = [vf.ReferenceJudgeConfig(name="correct")]` on the task config.
If the answer field contains a list, each item is an acceptable answer.

The `rubric` judge reads a JSON or TOML file from `path`. Its `criteria` list
contains a `name`, `text`, optional `weight`, and `choices` ordered from worst to
best for each criterion. It reads the full trace by default, records each
criterion as a metric, and returns their weighted average. Give each judge a
different reward `name`.

Every judge has its **own** `model`, `base_url`, `api_key_var`, and `sampling`.
Changing the evaluation's model or client does not change the judge. Set the
judge's endpoint explicitly if needed. Pass `trace=trace` to custom judge calls
to record their requests, verdicts, and token usage.

For a custom rubric or response format, subclass `vf.Judge`:

```python
import verifiers.v1 as vf


class CorrectnessJudge(vf.Judge[bool]):
    prompt = """Question: {question}
    Answer: {answer}
    Response: {response}
    Correct? Reply yes or no."""

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        verdict = vf.parse_judge_choice(response.text, choices=("yes", "no"))
        if verdict is None:
            raise ValueError("Judge returned no yes/no verdict")
        return verdict == "yes"


class JudgedData(vf.TaskData):
    answer: str


class JudgedTaskConfig(vf.TaskConfig):
    # Judge endpoint settings are independent of the evaluated agent's client.
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-5-mini")


class JudgedTask(vf.Task[JudgedData, vf.State, JudgedTaskConfig]):
    @vf.reward()
    async def correct(self, trace: vf.Trace) -> float:
        judge = CorrectnessJudge(self.config.judge)
        result = await judge.evaluate(
            trace=trace,
            question=self.data.prompt_text,
            answer=self.data.answer,
            response=trace.last_reply,
        )
        return float(result.parsed)


class SetConfig(vf.TasksetConfig):
    task: JudgedTaskConfig = JudgedTaskConfig()


class JudgeTraceTaskset(vf.Taskset[JudgedTask, SetConfig]):
    def load(self) -> list[JudgedTask]:
        return [
            JudgedTask(
                JudgedData(prompt="What is 2+2?", answer="4"),
                self.config.task,
            )
        ]
```

To override the judge model, set `env.taskset.task.judge.model` in your config (it is a string).

## Adding hooks through config

Config can add or replace stop conditions, metrics, and rewards. Set `fn` to a
function path: `pkg.module.function`, `pkg.module:function`, or
`path/to/file.py:function`. Async scoring functions request `task`, `trace`, or
`runtime` by parameter name. A stop function's type annotation determines when
it runs; see [stops and interception](#stops-and-interception).

```toml
[env.taskset.task.stops]
single_turn = { fn = "my_hooks.py:two_turns" }

[env.taskset.task.metrics]
reply_length = { fn = "my_hooks.py:reply_length", priority = 10 }

[env.taskset.task.rewards]
exact_match = { fn = "my_hooks.py:exact_match", weight = 0.5 }
```

A config hook replaces a decorated method with the same name, such as
`single_turn` above. Other hooks stay in place. Rewards accept `weight`; all
hooks accept `priority`. Higher priorities sort first, but scoring methods still
run concurrently. Fields you leave out keep the method's existing values.

Leave out `fn` to change settings while keeping the existing function. For
example, give an existing reward a weight of one:

```toml
[env.taskset.task.rewards]
exact_match = { weight = 1.0 }
```

## Stops and interception

Return `True` from a `@vf.stop` method to stop the run. Its name is recorded as
the stop condition. The parameter type decides when it runs:

- `vf.Request`: before the request goes to the provider.
- `vf.Response`: before the response reaches the harness.
- `vf.Trace`: before each model call, using the trace recorded so far.

Stop hooks can be synchronous or async.

`@vf.intercept` can replace a request or response. Return the same type, or
`None` to leave it unchanged. Request replacements may edit new user or tool
messages, but cannot add or remove messages, change tools, or change tool-call
names or IDs. Response replacements must contain only assistant text, with no
tool calls.

Use `TaskData` for ordinary network restrictions. To reject a tool call before
it runs and then let the agent continue, the harness needs
`SUPPORTS_TOOL_INTERCEPTION` and code that checks tool calls before execution.
Without that support, the run must stop.
See the bundled [interception](../../environments/interception/interception/taskset.py)
and [bash interception](../../environments/bash_interception/bash_interception/taskset.py)
examples for working hooks.

## Beyond one agent

To run several agents or score their results together, use an [Env](env.md).
