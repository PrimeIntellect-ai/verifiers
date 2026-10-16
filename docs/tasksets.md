# Building Tasksets

A taskset defines the work to be done: the tasks an agent receives, the setup and tools they require, and how each attempt is scored, which the agent solves inside a _harness_ running in a _runtime_.

> For a production catalog of tasksets (math, judge-graded QA, multimodal benchmarks, and containerized coding tasks), see [`prime-envs`](https://github.com/PrimeIntellect-ai/prime-envs). For an end-to-end walkthrough of packaging an environment with separate solver and grader sandboxes, see [Building Environments](building-environments.md).

## An example taskset

Tasksets are built from three core components:

- The **Taskset** loads the actual **Tasks** from a dataset or generator in `load()`. It is configured through **TasksetConfig** (for example, to pick a split or dataset size). Config classes are exposed on the CLI and in TOML, so they should only hold user-configurable settings.
- A **Task** defines the lifecycle hooks (`setup`, `finalize`), tools, stop conditions, metrics, and rewards for solving and grading a single problem. It is configured through **TaskConfig** (for example, to set a judge model or numeric tolerance).
- **TaskData** is the immutable record holding a single task's inputs and metadata, including its prompt, reference answer, artifacts, and any per-task timeout or resource requirements.

The following taskset generates addition questions and checks whether the model returned the exact answer:

```python
import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    # One immutable row in the dataset, including its reference answer.
    answer: int


class AdditionTask(vf.Task[AdditionData]):
    # @vf.reward marks a scoring function for the task.
    # It receives the trace (the full message graph, tool calls, and replies)
    # and returns this reward's score for the rollout.
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


class AdditionConfig(vf.TasksetConfig):
    # Values users can configure for the whole taskset.
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

If you don't need custom user-facing settings, you can omit the config subclass. Here `AdditionTask` uses the default `vf.TaskConfig`, while `AdditionConfig` exposes `num_tasks` on the taskset.

Following the standard [package setup](building-environments.md#create-a-package), a taskset package needs just two files:

```text
environments/addition_v1/src/addition_v1/
├── __init__.py  # exports the taskset entry point
└── taskset.py   # defines the data, tasks, and taskset
```

Re-export the taskset class in `src/addition_v1/__init__.py` so verifiers can discover and load it:

```python
from addition_v1.taskset import AdditionTaskset

__all__ = ["AdditionTaskset"]
```

Then install the package into the Python environment running the CLI and verify that it loads:

```bash
uv pip install -e environments/addition_v1
uv run vf-eval addition-v1 --dry-run
```

## Data and configuration

Keep settings on the narrowest config object that needs them:

- Put load-time values shared across the dataset, such as its split, name, seed, or size, on `TasksetConfig`.
- Put values used by each task during execution or scoring under `TasksetConfig.task`.

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

These values can be overridden on the CLI with `--env.taskset.num-tasks` and `--env.taskset.task.tolerance`, or in TOML under `[env.taskset]`.

## Lifecycle and scoring

Each single-agent rollout runs through five stages:

1. **Task `setup`** prepares files, repositories, or background services in the sandbox.
2. **Harness setup** prepares the agent program inside the runtime.
3. **Agent execution** runs the model and its tools until completion or a stop condition.
4. **Task `finalize`** captures any outputs or state needed for scoring and inspection before tests mutate the workspace.
5. **Scoring** runs metrics, rewards, and judges, after which the runtime is torn down.

Lifecycle hooks (`setup` and `finalize`) can accept `trace` and `runtime` by parameter name. See the [runtime guide](runtimes.md) for working with files, processes, and network rules, and [verifier placement](env.md#verifier-placement) when you want grading to run in a fresh container with selected paths transferred via `TaskData.artifacts`.

If you need mutable state during a rollout, such as tracking intermediate values across hooks or tool calls, define a `vf.State` subclass (`Task[YourData, YourState, YourConfig]`) and access it via `trace.state`. Because `trace.state` is ephemeral and not written to disk, anything you want preserved in the saved trace (such as extracted answers, patches, or parsed judge verdicts) should go into `trace.info` as JSON-serializable values.

Scoring methods are decorated with `@vf.metric` or `@vf.reward(weight=...)`:

- `@vf.metric` records an observational measurement on the trace without affecting the reward.
- `@vf.reward(weight=1.0)` contributes to `trace.reward`, which is the **sum** of each `score * weight`. A reward with `weight=0.0` is still computed and recorded on the trace, making it easy to log candidate signals without training on them.

Both decorators accept `trace`, `runtime`, or `task` by parameter name (where `task` is the `TaskData` instance and `self` is the `Task`), and each method can return either a single `float` or a `dict[str, float]` of named scores. During `Task.score`, metrics run first, followed by rewards, followed by configured judges. All methods within the same phase run concurrently via `asyncio.gather`; setting a higher `priority` controls display and registration order rather than making one method wait for another, so compute any shared inputs once in `finalize`. If an external service or judge fails during scoring, let the exception propagate instead of returning `0.0`, which would record an infrastructure failure as a wrong answer by the model.

`verifiers.v1` exports several helpers for common grading patterns:

- `extract_boxed_answer(text, strict=False)` — extracts the final balanced `\boxed{...}` expression (or returns `""` in strict mode when none is present).
- `verify_boxed_math_answer(response, answer)` — parses and compares a boxed math response against a reference answer using `math-verify`.
- `read_answer_file_or_last_reply(runtime, path, trace)` — reads a stripped answer file from the sandbox if non-empty, falling back to `trace.last_reply`.
- `parse_judge_choice(text, choices=("A", "B", "C"))` — extracts a discrete verdict token from a judge reply (stripping `<think>` blocks and handling boxed or labeled verdicts).
- `compare_stdout_results(actual, expected, tolerance=1e-3)` — compares program stdout against expected output with whitespace normalization and numeric tolerance.
- `parse_pytest_outcomes(output)` — parses a `pytest` short test summary into a `{node_id: outcome}` dictionary.

For software engineering tasks, call `await vf.capture_patch(trace, runtime, base_commit=...)` in `finalize` so the agent's git diff is saved to `trace.info["patch"]` before your grading script modifies the working tree or checks out test files. Pass a dataset base commit (or record `await vf.resolve_head(runtime)` during `setup`) so commits created by the agent are included in the diff. If the container image already has untracked files before the agent starts, snapshot them in `setup` with `verifiers.v1.utils.git.snapshot_untracked(runtime)` and pass that list as `ignore=...` so they aren't swept into the agent's patch.

When a task has a known reference solution or gold check that can be verified without calling a model, implement `async def validate(self, runtime: vf.Runtime) -> bool | None` on the task; see [Debugging and Validation](debugging.md) for running validation checks from the CLI.

### File answers and shared scoring inputs

When a task has a large input context or multiple scoring functions that inspect the same output, write the context into the sandbox during `setup` and read the agent's output once in `finalize`:

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

If you use `read_answer_file_or_last_reply`, make sure the prompt mentions that either writing `/workspace/answer.txt` or replying directly is accepted; if the benchmark strictly requires an output file, read it with `runtime.read` instead so missing files aren't masked by the final message.

When using [isolated verification](env.md#verifier-placement), `finalize` still runs in the solver's sandbox before artifacts are collected, whereas `@vf.metric`, `@vf.reward`, and configured judge plugins receive the fresh verifier runtime. Use `finalize` to capture solver-side evidence into `trace.info`, and declare any files the grader needs in `TaskData.artifacts`.

## Multimodal prompts

`TaskData.prompt` accepts a plain string, a list of typed messages (`vf.Messages`), or `None` when an external caller opens the conversation. To include images in a prompt, pass content parts on a `vf.UserMessage`:

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

Here `image` is a `PIL.Image` (converted to a `data:` URL by `image_data_url`), preserving the benchmark's exact interleaving of text and images. Note that `TaskData.prompt_text` concatenates only the text parts and drops images, so multimodal tasks must run with a vision-capable model and a harness that forwards typed messages (such as `null`) rather than one that flattens the prompt via `resolve_text_prompt`.

## Task identity

A task's `idx` is its position in the `load()` stream and is assigned automatically. You can also set `TaskData.id` to a durable identifier from the source dataset (such as an instance ID) and `TaskData.name` to a human-readable label; both are optional.

Every task has two identity properties recorded on traces:

- `task.hash` identifies the exact serialized `TaskData`. Evaluation `--resume` matches completed rollouts by `task.hash` (including repeated occurrences of the same hash), so changing a task's data intentionally invalidates prior cached results for that task.
- `task.key` provides stable logical identity across runs and defaults to `task.hash`. Override `Task.key` with a durable source ID when `TaskData` includes run-local fields like `idx`; keys must be unique within a taskset.

## Selecting tasks

Views on a taskset pick which tasks an iteration yields. In Python, they are lazy and can be chained in any order:

```python
taskset.include(idx=["0:100"]).exclude(names=["broken"]).shuffle(seed=0).take(5)
```

`include` keeps and `exclude` drops tasks matching any of `idx` (integers or non-negative Python slice strings like `"0:100"`, `"100:"`, or `"::2"`), `ids`, `keys`, or `names`; a task matches when any provided list names it. Any filter entry that matches nothing logs a warning once the stream has been consumed.

The eval, debug, validate, and GEPA entrypoints all accept a `select` config and apply it via `taskset.select(config.select)`, which chains the views in a fixed order: `include`, `exclude`, `shuffle`, `skip`, `limit`:

```bash
uv run vf-eval gsm8k --select.include.idx 0:100                 # the first 100 tasks
uv run vf-eval gsm8k -s -n 50                                   # 50 random tasks (--select.shuffle --select.limit 50)
uv run vf-eval gsm8k -s --select.skip 50 -n 50                  # the next 50 tasks from the same shuffled order
```

```toml
[select]
include = { idx = ["0:500"] }
exclude = { names = ["broken-task"] }
shuffle = true
limit = 100
```

Because `shuffle` (seeded by `select.seed`, which defaults to `0`) runs before `skip` and `limit`, increasing `limit` with the same seed extends the existing sample rather than drawing a different subset, and pairing `skip` with `limit` carves out disjoint splits.

## Lazy and infinite tasksets

`load()` can be a generator that yields tasks one at a time instead of returning a list. Because consumers iterate the taskset lazily, `vf-eval -n 5` only builds five tasks (unless filtering skips items or `shuffle` needs to materialize the candidate pool), which pays off whenever constructing a task is expensive.

A procedural taskset can also yield tasks forever. Declare `INFINITE = True` on the class so consumers know the stream has no end — infinity is a property of the taskset, while how many tasks a run takes (`-n`) is up to the caller:

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

Two rules follow from an infinite stream:

1. Every run must bound the taskset with `take` (`-n` / `--select.limit` on the CLI) or an `include` consisting only of closed `idx` ranges; omitting both raises an error.
2. `shuffle` needs a finite bound _before_ it runs, since you cannot shuffle an infinite stream. Because `select` applies `limit` after `shuffle`, shuffling an infinite taskset from the CLI requires a closed `include.idx` range first (for example, `--select.include.idx 0:1000 -s -n 50`); in Python, `taskset.take(1000).shuffle()` works as well.

The generator runs once on the client (the eval entrypoint or prime-rl orchestrator pulls tasks from `load()` and sends each task's `TaskData` to the environment worker), so workers never need to reproduce the generator sequence. Keep `load()` deterministic if you want `--resume` to regenerate the same first `n` tasks across runs (see `alphabet_sort`, `color_codeword`, or the built-in `textarena` taskset).

## Adding Tools

When a task requires custom tools beyond what the harness provides natively, bundle them in a `vf.Toolset`. Toolsets are exposed to the harness as MCP servers, so they require a harness that declares `SUPPORTS_MCP = True`.

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

Tools defined on `Taskset.toolsets(config)` run in a server shared across a worker's rollouts and take a `vf.SharedToolsetConfig`. When each rollout needs its own isolated tool server, define `Task.toolsets(config)` on the task class instead and store a `vf.ToolsetConfig` on your `TaskConfig`.

A toolset's config controls where the MCP server runs:

- **Host subprocess (default)** — launches the server as a local subprocess with its own filesystem.
- **`colocated = true`** (per-task toolsets only) — runs the tool server inside the agent's own runtime container so tools share the agent's filesystem directly.
- **`service = "<name>"`** (per-task toolsets on Prime or Modal VM Compose) — runs the tool server inside a named service from the task's Compose stack. The harness reaches it over the Compose network by service name, so the tool server can front internal services that the workspace cannot reach directly.
- **`runtime = { type = "docker", ... }`** — runs the tool server in its own dedicated container.
- **`url = "https://.../mcp"`** — connects to an already-running streamable-HTTP MCP server instead of launching one.

For local or containerized servers, put the `Toolset` subclass in a server module that calls `SearchToolset.run()` under `if __name__ == "__main__":`. On startup, the server runs `await self.setup()`, then `await self.setup_task(task)` (for per-task servers, receiving the rollout's `TaskData`), and finally `self.register(mcp)`. Shared servers skip `setup_task`, so keep shared indexes or clients on `self` during `setup()` and store per-rollout state in `self.state`.

### Stateful tools

To share mutable per-rollout state between a toolset and task scoring, parameterize both `Task` and `Toolset` with the same `vf.State` subclass. Each `@vf.tool` call automatically synchronizes `self.state` with the rollout, and scoring methods read the updated state from `trace.state`:

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

Initialize any per-rollout starting state in the task's `setup(trace, runtime)` via `trace.state` rather than in the tool server's `setup_task(task)`: state synchronization wraps individual `@vf.tool` calls, so mutations to `self.state` during server startup are not persisted back to the rollout. Because each tool call replaces the serialized state as a whole, avoid concurrent uncoordinated writes to `self.state` across parallel tool calls.

## Using Judges

When a task's reward is semantic and cannot be checked deterministically, use an LLM judge. For common grading patterns, you can plug in one of the built-in judges directly from config or `TaskConfig.judges` without writing a custom class:

- **`reference`** compares the model's response against a reference field on `TaskData` (if that field holds a list, matching any item in the list counts as correct):

  ```toml
  [[env.taskset.task.judges]]
  id = "reference"
  name = "correct"
  answer_field = "answer"
  view = "last_reply"
  ```

  To make this the default in code, set `judges: vf.Judges = [vf.ReferenceJudgeConfig(name="correct")]` on your `TaskConfig`.
- **`rubric`** loads a JSON or TOML rubric file from `path` containing a `criteria` list (each with `name`, `text`, optional `weight`, and `choices` ordered from worst to best). It evaluates the full trace by default, records each criterion as a `<judge_name>/<criterion>` metric, and returns their weighted average as the judge's reward.
- **`decision`** queries a decision-model endpoint (defaulting to `typesafe-ai/jev` at Prime's evaluations endpoint) across two to ten ordered grading `levels` and returns the probability-weighted level normalized to `[0, 1]`:

  ```toml
  [[env.taskset.task.judges]]
  id = "decision"
  name = "correct"
  answer_field = "answer"
  levels = ["Incorrect", "Correct"]
  ```

Every judge carries its own independent `model`, `base_url`, `api_key_var`, and `sampling` config; changing the evaluated agent's model or client does not alter the judge. When you configure multiple judges on a task, give each one a distinct reward `name`.

For custom prompts, structured outputs, or parsing logic, subclass `vf.Judge` and call it from a `@vf.reward` method:

```python
import verifiers.v1 as vf


class CorrectnessJudge(vf.Judge[bool]):
    # The prompt template rendered by build_messages(**fields)
    prompt = """Question: {question}
    Answer: {answer}
    Response: {response}
    Correct? Reply yes or no."""

    # Parse the judge's reply into the typed result
    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        verdict = vf.parse_judge_choice(response.text, choices=("yes", "no"))
        if verdict is None:
            raise ValueError("Judge returned no yes/no verdict")
        return verdict == "yes"


class JudgedData(vf.TaskData):
    answer: str


class JudgedTaskConfig(vf.TaskConfig):
    # Judge endpoint settings are independent of the evaluated agent's client.
    judge: vf.JudgeConfig = vf.JudgeConfig()


class JudgedTask(vf.Task[JudgedData, vf.State, JudgedTaskConfig]):
    @vf.reward()
    async def correct(self, trace: vf.Trace) -> float:
        # Keeping judge configuration on TaskConfig makes it overridable from CLI/TOML.
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

Passing `trace=trace` to `judge.evaluate` (or `judge.complete`) records the judge's prompt, raw response, parsed verdict, and token usage under `trace.info["judge_calls"]` and `trace.extra_usage`. To override the judge model from the CLI or TOML, set `env.taskset.task.judge.model`.

## Plugging hooks via config

Stop conditions, metrics, and rewards can also be plugged into any task from config without touching the taskset's code. Each entry names a function by import path (`pkg.module.function`, `pkg.module:function`, or `path/to/file.py:function`). Like decorated methods, scoring functions declare what they need by parameter name (`task`, `trace`, `runtime`), while a stop function's type annotation determines when it runs (see [stops and interception](#stops-and-interception)):

```toml
[env.taskset.task.stops]
single_turn = { fn = "my_hooks.py:two_turns" }

[env.taskset.task.metrics]
reply_length = { fn = "my_hooks.py:reply_length", priority = 10 }

[env.taskset.task.rewards]
exact_match = { fn = "my_hooks.py:exact_match", weight = 0.5 }
```

Plugged hooks merge with the task's decorated `@vf.stop`, `@vf.metric`, and `@vf.reward` methods; a plugged hook replaces a decorated method with the same name, making it easy to swap out an existing signal (such as `single_turn` above). Rewards accept a `weight` and every hook accepts a `priority`, while any omitted fields preserve the replaced method's existing settings.

Leaving `fn` out keeps the task's own decorated method and overrides only its metadata — for example, turning an existing zero-weight reward into an active training signal:

```toml
[env.taskset.task.rewards]
exact_match = { weight = 1.0 }
```

## Stops and interception

Decorate a task method with `@vf.stop` (synchronous or async) and return `True` to end the agent run early and record the method's name as `trace.stop_condition`. The parameter's type annotation controls where in the model-call lifecycle the check runs:

- `vf.Request` — before the outgoing request is sent to the model provider.
- `vf.Response` — while the model's response is buffered, before it is returned to the harness.
- `vf.Trace` — before each model call, inspecting the trace accumulated so far.

When you want to inspect or rewrite traffic in flight without ending the rollout, use `@vf.intercept` on a method taking `vf.Request` or `vf.Response` (plus optional `trace`). Return a modified copy of the same type to replace the payload, or `None` to leave it untouched:

- **Request interception** can rewrite new user or tool messages before the model sees them (without adding or removing messages, altering tool definitions, or changing tool-call names and IDs).
- **Response interception** can replace the assistant's text reply before it reaches the harness (replacement responses cannot carry tool calls).

For ordinary network isolation, configure `TaskData.network_allow` and `TaskData.network_block`. If you want to block a specific tool call before it executes and return a synthetic tool result so the agent can keep going, the harness must declare `SUPPORTS_TOOL_INTERCEPTION = True` and check tool calls before running them; otherwise, use a `vf.Response` stop hook to halt the rollout before the harness executes the tool. See the bundled [`interception`](../environments/interception/interception/taskset.py) and [`bash_interception`](../environments/bash_interception/bash_interception/taskset.py) environments for complete working examples.

## Beyond one agent

One episode doesn't have to be a single agent run: multi-agent topologies, the control flow between agents, and cross-agent rewards are the environment's job — see [The Env](env.md).
