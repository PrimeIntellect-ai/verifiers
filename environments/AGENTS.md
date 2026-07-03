# environments/AGENTS.md

<!-- Generated for repository development workflows. Do not edit directly. -->

This file mirrors the "Environments" documentation page.

---

To scaffold an environment, run the following:
```bash
prime env init <MY_ENV_NAME>
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

## A minimal environment

```python
import verifiers.v1 as vf 


# A task defines a single problem and is defined as a subclass of vf.Task
class AdditionTask(vf.Task):
    answer: int


# The taskset defines the dataset (collection of tasks) and needs a vf.TasksetConfig, which can be empty.
class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[AdditionTask]:
        # Load the dataset, in this case we built it on the initial load
        return [
            AdditionTask(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i)
            for i in range(100)
        ]

    # @vf.reward denotes the scoring function for the environment.
    # It needs the actual task (defined earlier) as well as the trace, which contains the whole message graph, including function calls, user messages etc.
    # It returns the reward for the rollout based on this function.
    @vf.reward
    async def exact_match(self, task: AdditionTask, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(task.answer))


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
    def load_tasks(self) -> list[AdditionTask]:
        return [
            AdditionTask(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i)
            for i in range(self.config.num_tasks) # <- re-use the value here
        ]

```
Common usages for `vf.TasksetConfig` are settings like splits (e.g., train/test), difficulty settings, judge model names, etc.

## Adding Tools

Some environments require custom tools, which are bundled as a `vf.Toolset` (similar to how a `vf.Taskset` bundles `vf.Task`).
Tools are exposed as MCP servers to the given harness and thus need a harness which exposes MCP support (via `SUPPORTS_MCP`).

You can create them like this (remember the bootstrapping with `prime env init MY_ENV -T`):
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
```

## Using Judges

If your reward is semantic, use a LLM judge.

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


class Config(vf.TasksetConfig):
    # The judge by default inherits base_url and api keys from the prime config
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-5-mini")


class JudgeTraceTaskset(vf.Taskset[Task, Config]):
    # Build the judge lazily from config — no Taskset.__init__ override needed.
    @cached_property
    def judge(self) -> CorrectnessJudge:
        return CorrectnessJudge(self.config.judge)

    def load_tasks(self) -> list[Task]:
        return [Task(idx=0, prompt="What is 2+2?", answer="4")]

    @vf.reward()
    async def correct(self, task, trace) -> float:
        result = await self.judge.evaluate(
            trace=trace, 
            question=task.prompt, 
            answer=task.answer, 
            # give the last assistant message to the judge
            response=trace.last_reply
        )
        return 1.0 if result.parsed else 0.0
```

To override the judge model, set `taskset.judge.model` in your config (it is a string).
Sampling knobs live under `taskset.judge.sampling` — e.g. `taskset.judge.sampling.max_tokens`.
