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
- `-H`, `--add-harness` — also scaffold a custom `vf.Harness` at `harness.py`, selectable via `--harness.id <name>`
  - Prefer a built-in harness unless the model needs to run inside a custom program.

Most tasksets do not need specific tools or custom harnesses. (To simulate a user interacting with the model, open a chat session from an env's `rollout()` and script the user's turns — see the [Agent docs](agent.md).)

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

You can also use `@vf.metric` to record non-scored values, which might be useful for training.

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
arbitrary sample. Generation must be deterministic — env-server pool workers each run
their own `load()` and rely on every worker producing the same sequence, so seed any
randomness with a constant (see `alphabet_sort_v1`, `color_codeword_v1`, or the built-in
`textarena` taskset).

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

## Multi-agent environments

One eval rollout doesn't have to be one agent run. `Environment` is a concrete class
whose defaults are the single-agent case; a package can export a subclass (via
`__all__`, alongside its `Taskset` — the same plugin idiom as a bundled harness) that
declares its parameters as an `EnvParams` subclass and overrides up to three methods:

```python
class DebateParams(vf.EnvParams):
    pro: vf.AgentConfig = vf.AgentConfig()
    con: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(model="openai/gpt-5-mini", trainable=False)


class DebateEnv(vf.Environment[DebateParams]):
    def roles(self):
        """The topology: who plays which role, and what each needs. The debaters
        play the dataset; the judge grades an env-minted verdict task."""
        return {
            "pro": vf.Role(self.params.pro),
            "con": vf.Role(self.params.con),
            "judge": vf.Role(self.params.judge, mcp=False, container=False),
        }

    async def rollout(self, task, agents):
        """How the agents interact on one task: imperative Python over Agent values.
        A loop is rounds, asyncio.gather is fan-out, a function from traces to task
        data is chaining. The returned traces are the rollout's record."""
        pro, con = await asyncio.gather(
            agents["pro"].run(task), agents["con"].run(task)
        )
        verdict = await agents["judge"].run(judge_task(task, pro, con))
        return [pro, con, verdict]

    async def score(self, task, traces):
        """Sibling-dependent judgement over the finished set (per-trace judgement
        already ran on each trace's own task). Attach via record_reward/record_metric."""
        pro, con, verdict = traces
        pro.record_reward("won", float("pro" in verdict.last_reply))
        con.record_reward("won", float("con" in verdict.last_reply))
```

- **Roles are typed fields on the env's params block** (`Environment[DebateParams]`
  binds it; `self.params` reads it), so the CLI addresses them for free:
  `--env.pro.model ...`, `--env.judge.client.base_url ...`, `--env.con.max_turns 4` —
  the framework narrows the `env` field by taskset id exactly as it narrows
  `taskset`/`harness`, and a partial override deep-merges with the declared role
  default (`--env.judge.sampling.temperature 0` doesn't reset the judge's pinned
  model). An `AgentConfig`'s every field defaults to the run's own settings —
  `AgentConfig()` is "the policy under evaluation/training" (which is what makes
  self-play trainable); a role pins only what makes it a different actor (its own
  harness, a frozen model, an off-train endpoint, tighter limits, `trainable=False`).
- **A role declares what it needs from the taskset's world.** `vf.Role(cfg)`
  plays the dataset: the taskset's needs apply (declared tools mean the role's
  harness must support MCP; `NEEDS_CONTAINER` means no subprocess runtime), and the
  role is handed the taskset's shared tool servers. A role whose tasks the env
  mints itself says so — `vf.Role(cfg, mcp=False, container=False)` for a bare
  model actor like a judge or a simulated user — and then pairs with *any* taskset.
  Keeping the declaration honest with `rollout()` is the env author's job;
  `Agent.run` still validates every concrete task it's given, as the backstop.
- **The base builds the agents** — one per role, inside the eval's serving resources
  (shared interception pool, shared tool servers, per-endpoint clients) — and hands
  them into `rollout()`. The hook never constructs agents.
- **One env-rollout is one `RolloutRecord`** on the wire (`traces.jsonl`, the serve
  protocol): the task, a rollout-level `errors` list, and one trace per agent run,
  each stamped with its `role` and `trainable`. Records succeed, resume, and retry
  as a unit. An agent failure is data on its trace (the hook decides what a failed
  participant means); an exception in `rollout()`/`score()` is the env-rollout
  failing, and every trace that completed before it is still captured on the record.
- `score()` is bounded by `--timeout.score`; `setup()`/`teardown()` hooks bracket the
  serving lifetime for env-owned shared resources.

For the single-agent case none of this is visible: the base `roles()` is one `"solver"`
role driven by `--harness.*`, `rollout()` is `[await agents["solver"].run(task)]`, and
the record wraps exactly one unstamped trace.

The three axes of a run are orthogonal:

- **taskset** — *what to solve*: the rows, their data, their per-trace judgement.
- **harness** — *how the LLM interfaces with the world*: the program driving model
  calls, tools, a runtime.
- **env** — *the control flow between agents*: who runs, in what order, judged how
  across the finished set.

### Reusable envs: `--env.id`

An interaction pattern that isn't specific to one dataset — n attempts, a judge, a
modeled user — is its own plugin, paired with any taskset from the CLI:

```bash
uv run eval --taskset.id gsm8k-v1 --env.id best-of-n --env.n 8
uv run eval --taskset.id my-task-v1 --env.id judge --env.judge.model openai/gpt-5-mini
```

`--env.id` resolves like every plugin id — a bundled env (below), a local package
exporting an `Environment` subclass via `__all__`, or a Hub `org/name[@version]` —
and its `EnvParams` surface typed on the CLI (`--env.<role>.*`, `-h` renders them).
Empty (the default) keeps the taskset's own story: the env its package ships (a
*recipe* env like `code_golf_v1`, where the interaction is intrinsic to the data),
else the single-agent base. An explicit id wins over a bundled recipe env.

Bundled envs (`verifiers/v1/envs/`):

| id | roles | what it does |
| --- | --- | --- |
| `best-of-n` | `solver` | `--env.n` independent attempts per rollout; `score()` marks the argmax-reward sibling (`best`) and whether any reached `--env.threshold` (`pass_at_n`) — rejection sampling and pass@k. |
| `judge` | `solver`, `judge` | the solver plays the task; a judge agent (in-process `direct` harness, `trainable=False` by default) grades the finished attempt. The verdict spec is a **judge plugin** (`--env.spec.id score\|rubric\|reference`, same registry and format as `taskset.task.judges`) — write your grading criteria once, run them as a bare call or as an agent. Verdict + per-criterion metrics land on the solver's trace; point `--env.judge.harness.id` at a real harness and the judge investigates with tools, `--env.spec.view full_trace` shows it the whole transcript. |
| `user-sim` | `assistant`, `user` | a modeled user (direct harness, untrainable) opens and drives the conversation from the task's prompt-as-scenario (`--env.persona`); the assistant plays the same task through a masked chat session (`mask_prompt`) — the prompt is hidden from its harness while the task's own rewards and judges still score the real row. The substrate for tau-bench-style evals. |

### User simulation: the user is just another agent

There is exactly one exchange mechanism: the chat session (`agents[...].chat(task)`)
— whoever calls `turn()` is the run's user, and each turn runs one harness segment
(the program runs until it yields, the caller answers its final message, the next
segment resumes the exchange with the answer). A prompt-less task is opened by the
first `turn(message)`; a prompted task speaks first (take its opening reply with a
bare `turn()`). Who computes the turns is the env's control flow, not framework
machinery:

```python
class SortEnv(vf.Environment):
    async def rollout(self, task, agents):
        # a pre-scripted episode: the task is prompt-less, so the first turn opens
        async with agents["solver"].chat(task) as session:
            for prompt in task.data.info["user_turns"]:
                if (await session.turn(prompt)).stopped:
                    break                        # a limit or @stop ended the run
        return [session.trace]
```

A *scripted* user is a plain loop like this (a game engine stepping in-process works
the same way — see the bundled `textarena` taskset). A *modeled* user is another agent
role: open both sessions and relay their `turn()`s into each other — see the bundled
`user-sim` env, and [chat() in the Agent docs](agent.md). The user runs in the eval
process, so there is nothing to declare, place, or serve.
