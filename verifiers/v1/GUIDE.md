# verifiers.v1 — user guide

How to build and run a v1 environment end-to-end. For the one-page tour of what v1 is and
why, see the [README](README.md); this guide is the longer-form how-to. `import verifiers.v1 as vf`.

## Mental model

A v1 environment is three decoupled pieces, each packagable and configured through typed
config. You'll likely work with them in different proportions:

- **Taskset** — the data and the scoring: it produces typed `Task`s and owns every `@reward` /
  `@metric`, plus any tools and a user simulator (*what* the model is asked and *how* it's
  graded). For many environments, this is the only piece you write.
- **Harness** — the program that drives the rollout turn to turn, a chat loop or an agent CLI
  (*how* the model is called). **Usually you just pick a built-in** (`default` / `rlm` /
  `codex`); you only write your own if you need a custom rollout loop. With some exceptions, any
  taskset runs under any harness.
- **Runtime** — *where* the harness (and the taskset's tools / user simulator) executes:
  `subprocess` / `docker` / `prime` / `modal`. **You never write one** — runtimes ship with the
  framework behind one `Runtime` contract and compose with any taskset/harness; you just choose
  where code runs.

The output of a rollout is a `Trace` — the full record of the agent's trajectory: every
message plus all the metadata captured along the way (token ids, logprobs, tool calls,
rewards, timing). The same `Trace` is what eval scores, the dashboard renders, and prime-rl
trains on.

## Quickstart

```bash
uv sync                          # core + shipped packages + examples
uv run eval gsm8k-v1 -n 5 -r 3   # 5 tasks, 3 rollouts each; default harness, subprocess runtime
uv run eval -h                   # typed help (lists local tasksets + harnesses)
```

Everything below has a CLI flag *and* a TOML equivalent (`uv run eval @ config.toml`); the
flag names are the dotted config path (`--harness.runtime.type docker`). See the
[CLI reference](#cli-reference) for the full command surface.

---

# Authoring a taskset

A taskset is a package exporting a `vf.Taskset`. Scaffold one with `uv run init my-task-v1` (add
`--add-tool` / `--add-user` / `--add-harness` for more pieces, `--v0` for a legacy environment),
or copy the closest `environments/<name>_v1` and edit. The minimal shape is:

```python
import verifiers.v1 as vf


class ReverseTask(vf.Task):
    answer: str                     # your own fields, alongside vf.Task's (prompt, system_prompt, ...)


class ReverseConfig(vf.TasksetConfig):
    num_tasks: int = 100            # knobs surface as --taskset.num-tasks


class ReverseTaskset(vf.Taskset[ReverseTask, ReverseConfig]):
    def load_tasks(self) -> list[ReverseTask]:
        return [
            ReverseTask(idx=i, prompt=f"Reverse: {w}", answer=w[::-1])
            for i, w in enumerate(WORDS[: self.config.num_tasks])
        ]

    @vf.reward()
    async def exact_match(self, task: ReverseTask, trace: vf.Trace) -> float:
        return float(trace.last_reply == task.answer)


__all__ = ["ReverseTaskset"]   # vf resolves the taskset by finding this Taskset subclass
```

`vf.Taskset[TaskT, ConfigT, StateT]` is generic over three types: your `Task` subclass, your
`TasksetConfig` subclass, and (optionally) your `State` subclass. The third defaults to the base
`vf.State`, so a stateless taskset writes just `Taskset[MyTask, MyConfig]`. The framework reads
these off the generic bases to type `self.config`, `trace.task`, and `trace.state`.

The taskset module must export its `Taskset` subclass via `__all__` — the loader walks the
exported names and finds the single `Taskset` subclass.

**Capability flag.** A taskset has one class var, `NEEDS_CONTAINER` (default `False`); set it `True`
to declare the taskset only runs in a container runtime (`docker` / `prime`), so the framework
refuses the subprocess runtime up front — the taskset-wide counterpart to a task's per-row `image`
(see [Runtimes](#runtimes)).

## The task

`vf.Task` is a frozen pydantic model. Subclass it to add typed, task-specific fields (the
reference answer, ground truths, per-row metadata) that flow — fully typed — into your rewards as
`trace.task`. The base fields every task has:

| field | type | meaning |
| --- | --- | --- |
| `idx` | `int` *(required)* | stable index within the taskset |
| `name` | `str \| None` | human-readable label (display / filtering) |
| `description` | `str \| None` | human-readable description |
| `prompt` | `str \| Messages \| None` *(required)* | the opening user message. A `str` is the usual case; a `Messages` list seeds a full initial conversation (e.g. a user message carrying images — only harnesses with `SUPPORTS_MESSAGE_PROMPT`); **`None` means the task carries no prompt** and the user simulator opens the conversation (see [User simulators](#user-simulators)) |
| `system_prompt` | `str \| None` | system prompt; emitted as a real system message by harnesses that set `APPENDS_SYSTEM_PROMPT`, else folded into `prompt` |
| `image` | `str \| None` | container image the task needs — forces a container runtime (the subprocess runtime is refused) |
| `workdir` | `str \| None` | working directory the harness and scoring run in |
| `timeout` | `TaskTimeout` | per-task, per-stage wall-clock overrides |
| `resources` | `TaskResources` | per-task runtime resource requests |

`prompt` is *required* on purpose — set it to `None` explicitly to opt into a user-sim-opened
conversation rather than forgetting it.

**Per-task overrides.** `timeout` and `resources` are small frozen submodels that let a single
row override the eval-wide defaults (precedence is always `cli/toml > task > default`):

```python
from verifiers.v1 import TaskTimeout, TaskResources

MyTask(
    idx=0, prompt=...,
    timeout=TaskTimeout(setup=300, harness=1200, scoring=120),   # seconds; per stage, None = no limit
    resources=TaskResources(cpu=4, memory=8, gpu="A100:2", disk=20),  # None = runtime default
)
```

`TaskTimeout` has `setup` / `harness` / `finalize` / `scoring`; `TaskResources` has `cpu` /
`memory` (GB) / `gpu` (`"type[:count]"`) / `disk` (GB). A field the runtime doesn't support is
warned about and ignored. SWE-style tasksets typically set these per row from dataset metadata.

## The config

A `vf.TasksetConfig` subclass is the taskset's typed knobs. Its fields become `--taskset.<field>`
CLI flags (and TOML keys), and the instance reaches the taskset as `self.config`:

```python
class GSM8KConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"   # --taskset.split test
```

Nested configs nest the flag path: a `tools: vf.ToolsetConfig` field is set with
`--taskset.tools.shared true` / `--taskset.tools.runtime.type docker`. Keep **fixed** data
(prompt templates, lookup tables) in module constants; config is for things a *runner* should be
able to change. The base `TasksetConfig` carries `id` (the taskset's id, set via `--taskset.id`).

## Loading tasks

`def load_tasks(self) -> list[TaskT]` builds the task list. It runs **once at load** (not per
rollout), so do dataset loading / filtering / slicing here off `self.config`. Return your typed
`Task` subclass instances.

```python
class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load_tasks(self) -> list[GSM8KTask]:
        rows = load_dataset("openai/gsm8k", "main", split=self.config.split)   # read knobs off self.config
        return [
            GSM8KTask(
                idx=i,
                prompt=row["question"],
                answer=row["answer"].split("####")[-1].strip(),
            )
            for i, row in enumerate(rows)
        ]
```

## Scoring — rewards, metrics, group rewards

Rewards and metrics are decorated `async` methods. The framework **injects whichever arguments
you name** — declare any subset of `task` / `trace` / `runtime` and you get exactly those:

```python
@vf.reward(weight=1.0)                 # summed (weighted) into trace.reward — a float or a dict to merge
async def correct(self, task, trace) -> float: ...

@vf.metric()                           # recorded, not summed — a float or a dict to merge
async def enthusiasm(self, trace) -> float:

@vf.group_reward(weight=0.1)           # compares a task's N rollouts — here, a length penalty
async def brevity(self, traces: list[vf.Trace]) -> list[float]:
```

The decorators and what each can receive:

| decorator | params | optional kwargs | returns |
| --- | --- | --- | --- |
| `@vf.reward` | `task`, `trace`, `runtime` | `weight=1.0`, `priority=0` | `float`, **or a `dict[str, float]`** (each × weight → summed into `trace.reward`) |
| `@vf.metric` | `task`, `trace`, `runtime` | `priority=0` | `float`, **or a `dict[str, float]`** merged into `trace.metrics` |
| `@vf.group_reward` | `task`, `traces` | `weight=1.0`, `priority=0` | `list[float]`, one per trace |
| `@vf.stop` | `trace` | `priority=0` | `bool` |

Good to know:

- **`@group_reward` gets no `runtime` and no single `trace`** — only `task` and `traces` (it runs
  after the per-rollout runtimes are gone). To compare a runtime-derived signal across a task's
  rollouts, record it per-rollout as a `@metric`/`@reward` first, then read it off each trace in
  the group reward. Group rewards need `-r/--num-rollouts ≥ 2`.
- **`priority`** orders execution within a kind (higher first, then by name). It mostly matters for
  `@stop` — the highest-priority stop that fires sets the stop reason.

### Reading the trace

A reward reads the finished trajectory off `trace`. The most useful members, by area:

**Task & messages**

| member | type | what |
| --- | --- | --- |
| `trace.task` | `TaskT` | the typed task (your subclass) |
| `trace.assistant_messages` | `list[AssistantMessage]` | the model's responses in order (excludes prompt-supplied messages) |
| `trace.tool_messages` | `list[ToolMessage]` | tool results (main branch) |
| `trace.branches[-1].messages` | `Messages` | the full conversation of the main (last) branch |

**Carried state** (set during the rollout / `finalize`, read back here)

| member | type | what |
| --- | --- | --- |
| `trace.info` | `dict` | free-form persisted artifact bag (see [Persisted info](#persisted-info)) |
| `trace.state` | `StateT` | transient per-rollout state (see [State](#per-rollout-state)) |

**Status & lifecycle**

| member | type | what |
| --- | --- | --- |
| `trace.is_completed` / `trace.stop_condition` | `bool` / `str \| None` | whether / why the rollout ended |
| `trace.is_truncated` | `bool` | ended by hitting a turn/token/length cap |
| `trace.has_response` | `bool` | the last turn produced non-empty content |
| `trace.has_error` / `trace.error` / `trace.errors` | `bool` / `Error \| None` / `list[Error]` | error state (most recent / all attempts) |

**Counts, tokens & timing**

| member | type | what |
| --- | --- | --- |
| `trace.num_turns` | `int` | sampled model turns |
| `trace.num_branches` / `trace.branches` | `int` / `list[Branch]` | branch count / the branches (>1 under compaction or subagents) |
| `trace.prompt_len` / `trace.completion_len` / `trace.total_tokens` | `int` | token counts (summed over branches) |
| `trace.usage` | `Usage \| None` | provider-reported token usage |
| `trace.timing` | `Timing` | per-stage durations |
| `trace.id` | `str` | unique rollout id |

`trace.reward` / `trace.rewards` / `trace.metrics` are scoring *outputs*, filled in during the
scoring pass — don't read them from inside a `@reward`/`@metric`; a `@group_reward` reads metrics
off each finished trace instead.

### In-runtime scoring

Declare `runtime` on a `@reward`/`@metric` when scoring needs the rollout's runtime — either
because it requires **heavy computation that shouldn't run on the host** (e.g. a verifier with its
own dependencies like `math-verify`), or because it needs **information that only lives in the
agent's runtime** (files the agent wrote, command output, container state). The `runtime` object
gives you read/write/exec in there; a common pattern is a uv script whose PEP 723 deps resolve
inside the runtime and never touch the host:

```python
VERIFY = (Path(__file__).parent / "verify.py").read_text()   # PEP 723 header declares its deps

@vf.reward()
async def verify(self, task, trace, runtime) -> float:
    r = await runtime.run_uv_script(VERIFY, args=[task.answer, trace.last_reply])
    return float(r.stdout.strip() == "1.0")
```

### Judges

When grading can't be deterministic, `vf.Judge` is a reusable LLM judge: it owns the OpenAI client
(with the Prime key/team fallback), the call, and usage/cost capture, and leaves the two things that
differ as hooks — `build_messages` (prompt) and `parse` (verdict). Subclass it, build it once in the
taskset `__init__`, and call it from a `@reward`:

```python
import verifiers.v1 as vf

class CorrectnessJudge(vf.Judge[bool]):                 # Judge[ParsedT] — ParsedT is your verdict type
    prompt = "Question: {question}\nAnswer: {answer}\nResponse: {response}\nCorrect? Reply yes or no."

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return response.text.strip().lower().startswith("yes")

class MyConfig(vf.TasksetConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig()

class MyTaskset(vf.Taskset[MyTask, MyConfig]):
    def __init__(self, config: MyConfig) -> None:
        super().__init__(config)
        self.judge = CorrectnessJudge(config.judge)

    @vf.reward()
    async def correct(self, task, trace) -> float:
        result = await self.judge.evaluate(trace=trace, question=task.question, answer=task.answer, response=...)
        return 1.0 if result.parsed else 0.0
```

`evaluate(*, trace=None, **fields)` renders the prompt (`build_messages`), calls the model, and
parses the verdict (`parse`), returning a `JudgeResponse{text, parsed, usage}`. Passing `trace=`
**records the call onto it** — a typed record appended to `trace.info["judge"]` (for debugging) and
the call's tokens + cost added to `trace.extra_usage`, kept separate from the agent's `trace.usage`
and off the message graph (so the trainer's token math is unaffected); the eval dashboard shows the
agent's usage and `+judge` separately. The record lands even if the judge refuses, returns an empty
structured output, or `parse` raises (the request was already billed). Omit `trace` for a pure call
(e.g. in tests).

The two hooks:

| hook | default | override for |
| --- | --- | --- |
| `build_messages(**fields) -> str \| Messages` | formats the `prompt` template with the fields into one user message | a system+user / non-template prompt (return a `vf.Messages` list) |
| `parse(response) -> ParsedT` | the structured object if `schema` is set, else `response.text` | your verdict (`bool`, a grade `str`, a pydantic model, a `list[float]`, …) |

Good to know:

- **Per-task rubric** is just a field — `prompt = "{task.rubric}\n…"` with `evaluate(trace=trace, task=task, …)` (`str.format` does attribute access on a passed-in `task`). For per-task *parsing*, parse in the reward, where the task is in scope.
- **Structured outputs**: set `schema` to a pydantic model to use OpenAI structured outputs (where the provider supports it — most do); `JudgeResponse.parsed` is then the validated object. For an unsupported model, prompt for JSON and call `Model.model_validate_json(response.text)` in `parse`.
- **Multiple / dynamic calls per rollout** (e.g. one per table column): call the low-level `complete(messages, *, trace=, schema=, parse=)` directly with `vf.Messages` you build — it records each call when passed `trace`.
- **Config**: `JudgeConfig` adds `model` + `sampling` (a `JudgeSamplingConfig`) to `BaseClientConfig` (`base_url`/`api_key_var`/`headers`, Prime auto-config). CLI-overridable: `--taskset.judge.model …`, `--taskset.judge.sampling.max-tokens …`.
- **Errors propagate**: a judge API failure errors the rollout (recorded as a `TasksetError` on the trace); the OpenAI SDK already retries transient 429/5xx/connection errors.

### Plugged judges

A judge can also be **plugged from config alone** — no taskset code. The base `TasksetConfig` has a
`judges` list; each entry names a judge plugin by `id` (resolved exactly like a taskset or harness id:
a built-in, a local package, or a hub `org/name[@version]` package) and is run by `Taskset.score`
after the taskset's own `@reward`s, recording its verdict into `trace.rewards` under its `name`
(default: the id's package name; entries that would share a reward key are rejected at config
time — set a distinct `name` on each) with its `weight`. That makes off-the-shelf grading
composable with **any** taskset × harness pair straight from the eval TOML:

```toml
[taskset]
id = "gsm8k-v1"

[[taskset.judges]]                       # reference-answer yes/no -> 1/0
id = "binary"
answer_field = "answer"                  # the task field holding the reference answer

[[taskset.judges]]                       # rubric criteria, each scored 1/0
id = "rubric"
path = "rubrics/quality.toml"
weight = 0.5                             # the aggregate's weight in trace.reward
weights = { cites_sources = 2.0 }        # per-criterion override of the file's weight
```

The built-ins (in `verifiers.v1.judges`):

| id | class | grades |
| --- | --- | --- |
| `binary` | `vf.BinaryJudge` | the reply against the task's reference answer (`answer_field`; a list = any acceptable answer), 1/0. Knobs: `choices` (verdict labels, e.g. `["A", "B"]`) |
| `rubric` | `vf.RubricJudge` | each criterion in a `.toml`/`.json` rubric file 1/0; rewards the weighted mean (`Σwv/Σw`), records each verdict as a `<name>/<criterion>` metric |

A rubric file lists `[[criteria]]` entries (`name`, `text`, optional `weight`; JSON takes
`{"criteria": [...]}` or a bare list) — criterion weights come from the file, overridable per name
via the config's `weights`:

```toml
[[criteria]]
name = "cites_sources"
text = "The response cites at least one specific source."
weight = 1.0
```

Good to know:

- **Judges alone are a full reward signal.** `score` runs `@metric`s, `@reward`s, then plugged
  judges — each phase optional. A taskset with no `@reward` and one judge gets `trace.reward`
  entirely from the judge (wiki-search-v1's shape); mixing them sums weighted contributions.
- **What the built-ins see** is config-selectable: `{question}` = the task field named by
  `question_field` (else `task.prompt_text` — the prompt as plain text, `Messages` reduced to
  text parts); `{response}` = the `view` (`vf.JudgeView`): `last_reply` (binary's default —
  grade the answer) or `full_trace` (rubric's default — grade the process via
  `Trace.transcript`, every recorded turn minus reasoning). Anything beyond that is a custom
  judge: `score` gets the full `trace` (and `runtime`) and builds any messages from it.
- **Error attribution:** a *model* failure scores 0 (empty replies short-circuit without a
  judge call; wrong/non-committal replies get the negative verdict). A *judge* failure — API
  error, refusal, unparseable verdict, bad rubric — **raises**: the rollout errors and training
  excludes/retries the sample. Never silently 0 a judge failure.
- **What lands on the trace:** the weighted verdict in `trace.rewards[<reward key>]`; rubric's
  raw per-criterion verdicts in `trace.metrics`; every call's `JudgeResponse` in
  `trace.info["judge"]` with its tokens + cost in `trace.extra_usage` (shown as `+judge` in the
  dashboard). All persisted to `results.jsonl`.

Writing your own pluggable judge is the same recipe as any plugin: a package exporting a `Judge`
subclass via `__all__` that implements `score` — declare any subset of `task` / `trace` / `runtime`
by parameter name (like a `@reward`) and return a `float` (or a `dict[str, float]`). Declare its
config via the second generic param so `--taskset.judges` narrows to it:

```python
class MyJudgeConfig(vf.JudgeConfig):
    strictness: int = 3

class MyJudge(vf.Judge[float, MyJudgeConfig]):
    prompt = "…{question}…{response}… Reply yes or no."

    async def score(self, task: vf.Task, trace: vf.Trace) -> float:
        result = await self.evaluate(trace=trace, question=task.prompt_text, response=trace.last_reply)
        return float(vf.parse_judge_choice(result.text, ("yes", "no")) == "yes")

__all__ = ["MyJudge", "MyJudgeConfig"]
```

Every judge's endpoint, model, sampling, prompt template (`prompt` inline, or `prompt_file`
pointing at a text file with the same `{field}` placeholders), reward key (`name`), and
`weight` are config fields, so a plugged judge is fully tunable per eval without touching code.

## Stop conditions

A rollout ends when the harness finishes, a framework budget trips (`--max-turns`, token caps), or
a taskset `@vf.stop` fires. A stop is an `async (self, trace) -> bool` checked between turns; its
**method name becomes the stop reason**:

```python
@vf.stop
async def saw_answer(self, trace) -> bool:
    last = trace.last_reply
    return "FINAL:" in last
```

The framework has no built-in "the task is done" signal — multi-turn tasksets end either from the
trace (above) or from per-rollout state set by a tool / user sim (see [State](#per-rollout-state)).

## Lifecycle hooks

A rollout runs **`setup → harness → finalize → scoring`**. A taskset can hook any stage:

| hook | signature | when | gets runtime? |
| --- | --- | --- | --- |
| `setup` | `(self, task, runtime)` | per-task prep before the harness (clone a repo, start a service) — the trace doesn't exist yet | ✓ |
| `finalize` | `(self, task, trace, runtime)` | after the harness, before scoring — apply a diff, snapshot, scrape artifacts into `trace.info` | ✓ |
| `tools` | `(self, task) -> list[vf.Toolset]` | per task, before the harness — the task's tool servers | ✗ |
| `user` | `(self, task) -> vf.User \| None` | per task, before the harness — the user simulator | ✗ |

`setup`/`finalize` errors fail the rollout legibly (captured onto the trace, not a crash).

## Runtime access

Most hooks run *with the live runtime* and can execute in it, so the whole rollout shares one
isolated environment. On a `runtime` you can call:

| method | what |
| --- | --- |
| `run(argv, env)` | exec a command to completion → `ProgramResult(exit_code, stdout, stderr)` |
| `run_uv_script(src, args, env)` | run a PEP 723 script (inline deps resolve in-runtime); `args` are shell-`"$@"`-safe |
| `run_background(argv, env, log)` | start a long-lived process (e.g. a colocated server) |
| `read(path)` / `write(path, data)` | workspace files (bytes), across the container/sandbox boundary |
| `expose(port)` | publish a port *inside* the runtime to a host-reachable URL (`None` when local) |

A non-zero `exit_code` is a normal result, not an exception — check it and `raise` (a plain
Python error) yourself if it should fail the stage; the framework records a failure in your
taskset code as a `TasksetError`. The same code works on subprocess / docker / prime / modal.

A SWE taskset is the canonical case: `setup` provisions the repo, the agent edits it during the
rollout, and a `@reward` runs the tests in the *same* runtime:

```python
class SWETaskset(vf.Taskset[SWETask, SWEConfig]):
    NEEDS_CONTAINER = True   # the only Taskset class var: refuse the subprocess runtime

    async def setup(self, task, runtime) -> None:
        await runtime.run(["git", "clone", task.repo_url, "/repo"], {})
        await runtime.run(["git", "-C", "/repo", "checkout", task.base_commit], {})

    async def finalize(self, task, trace, runtime) -> None:
        diff = await runtime.run(["git", "-C", "/repo", "diff"], {})
        trace.info["diff"] = diff.stdout   # scrape the agent's diff off the live runtime

    @vf.reward()
    async def tests_pass(self, task, trace, runtime) -> float:
        result = await runtime.run(["bash", "-lc", task.test_cmd], {})
        return 1.0 if result.exit_code == 0 else 0.0
```

## Persisted info

`trace.info` is a free-form, **JSON-serializable** dict for per-rollout artifacts that are neither a
reward nor a metric — the diff above, captured logs, command output, file paths. Write to it from
`finalize` or a `@reward`/`@metric` by assigning into the dict; it is persisted with the trace
(dumped to `results.jsonl` and sent over the wire), so every value must be JSON-serializable — a
non-serializable value fails the trace dump rather than being silently dropped.

```python
trace.info["build_log"] = result.stdout
```

It pairs with [`trace.state`](#per-rollout-state) — the two per-rollout stores are opposites:

| | `trace.info` | `trace.state` |
| --- | --- | --- |
| lifetime | **persisted** (dumped + sent over the wire) | **transient** (never dumped or sent) |
| for | artifacts to inspect after the run | live state the rollout reads and acts on |
| shape | free-form `dict[str, Any]`, JSON-serializable | typed `vf.State` subclass |
| written by | `finalize` / `@reward` / `@metric` | tools / user sim (`self.state`) + scoring |

## Per-rollout state

`trace.state` is the complementary **transient** store: a typed, mutable `vf.State` shared across
the rollout's tool servers, user simulator, and scoring — the one place per-rollout *runtime* state
lives (counters, game progress, your own end-of-trajectory flag). Unlike [`info`](#persisted-info) it is **never**
persisted to disk or sent over the wire. Subclass `vf.State` to declare typed fields (each needs a
default) and parameterize the taskset and any stateful server on it:

```python
class GameState(vf.State):
    game_over: bool = False

class GameUser(vf.User[vf.UserConfig, GameState]):
    async def respond(self, message: str) -> vf.Messages:
        ...
        if finished:
            self.state.game_over = True   # the @vf.stop below ends the rollout
        return [{"role": "user", "content": reply}]

class GameTaskset(vf.Taskset[GameTask, GameConfig, GameState]):
    @vf.stop
    async def game_over(self, trace) -> bool:   # stop reason is this method's name
        return trace.state.game_over
```

Who touches it how:

- A `@vf.tool` / `respond` reads+writes it as `self.state` — synced over the interception server
  per call, so tools and the user sim see each other's writes.
- `@reward` / `@metric` / `finalize` / `@stop` read+write `trace.state` directly.

> **Concurrency — `self.state` is last-write-wins.** Each `@vf.tool` / `respond` call syncs the
> *whole* state as a read-modify-write: pull `self.state` from the host, run, push it back. Tool
> calls a harness runs **concurrently** (several `tool_calls` in one assistant turn) therefore race
> — each reads the same starting state and the last push wins, so concurrent increments/appends are
> lost. Tools the harness runs **sequentially** compose correctly. Keep shared-state mutations on
> the sequential path. The taskset and its servers must share **one** `State` subclass — a server
> pushing a mismatching shape is rejected (the rollout fails legibly).

## Tools

A tool server is a **vf-native class** that wraps an **MCP server** — authored from a config, the
same shape as a taskset. Define `@vf.tool` methods on a `vf.Toolset[ConfigT]` (or
`vf.Toolset[ConfigT, StateT]` for one that shares state); the framework serves them as MCP tools the
harness connects to, so the model sees `<TOOL_PREFIX>_<method>` and the docstring is the description:

```python
class GlossaryToolset(vf.Toolset[GlossaryToolsetConfig]):
    TOOL_PREFIX = "glossary"            # model sees glossary_lookup (empty → class name snake-cased)

    async def setup(self) -> None:
        self.facts = load_facts()       # task-agnostic, runs once per server process

    @vf.tool
    def lookup(self, name: str) -> str:  # typed params the model fills; docstring → description
        """Look up a glossary term."""
        return self.facts.get(name.lower(), "unknown")


if __name__ == "__main__":
    GlossaryToolset.run()                # self-launching module under servers/ (see below)
```

Two setup hooks, plus the shared state:

- `async def setup(self)` — task-agnostic, runs for **every** server (shared or per-rollout); build
  expensive global data as plain `self.x` attributes here.
- `async def setup_task(self, task)` — per-rollout init off `task`; **skipped for a `shared`
  server** (warns if defined there).
- `self.config` is the server's typed knobs; `self.state` is the shared per-rollout `State`.

A taskset exposes a task's tools via `tools(task) -> list[vf.Toolset]`, constructing each from a
config field so placement is CLI-tunable (below).

## User simulators

A user simulator is a `vf.User[ConfigT]` (or `[ConfigT, StateT]`) with one hook:

```python
class HagglerUser(vf.User[vf.UserConfig, HagglerState]):
    async def respond(self, message: str) -> vf.Messages:
        # the model's last assistant text in → the next user message(s) out ([] to emit nothing)
        if done:
            self.state.deal_closed = True       # end via a @vf.stop the taskset declares
        return [{"role": "user", "content": reply}]


if __name__ == "__main__":
    HagglerUser.run()                           # self-launching module under servers/ (see below)
```

The framework calls `respond` after each agent turn and injects the reply as the next user
message; it's consumed by the framework, never shown to the model. A taskset supplies one via
`user(task) -> vf.User | None`. If a task carries **no prompt** (`prompt=None`), the simulator also
**opens the conversation**: the framework calls `respond("")` once before the first model turn and
seeds its reply as the initial user message. End the trajectory by setting a `self.state` flag a
taskset `@vf.stop` checks (there's no built-in end signal).

A taskset may expose **both** tools and a user sim at once — they're served together each rollout;
the harness just needs to support both.

## Server placement & isolation

Each tool/user server is its own self-launching module under the env package's `servers/`, ending
with `if __name__ == "__main__": <Server>.run()`; the framework launches it with
`python -m <env>.servers.<name>`. **Placement** lives on the server's config (a `vf.ToolsetConfig`
/ `vf.UserConfig` field on the taskset config), so it's per-server and CLI-tunable
(`--taskset.tools.shared true`). It decides **where** the server runs and **how many** there are —
the default is the cheapest correct thing; the rest trade setup cost for isolation:

| mode | config | runs | pros | cons |
| --- | --- | --- | --- | --- |
| **own runtime** *(default)* | *(nothing; `runtime = {type = "docker"\|"prime"}` for a sandbox)* | own runtime per rollout — `subprocess` on the host (default), or a `docker`/`prime` sandbox over a tunnel | full per-rollout isolation; a sandbox also isolates untrusted code / deps / network | pays `setup` every rollout (a sandbox adds spin-up + env install) |
| **colocated** | `colocated = true` | inside the harness's runtime, one per rollout (no tunnel) | no extra runtime/tunnel; can touch the harness's filesystem | couples to the harness; `setup` per rollout |
| **shared** | `shared = true` | one instance for the whole eval | `setup` once; writable per-rollout if state lives in `self.state` | state outside `self.state` corrupts across rollouts; `setup_task` skipped |
| **remote** *(tools only)* | `url = "https://…"` | connects to an already-running MCP endpoint | zero hosting; use a public/third-party server | no isolation, state, or lifecycle control |

`shared` works on any runtime — the framework makes the rollout's `/state` channel reachable from the
shared server (localhost, or a host tunnel when remote). Keep big shared data off the Python heap
(numpy / mmap / an on-disk index) so one instance scales across rollouts.

**Choosing per-rollout state:** read-only resource → `shared`; state that fits a `State` model →
`shared` + `self.state` (no extra process, the scalable default); state that can't (module globals,
on-disk scratch, a stateful C library) → a per-rollout placement (own runtime or `colocated`) that
pays `setup` each time; cheap `setup` → just use the default. **User simulators** support only the
per-rollout placements (own runtime or `colocated`); `shared` / `url` are tools-only.

## Learn from the examples

The `*_v1` tasksets under `environments/` are the reference library — each shows one pattern:

| example | pattern |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1` | single-turn + in-runtime scoring (a `@reward` uv script) |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | multi-turn, stateful, driven by a `vf.User` simulator |
| `glossary-v1` | the simplest tool server (own host runtime) |
| `wiki-search-v1` | a shared, read-only tool server (built once) + an LLM judge |
| `scratchpad-v1` | a shared, **writable** tool server — per-rollout state isolated via `self.state` |
| `deepwiki-v1` | an existing remote tool server, by URL |
| `color-codeword-v1` | a multimodal (image) task |
| `wordle-v1` | a thin config over the shipped `textarena` integration |

---

# Authoring a harness

If you need to customize the rollout logic beyond what the built-in harnesses provide — a loop they
can't express (context compaction, subagents, a bespoke agent CLI) — author a custom harness.
Otherwise pick a built-in, selected with `--harness.id`:

| id | what it is |
| --- | --- |
| `default` | a `bash` + `edit` coding agent (`edit` on by default — `--harness.edit false` for bash-only; `--harness.search true` adds a serper.dev `search` tool) — the fallback when no harness is given |
| `null` | a tiny OpenAI chat loop (MCP tools only, no tools of its own) |
| `rlm` | the RLM CLI agent |
| `codex` | the Codex CLI (Responses dialect + SSE relay) |
| `mini-swe-agent` | the mini-swe-agent CLI (a minimal SWE agent) |
| `kimi-code` | the Kimi Code CLI agent |

```bash
uv run eval gsm8k-v1 -n 1                    # default harness (bash + edit; the fallback)
uv run eval gsm8k-v1 -n 1 --harness.id null  # bare chat loop, no local tools
uv run eval gsm8k-v1 -n 1 --harness.id rlm   # same taskset, different driver
```

## Capability flags

Class vars on the harness gate which tasksets it can run, so an incompatible pairing fails fast at
load instead of mis-running:

| flag | default | gates |
| --- | --- | --- |
| `SUPPORTS_MCP` | `False` | exposes the task's MCP tools to the model (opt in: set `True` for a harness with an MCP client) |
| `SUPPORTS_USER_SIM` | `False` | drives a task's user simulator (multi-turn user injection) |
| `SUPPORTS_MESSAGE_PROMPT` | `False` | accepts a `Messages`-list `task.prompt` (e.g. image-bearing) |
| `APPENDS_SYSTEM_PROMPT` | `False` | emits `task.system_prompt` as a real system message (else it's folded into the user prompt with a warning) |

## Writing one

Define a `HarnessConfig` (any knobs surface as `--harness.*`), subclass `vf.Harness[ConfigT]`,
declare the capability flags, and implement `launch`. Export the class via `__all__`.

```python
import verifiers.v1 as vf

PROGRAM = (Path(__file__).parent / "program.py").read_text()  # a uv script, deps = ["openai"]


class MyHarnessConfig(vf.HarnessConfig):
    """Run knobs for this harness (surface as --harness.*)."""


class MyHarness(vf.Harness[MyHarnessConfig]):
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls) -> vf.ProgramResult:
        system, prompt = self.resolve_prompt(trace.task)
        # configure the program with CLI args, not OPENAI_* env vars (less footgun-prone)
        args = [f"--base-url={endpoint}", f"--api-key={secret}", f"--model={ctx.model}"]
        if system:
            args.append(f"--system-prompt={system}")
        if mcp_urls:                 # standard mcpServers map the program connects to
            args.append("--mcp-config=" + json.dumps({"mcpServers": {n: {"url": u} for n, u in mcp_urls.items()}}))
        if prompt is not None:
            args.append(f"--prompt={prompt}")
        return await runtime.run_uv_script(PROGRAM, args=args, env=self.config.env)


__all__ = ["MyHarness"]
```

### The contract

A harness never builds the trace itself: it just points *a program* at `endpoint` (authorized with
`secret`), and the interception server records every model call — **as long as the program makes its
requests in one of the supported dialects** (chat-completions, Responses, Anthropic Messages). The
program can be any executable the runtime can run.

### The `launch` hook

`launch` receives:

- `ctx` — the rollout's collaborators; read `ctx.model` for the model id (`ctx.client` /
  `ctx.sampling` exist but model calls flow through `endpoint`, not the client object).
- `trace` — the task is `trace.task`.
- `runtime` — where to run the program.
- `endpoint` / `secret` — the interception server URL and its bearer token.
- `mcp_urls` — the task's tool servers as `{name: url}` to wire in.

It must return a `vf.ProgramResult` (`exit_code`, `stdout`, `stderr`) — usually the return of
`runtime.run(...)` / `runtime.run_uv_script(...)`. The base `run` wraps `launch`: a clean exit
becomes `trace.stop("agent_completed")`; a non-zero exit (or an unexpected exception from `launch`)
raises `HarnessError` with the tail of stderr — **unless** a `@stop` already fired (the program
dying because the interception server cut a turn is expected, not an error).

### `resolve_prompt`

`resolve_prompt(trace.task)` returns `(system_prompt, prompt)` already reconciled with your
capability flags: the system prompt is handed back only if `APPENDS_SYSTEM_PROMPT` (else folded
into `prompt`); `prompt` is a `str`, a `Messages` list (if `SUPPORTS_MESSAGE_PROMPT`), or `None`
(no prompt → let the user simulator / interception server open the conversation, so send no opening
user message).

### Program styles

A self-contained chat loop is usually a single-file uv script (`runtime.run_uv_script`, so the
harness needs only `uv` in the runtime — its inline deps resolve there, never on the host; identical
scripts share one content-addressed uv env). An agent CLI / binary is installed and launched with
`runtime.run(...)`. Either way, pass `endpoint` / `secret` / `ctx.model` to the program as **CLI
args** (as above) rather than `OPENAI_*` env vars — an inherited or stray env var can silently
redirect the program's model calls; `self.config.env` just supplies any extra environment.

### Harness metrics

A harness can define its own `@vf.metric` methods (injected `task` / `trace` / `runtime`), run over
the finished trace alongside the taskset's — handy to surface what the program left behind in the
runtime (e.g. read a `meta.json` the binary wrote). A harness can't define rewards.

---

# Runtimes

The same `Runtime` contract backs the harness (`--harness.runtime`), a task's tools
(`--taskset.tools.runtime`), and the user simulator. You choose where code runs; you never write
one:

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # local process (eval default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # local container
uv run eval gsm8k-v1 -n 1 --harness.runtime.type prime       # remote prime sandbox (requires auth)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type modal       # remote modal sandbox (requires auth)
```

A taskset that sets `NEEDS_CONTAINER` (or a task with an `image`) refuses the subprocess runtime —
pass `docker` / `prime` / `modal`.

---

# CLI reference

Three commands, all `uv run <cmd>`: **`eval`** (run + score a model), **`validate`** (model-free
gold check), **`init`** (scaffold). They share a resolution layer:

- **Positional id** — a leading bare token is the taskset id: `eval gsm8k-v1` == `eval --taskset.id
  gsm8k-v1` (for `init` it's the env name).
- **`@ file.toml`** — load a saved config; extra flags still override (`eval @ config.toml -n 5`).
- **`-h` / no args** — print help, *narrowed to the chosen taskset/harness* so it shows their real
  `--taskset.*` / `--harness.*` fields, not the generic base.

## `eval`

Run a model rollout per task and score it: fan out `-r` rollouts per task with bounded
concurrency, score each trace, and persist everything.

```bash
uv run eval gsm8k-v1 -n 5 -r 3 \
  -m openai/gpt-5-mini \                                            # model
  --max-turns 8 --max-total-tokens 8192 \                          # per-rollout budgets
  --sampling.temperature 0 --sampling.max-tokens 2048 \            # generation knobs
  --timeout.rollout 600 --timeout.scoring 120 \                    # per-stage wall-clock caps (s)
  --retries.rollout.max-retries 3 --retries.rollout.include SandboxError  # retry a whole rollout
```

**Common flags** (each has a TOML equivalent at the dotted path):

| group | flags |
| --- | --- |
| selection | `<taskset-id>` / `--taskset.id`, `--harness.id` (`default`), `-m`/`--model` |
| counts | `-n`/`--num-tasks` (all), `-r`/`--num-rollouts` (1; `@group_reward` needs ≥2), `-s`/`--shuffle` |
| budgets | `--max-turns`, `--max-input-tokens`, `--max-output-tokens`, `--max-total-tokens` (all None) |
| sampling | `--sampling.temperature`, `--sampling.top-p`, `--sampling.max-tokens`, `--sampling.reasoning-effort` (provider keys pass through) |
| timeouts | `--timeout.setup`, `--timeout.rollout`, `--timeout.finalize`, `--timeout.scoring` (None = no limit) |
| retries | `--retries.rollout.max-retries` (0), `--retries.rollout.include`/`.exclude` (by exception name); per-call model/runtime retries are owned by the harness/runtime SDKs |
| client | `--client.type` (`eval`\|`train`), `--client.base-url`, `--client.api-key-var` |
| runtime | `--harness.runtime.type` (`subprocess`), `--harness.env`, `--harness.disabled-tools` |
| concurrency | `-c`/`--max-concurrent` (128), `--multiplex` (32), `--pool.type` (`elastic`\|`static`) |
| output | `-o`/`--output-dir`, `--dry-run`, `--no-rich`, `-v`/`--verbose` |

**Sampling.** `reasoning_effort` is a string (not a fixed enum) — the active dialect maps it to the
provider's shape (`reasoning_effort` for chat-completions, `reasoning.effort` for Responses,
`output_config.effort` for Anthropic).

**Clients.** `eval` (default) is a plain chat-completions relay. `--client.type train` tokenizes
client-side so each node carries the exact `token_ids` / `mask` / `logprobs` (the training dialect)
— point it at a vLLM engine with `--client.base-url`.

**What it writes** (into `--output-dir`, default `outputs/<taskset>--<model>--<harness>/<uuid>`):
`config.toml` (the resolved config, re-runnable via `@ config.toml`), `results.jsonl` (one full
trace per line, appended as each rollout finishes — durable mid-run), and `eval.log`.

**`--dry-run`** writes `config.toml` and exits (resolve + validate, no run). **`--resume
<output-dir>`** re-runs only the missing/errored rollouts of a previous run (it reloads that run's
`config.toml`, so it takes no other args).

## `validate`

Run each task's `validate` hook — a model-free check that the ground truth holds (the gold patch
makes the tests pass, the verifier accepts the gold answer) — in a runtime with the taskset's
`setup` applied. No model, no harness.

```bash
uv run validate gsm8k-v1 -n 20 --runtime.type subprocess
```

| flag | default | meaning |
| --- | --- | --- |
| `<taskset-id>` / `--taskset.id` | — | taskset to validate |
| `--runtime.type` | `docker` | runtime for `setup` + `validate` (a gold check often needs the task's container) |
| `--setup-timeout` / `--validate-timeout` | None | per-hook wall-clock caps |
| `-n`/`--num-tasks`, `-s`/`--shuffle`, `-c`/`--max-concurrent` (128) | | task selection + concurrency |
| `-v`/`--verbose`, `--no-rich` | | logging / disable the dashboard |

Each task is provisioned, set up, validated, and torn down independently; a raised error is
captured as a result row (one bad task is data, not a crash) with reason `valid` / `invalid` /
`timeout` / `error`. **Fire-and-forget — nothing is written to disk**; results show live. Note the
default runtime is **docker** (unlike eval's subprocess), and a subprocess runtime against a
`NEEDS_CONTAINER` / image-bearing taskset aborts with a clear error.

## `init`

Scaffold a new v1 environment package under `--path` (default `./environments`), following the
shipped `*_v1` layout: a `pyproject.toml`, a `README.md`, a package whose `__init__.py` re-exports
the plugin via `__all__`, and a `taskset.py` that runs out of the box (replace `load_tasks` and the
`@reward`).

```bash
uv run init my-task-v1                  # minimal taskset package
uv run init my-task-v1 -T -U -H         # + a tool server, a user sim, a custom harness
uv run init legacy-env --v0             # a legacy v0 load_environment package instead
```

| flag | meaning |
| --- | --- |
| `<name>` / `--name` | the new env id (`my-task-v1`); package dir, ids, class names are derived from it |
| `-p`/`--path` | parent directory (default `./environments`) |
| `-T`/`--add-tool` | also scaffold a `vf.Toolset` (`servers/tool.py`), wired into the taskset |
| `-U`/`--add-user` | also scaffold a `vf.User` simulator (`servers/user.py`) + a typed `State` + `@vf.stop` |
| `-H`/`--add-harness` | also scaffold a custom `vf.Harness` (`harness.py`), selectable via `--harness.id <name>` |
| `--v0` | scaffold a legacy v0 environment instead (can't combine with `--add-*`) |
| `--force` | overwrite an existing package (default: refuse) |

---

# Training

prime-rl consumes the same env over the env-server, so a training env is the eval config in TOML
form. In a prime-rl config:

```toml
[[orchestrator.train.env]]
name    = "gsm8k"
taskset = { id = "gsm8k-v1", dataset_name = "..." }                 # any v1 taskset id
harness = { id = "null", runtime = { type = "subprocess" } }
timeout = { scoring = 10 }                                          # per-stage cap (default: no limit)
# pool  = { type = "elastic", max_workers = 8, multiplex = 128 }    # env-server pool (default elastic, self-sizing)
```

`[orchestrator.renderer]` is required (set `name = "auto"` or a specific renderer) — the renderer
tokenizes rollouts into training samples.

# Backwards compatibility

A classic v0 `verifiers.load_environment` env runs through the v1 CLIs via the legacy bridge — its
rollouts mapped to v1 `Trace`s. Use `--id` instead of a `taskset`:

```bash
uv run eval --id reverse-text -n 2                                  # eval a v0 env
uv run eval --id reverse-text --args.num_train_examples 50          # v0 construction args
```
