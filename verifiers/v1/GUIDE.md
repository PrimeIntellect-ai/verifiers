# verifiers.v1 — user guide

How to build and run a v1 environment end-to-end. For the one-page tour of what v1 is and
why, see the [README](README.md); this guide is the longer-form how-to. `import verifiers.v1 as vf`.

## Mental model

A v1 environment is built from decoupled pieces, each packagable and configured through typed
config. You'll likely work with them in different proportions:

- **Task** — the unit of work: its *instance* is frozen, typed data (the prompt, the ground
  truth, runtime requests) that rides the wire and persists with the trace; its *class*
  carries the episode behavior — every `@reward` / `@metric` / `@stop`, lifecycle hooks,
  tools, and a user simulator (*what* the model is asked and *how* it's graded). For many
  environments, this is the only piece you write.
- **Taskset** — the optional preprocessing factory that derives tasks: config + `load_tasks()`,
  nothing else. It makes a dataset one `id` away (`--taskset.id gsm8k-v1`, typed knobs like
  `--taskset.split`), but it is one task constructor among equals — a topology's `load_tasks`
  or a task minted mid-run is just as first-class.
- **Harness** — the program that drives the rollout turn to turn, a chat loop or an agent CLI
  (*how* the model is called). **Usually you just pick a built-in** (`default` / `rlm` /
  `codex`); you only write your own if you need a custom rollout loop. With some exceptions, any
  task runs under any harness.
- **Runtime** — *where* the harness (and the task's tools / user simulator) executes:
  `subprocess` / `docker` / `prime` / `modal`. **You never write one** — runtimes ship with the
  framework behind one `Runtime` contract and compose with any task/harness; you just choose
  where code runs.

A **topology** composes several agents over these same pieces — one agent's trace becomes the
next agent's task (see [Authoring a topology](#authoring-a-topology)).

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

# Authoring tasks

Tasks ship in a package exporting a `vf.Taskset` — the factory that derives them. Scaffold one
with `uv run init my-task-v1` (add `--add-tool` / `--add-user` / `--add-harness` for more
pieces, `--v0` for a legacy environment), or copy the closest `environments/<name>_v1` and
edit. The minimal shape is:

```python
import verifiers.v1 as vf


class ReverseTask(vf.Task):
    answer: str                     # your own fields, alongside vf.Task's (prompt, system_prompt, ...)

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == self.answer)   # self IS the task


class ReverseConfig(vf.TasksetConfig):
    num_tasks: int = 100            # knobs surface as --taskset.num-tasks


class ReverseTaskset(vf.Taskset[ReverseTask, ReverseConfig]):
    def load_tasks(self) -> list[ReverseTask]:
        return [
            ReverseTask(idx=i, prompt=f"Reverse: {w}", answer=w[::-1])
            for i, w in enumerate(WORDS[: self.config.num_tasks])
        ]


__all__ = ["ReverseTaskset"]   # vf resolves the taskset by finding this Taskset subclass
```

Data on the instance, behavior on the class: the `Task` instance stays pure serializable data
(it rides the wire and persists with the trace) while the class carries the code. Methods
aren't fields, so the split costs nothing — and every way of minting a task is equal: this
factory, a topology's `load_tasks`, a task constructed mid-`go` from an upstream trace.

`vf.Taskset[TaskT, ConfigT]` is generic over two types: your `Task` subclass and your
`TasksetConfig` subclass. The framework reads these off the generic bases to type
`self.config` and `trace.task`, narrow the `--taskset.*` CLI flags, and validate the task
class against the harness before any data is loaded.

The taskset module must export its `Taskset` subclass via `__all__` — the loader walks the
exported names and finds the single `Taskset` subclass.

**Class vars.** A `Task` subclass has two: `NEEDS_CONTAINER` (default `False`) declares the
task class only runs in a container runtime (`docker` / `prime`), so the framework refuses the
subprocess runtime up front — the class-wide counterpart to a per-row `image` (see
[Runtimes](#runtimes)). `STATE` names the per-rollout `State` type its episodes carry (see
[Per-rollout state](#per-rollout-state)).

## The task

`vf.Task` is a frozen pydantic model. Subclass it to add typed, task-specific fields (the
reference answer, ground truths, per-row metadata) — available as `self` in every reward and
hook, and as `trace.task` to everything else. The base fields every task has:

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

`def load_tasks(self) -> list[TaskT]` builds the task list — the taskset's *only* job. It runs
**once at load** (not per rollout), so do dataset loading / filtering / slicing here off
`self.config`. Return your typed `Task` subclass instances; everything the episode needs
(config-derived values included) rides on their fields.

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

Rewards and metrics are decorated `async` methods **on the task class**. The framework
**injects whichever arguments you name** — declare any subset of `task` / `trace` / `runtime`
and you get exactly those (`task` is the task itself, i.e. `self`, injectable for signature
symmetry — usually you just read `self`):

```python
@vf.reward(weight=1.0)                 # summed (weighted) into trace.reward — a float or a dict to merge
async def correct(self, trace) -> float: ...

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

class GSM8KTask(vf.Task):
    answer: str

    @vf.reward()
    async def verify(self, trace, runtime) -> float:
        r = await runtime.run_uv_script(VERIFY, args=[self.answer, trace.last_reply])
        return float(r.stdout.strip() == "1.0")
```

### Judges

When grading can't be deterministic, `vf.Judge` is a **single-call** LLM judge: it owns the
OpenAI client (with the Prime key/team fallback), the call, and usage/cost capture, and leaves
the two things that differ as hooks — `build_messages` (prompt) and `parse` (verdict). It is a
utility, not an abstraction: owned by whoever's reward calls it — a task's `@reward` when the
verdict is part of the env's own grading, a topology's `@reward(agent=...)` when it crosses
agents. Subclass it, build one instance (it holds an HTTP client, so cache it rather than
constructing one per reward call), and call it from a `@reward`:

```python
import functools

import verifiers.v1 as vf

class CorrectnessJudge(vf.Judge[bool]):                 # Judge[ParsedT] — ParsedT is your verdict type
    prompt = "Question: {question}\nAnswer: {answer}\nResponse: {response}\nCorrect? Reply yes or no."

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return response.text.strip().lower().startswith("yes")

@functools.cache
def judge() -> CorrectnessJudge:
    return CorrectnessJudge(vf.JudgeConfig())           # one instance per eval, not per call

class MathTask(vf.Task):
    question: str
    answer: str

    @vf.reward()
    async def correct(self, trace) -> float:
        result = await judge().evaluate(
            trace=trace, question=self.question, answer=self.answer, response=trace.last_reply
        )
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
- **Config**: `JudgeConfig` adds `model` + `sampling` (a `JudgeSamplingConfig`) and a `prompt` / `prompt_file` template override to `BaseClientConfig` (`base_url`/`api_key_var`/`headers`, Prime auto-config). Declare a `judge: vf.JudgeConfig` field on your taskset config and bake it onto a task field to make the judge CLI-tunable (`--taskset.judge.model …`, `--taskset.judge.sampling.max-tokens …`).
- **Errors propagate**: a judge API failure errors the rollout (recorded as a `TaskError` on the trace); the OpenAI SDK already retries transient 429/5xx/connection errors. `vf.judge_verdict(text, ("yes", "no"))` is the parsing counterpart: it returns the verdict label in `text` or raises when none is found — a wrong or non-committal *model* reply scores 0, but an unparseable verdict is a *judge* failure and must error the rollout, never silently score the model 0.

### Judge-as-agent — the `llm-judge` and `agentic-judge` topologies

There is no judge plugin tier: `vf.Judge` stays a plain utility owned by the reward that calls
it. To plug grading in **from eval config alone**, use the built-in
[`llm-judge` topology](#the-built-in-judge-topologies): a `solver` running any taskset and a
non-trainable `judge` grading its final answer against the task and its ground truth,
composable with any taskset × harness pair straight from the CLI. When the judge should
itself be an agent with tools, its own model, and its own runtime, use `agentic-judge` — the
solver's entire trace is uploaded into the judge's runtime for it to investigate:

```bash
uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
uv run eval --topology.id agentic-judge --topology.taskset.id gsm8k-v1 -n 4
```

The `llm-judge` judge is fixed to the in-process `direct` harness, so an episode costs
roughly one API call — the plain-judge price, but the verdict comes back as a real,
inspectable trace.

## Stop conditions

A rollout ends when the harness finishes, a framework budget trips (`--max-turns`, token caps), or
a task `@vf.stop` fires. A stop is an `async (self, trace) -> bool` checked between turns; its
**method name becomes the stop reason**:

```python
@vf.stop
async def saw_answer(self, trace) -> bool:
    last = trace.last_reply
    return "FINAL:" in last
```

The framework has no built-in "the task is done" signal — multi-turn tasks end either from the
trace (above) or from per-rollout state set by a tool / user sim (see [State](#per-rollout-state)).

## Lifecycle hooks

A rollout runs **`setup → harness → finalize → scoring`**. The task class can hook any stage —
like the scoring methods, `setup` / `finalize` declare any subset of `task` / `trace` /
`runtime` by parameter name and the framework injects them:

| hook | signature | when |
| --- | --- | --- |
| `setup` | `(self, trace, runtime)` | per-rollout prep in the live runtime before the harness (clone a repo, start a service); the trace — and its `trace.state` — already exists, so it may stash per-rollout state |
| `finalize` | `(self, trace, runtime)` | after the harness, before scoring — apply a diff, snapshot, scrape artifacts into `trace.info` |
| `validate` | `(self, runtime) -> bool` | never during a rollout — the model-free gold check the [`validate` CLI](#validate) runs, in a runtime with `setup` already applied |
| `load_tools` | `(self) -> list[vf.Toolset]` | per rollout, before the harness — the task's tool servers |
| `load_user` | `(self) -> vf.User \| None` | per rollout, before the harness — the user simulator |

`setup`/`finalize` errors fail the rollout legibly (captured onto the trace, not a crash);
`validate` returns `False` — or raises — to mark the task invalid.

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
task code as a `TaskError`. The same code works on subprocess / docker / prime / modal.

A SWE task is the canonical case: `setup` provisions the repo, the agent edits it during the
rollout, and a `@reward` runs the tests in the *same* runtime:

```python
class SWETask(vf.Task):
    NEEDS_CONTAINER: ClassVar[bool] = True   # refuse the subprocess runtime

    repo_url: str
    base_commit: str
    test_cmd: str

    async def setup(self, trace, runtime) -> None:
        await runtime.run(["git", "clone", self.repo_url, "/repo"], {})
        await runtime.run(["git", "-C", "/repo", "checkout", self.base_commit], {})

    async def finalize(self, trace, runtime) -> None:
        diff = await runtime.run(["git", "-C", "/repo", "diff"], {})
        trace.info["diff"] = diff.stdout   # scrape the agent's diff off the live runtime

    @vf.reward()
    async def tests_pass(self, trace, runtime) -> float:
        result = await runtime.run(["bash", "-lc", self.test_cmd], {})
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
default), declare it on the task class via the `STATE` classvar, and parameterize any stateful
server on it:

```python
class GameState(vf.State):
    game_over: bool = False

class GameUser(vf.User[vf.UserConfig, GameState]):
    async def respond(self, message: str) -> vf.Messages:
        ...
        if finished:
            self.state.game_over = True   # the @vf.stop below ends the rollout
        return [{"role": "user", "content": reply}]

class GameTask(vf.Task):
    STATE: ClassVar[type[vf.State]] = GameState

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
> the sequential path. The task class and its servers must share **one** `State` subclass — a server
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

A task exposes its tools via `load_tools() -> list[vf.Toolset]`, constructing each from the
`tools` config field the instance carries (plain name = data, `load_` prefix = constructor —
the same convention as `load_tasks`); the factory bakes its own `tools` knob onto that field,
so placement stays CLI-tunable (below). Placement splits into two kinds with different homes:
**structural intent** — this corpus is expensive so it's `shared`, this tool needs the agent's
filesystem so it's `colocated`, this endpoint is a remote `url` — is part of what the task
*is*, so pin it in code inside `load_tools()` (or as the field's default); **infra tuning** —
which runtime type hosts the server — is what the CLI-tunable knob is for:

```python
class GlossaryTask(vf.Task):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()

    def load_tools(self) -> list[vf.Toolset]:
        return [GlossaryToolset(self.tools)]

class GlossaryConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()          # --taskset.tools.*

class GlossaryTaskset(vf.Taskset[GlossaryTask, GlossaryConfig]):
    def load_tasks(self) -> list[GlossaryTask]:
        return [
            GlossaryTask(idx=i, prompt=..., tools=self.config.tools)
            for i, entry in enumerate(ENTRIES)
        ]
```

## User simulators

A user simulator is a `vf.User[ConfigT]` (or `[ConfigT, StateT]`) with one hook:

```python
class HagglerUser(vf.User[vf.UserConfig, HagglerState]):
    async def respond(self, message: str) -> vf.Messages:
        # the model's last assistant text in → the next user message(s) out ([] to emit nothing)
        if done:
            self.state.deal_closed = True       # end via a @vf.stop the task class declares
        return [{"role": "user", "content": reply}]


if __name__ == "__main__":
    HagglerUser.run()                           # self-launching module under servers/ (see below)
```

The framework calls `respond` after each agent turn and injects the reply as the next user
message; it's consumed by the framework, never shown to the model. A task supplies one via
`load_user() -> vf.User | None`, constructed from the task's `user` config field exactly
like tools. If a task carries
**no prompt** (`prompt=None`), the simulator also **opens the conversation**: the framework
calls `respond("")` once before the first model turn and seeds its reply as the initial user
message. End the trajectory by setting a `self.state` flag a task `@vf.stop` checks (there's no
built-in end signal).

A task may expose **both** tools and a user sim at once — they're served together each rollout;
the harness just needs to support both.

## Server placement & isolation

Each tool/user server is its own self-launching module under the env package's `servers/`, ending
with `if __name__ == "__main__": <Server>.run()`; the framework launches it with
`python -m <env>.servers.<name>`. **Placement** lives on the server's config (a `vf.ToolsetConfig`
/ `vf.UserConfig` the factory bakes onto a task field from a taskset-config knob), so it's
per-server and CLI-tunable (`--taskset.tools.shared true`). It decides **where** the server runs
and **how many** there are — the default is the cheapest correct thing; the rest trade setup cost
for isolation:

| mode | config | runs | pros | cons |
| --- | --- | --- | --- | --- |
| **own runtime** *(default)* | *(nothing; `runtime = {type = "docker"\|"prime"}` for a sandbox)* | own runtime per rollout — `subprocess` on the host (default), or a `docker`/`prime` sandbox over a tunnel | full per-rollout isolation; a sandbox also isolates untrusted code / deps / network | pays `setup` every rollout (a sandbox adds spin-up + env install) |
| **colocated** | `colocated = true` | inside the harness's runtime, one per rollout (no tunnel) | no extra runtime/tunnel; can touch the harness's filesystem | couples to the harness; `setup` per rollout |
| **shared** | `shared = true` | one instance for the whole run | `setup` once; writable per-rollout if state lives in `self.state` | state outside `self.state` corrupts across rollouts; `setup_task` skipped |
| **remote** *(tools only)* | `url = "https://…"` | connects to an already-running MCP endpoint | zero hosting; use a public/third-party server | no isolation, state, or lifecycle control |

`shared` servers live in a **lazy, run-scoped registry**: the first rollout whose task declares
one starts it, every later rollout reuses it — deduped by the toolset's identity (its class +
its config) — and it's torn down with the run. Lazy start means a topology's *derived* tasks get
shared servers too; seed tasks aren't special. The semantics for authors are unchanged:
`shared=True` still means one task-agnostic instance per run (`setup` once, no `setup_task`).
`shared` works on any runtime — the framework makes the rollout's `/state` channel reachable from the
shared server (localhost, or a host tunnel when remote). Keep big shared data off the Python heap
(numpy / mmap / an on-disk index) so one instance scales across rollouts.

**Choosing per-rollout state:** read-only resource → `shared`; state that fits a `State` model →
`shared` + `self.state` (no extra process, the scalable default); state that can't (module globals,
on-disk scratch, a stateful C library) → a per-rollout placement (own runtime or `colocated`) that
pays `setup` each time; cheap `setup` → just use the default. **User simulators** support only the
per-rollout placements (own runtime or `colocated`); `shared` / `url` are tools-only.

## Learn from the examples

The `*_v1` packages under `environments/` are the reference library — each shows one pattern:

| example | pattern |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn task |
| `gsm8k-v1` | single-turn + in-runtime scoring (a `@reward` uv script) |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | multi-turn, stateful, driven by a `vf.User` simulator |
| `glossary-v1` | the simplest tool server (own host runtime) |
| `wiki-search-v1` | a shared, read-only tool server (built once) + an LLM judge |
| `scratchpad-v1` | a shared, **writable** tool server — per-rollout state isolated via `self.state` |
| `deepwiki-v1` | an existing remote tool server, by URL |
| `color-codeword-v1` | a multimodal (image) task |
| `wordle-v1` | a thin config over the shipped `textarena` integration |
| `proposer-solver-v1` | a multi-agent **topology**: proposer → n solvers (fan-out), deferred rewards |
| `writer-editors-v1` | a multi-agent **topology**: rounds + fan-in (writer → n editors → revision), one `vf.Judge` verdict rewarding every trace |

---

# Authoring a harness

If you need to customize the rollout logic beyond what the built-in harnesses provide — a loop they
can't express (context compaction, subagents, a bespoke agent CLI) — author a custom harness.
Otherwise pick a built-in, selected with `--harness.id`:

| id | what it is |
| --- | --- |
| `default` | a `bash` + `edit` coding agent (`edit` on by default — `--harness.edit false` for bash-only; `--harness.search true` adds a serper.dev `search` tool) — the fallback when no harness is given |
| `null` | a tiny OpenAI chat loop (MCP tools only, no tools of its own) |
| `direct` | an in-process chat loop — no subprocess, no tools; an episode ≈ one API call (the `llm-judge` topology's judge) |
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

Class vars on the harness gate which tasks it can run, so an incompatible pairing fails fast at
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
the finished trace alongside the task's — handy to surface what the program left behind in the
runtime (e.g. read a `meta.json` the binary wrote). A harness can't define rewards.

---

# Authoring a topology

A **topology** composes multiple agents over episodes: which agents exist, how one agent's
trace becomes the next agent's task, and how rewards flow backwards once downstream agents
have run. Each *episode* is an ordinary rollout — one agent consuming one task and producing
one trace — and tasks carry their own behavior, so nothing about a task or a harness is
topology-specific. Run one with `--topology.id` (it replaces the eval's own
`taskset` × `harness` pair):

```bash
uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
uv run eval --topology.id proposer-solver-v1 -n 3
```

## Agents

An agent is **pure routing** — a name + the harness driving its episodes + how its model calls
are routed; it carries nothing task-side. Declare agents as typed `AgentConfig` fields on your
config — the field *name* is the agent's name, and every agent is CLI-addressable
(`--topology.solver.harness.id rlm`, `--topology.judge.model <id>`):

```python
class ProposerSolverConfig(vf.TopologyConfig):
    proposer: vf.AgentConfig = vf.AgentConfig(harness={"id": "direct"})
    solver: vf.AgentConfig = vf.AgentConfig(harness={"id": "direct"})
    num_solvers: int = 4
```

An `AgentConfig` binds a harness (and where it runs — `harness.runtime`) plus per-agent
routing: `model` / `client` / `sampling` overrides and a `trainable` flag (stamped onto every
trace the agent produces, so a trainer can drop e.g. judge traces without the topology
config). Subclass it to pin typed per-agent defaults — the `llm-judge` topology's judge pins
the `direct` harness and `trainable=False`. The tasks an agent consumes (each carrying its own
behavior) arrive per episode, from the topology's seeds or constructed in `go`.

`Agent(name, config)` is the one constructor. The default `load_agents` builds one per
`AgentConfig` field, in declaration order; override it only to compose agents
programmatically. Loading also validates the topology's declared judgement
(`@reward(agent=...)` / `@metric(agent=...)`) against the agents, so a typo'd or missing agent
scope fails at load time, not mid-eval.

## Seed tasks

One topology instance runs per seed task (× `-r`). Seeds come from the config's `tasks`
factory — any taskset, plugged by id (`--topology.taskset.id gsm8k-v1`; its knobs validate
typed, e.g. `--topology.taskset.split train`) — or from a `load_tasks` override for a
self-seeding topology. **Exclusive-or, enforced at load**: when the slot can be set it IS
the seed source, verbatim; a topology that overrides `load_tasks` is refused the flag
(rather than silently ignoring it), and a custom `load_tasks` wanting a config-driven
source declares its own factory field:

```python
class ProposerSolverTopology(vf.Topology[ProposerSolverConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: the references are baked in, so no `--topology.taskset.id` needed."""
        return [
            ProposeTask(idx=i, prompt=PROPOSE_PROMPT.format(reference=reference))
            for i, reference in enumerate(REFERENCES)
        ]
```

Per-role behavior lives on **task classes**, minted anywhere. In `proposer-solver-v1`,
`ProposeTask` judges its own episode (a format reward), and the `SolverTask` built mid-`go`
carries the ground truth *and* the `correct` reward — question and verifier in one typed
object, serialized with each solver trace so the record shows exactly what was asked:

```python
class ProposeTask(vf.Task):
    @vf.reward(weight=0.1)
    async def well_formed(self, trace: vf.Trace) -> float:
        answer = parse_number(parse_labeled(trace, "ANSWER") or "")
        return float(bool(parse_labeled(trace, "QUESTION")) and answer is not None)


class SolverTask(vf.Task):
    answer: str   # the proposer's canonical numeric answer

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return float(parse_number(parse_labeled(trace, "ANSWER") or "") == self.answer)
```

## The interaction pattern — `go`

`go` is plain imperative Python over a `TopologyRun`; interaction patterns are code, not a
DSL. `run.agent(name).run(task, parents=...)` runs one episode and links it into the
agent graph; `asyncio.gather` fans out; loops are rounds; awaiting several traces before
building the next task is fan-in:

```python
    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        proposer = await run.agent("proposer").run(task)
        # Forward arrow: read the proposal straight off the trace, pure host-side.
        question = parse_labeled(proposer, "QUESTION")
        answer = parse_number(parse_labeled(proposer, "ANSWER") or "")
        if not question or answer is None:
            return  # malformed proposal — `well_formed` scored it; nothing to solve
        derived = SolverTask(idx=task.idx, prompt=SOLVE_PROMPT.format(question=question), answer=answer)
        solver = run.agent("solver")
        await asyncio.gather(
            *(
                solver.run(derived, parents=[proposer])
                for _ in range(self.config.num_solvers)
            )
        )
```

## Topology rewards — declared, cross-agent judgement

Per-episode judgement rides on the task classes (`SolverTask.correct` above). Cross-agent
judgement is *not* written inline in `go`: declare it as `@vf.reward(agent=...)` /
`@vf.metric(agent=...)` methods on the topology — the same decorators tasks use, scoped
to an agent. Each runs once per matching trace **after the whole instance completes**, with
any of `task` / `trace` / `graph` injected by parameter name, and records under the method
name (weighted) exactly like a task reward:

```python
    @vf.metric(agent="proposer")
    async def solve_rate(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        graded = [t for t in graph.children(trace, agent="solver") if not t.has_error]
        return sum(t.rewards.get("correct", 0.0) for t in graded) / len(graded) if graded else 0.0

    @vf.reward(agent="proposer")
    async def difficulty(self, trace: vf.Trace) -> float:
        return 1.0 - 2.0 * abs(trace.metrics["solve_rate"] - 0.5)   # rewards may read metrics
```

The contract, chosen to fail loudly and stay predictable:

- **Validated at load**: an `agent=` scope that doesn't exist — or a topology `@reward`
  with no scope — is refused when the topology loads, before anything runs.
- **Runs at instance end**, never earlier: nothing can observe a reward before the
  instance persists, so there is no "earliest possible" scheduling to reason about. Every
  trace in scope is scored — across all rounds and fan-outs — automatically.
- **Ordering**: methods run sequentially, metrics before rewards, each phase in
  (priority, name) order. A method may read task-recorded rewards (final since the
  episode ended) and, in the rewards phase, any metric — but topology rewards must not
  read each other; derive shared inputs from the traces or a metric.
- **Failures**: a raise during instance scoring is classified `TopologyError` and recorded
  on the graph — episodes stay as data, siblings unaffected.

`trace.record_reward(...)` inside `go` still works as the escape hatch for exotic shapes
(e.g. a mid-round adjustment), but the declared methods are the norm.

The forward arrow stays in `go`: construct the downstream agent's typed `Task` from an
upstream trace — its typed task, `last_reply`, `transcript`, or `trace.info`. This is pure
host-side code; only when peeling requires the episode's *live runtime* (scraping files,
running a build) does the upstream task class need a `finalize` hook, which runs before
teardown and parks results in `trace.info`. The backward arrow — cross-agent rewards — is
declared, not inlined (above).

Episode failures never raise into `go` — they come back as data on the trace
(`trace.has_error`), and `go` decides what a failed child means (drop it, count it against a
pass rate, retry the round). A crash in `go` itself is recorded on the instance's graph as a
`TopologyError` and doesn't touch sibling instances.

## The agent graph

Running one instance produces an `AgentGraph` — the serialized instance artifact
`{id, topology, error, traces[]}`: every episode's trace in completion (= topological) order,
each stamped with `trace.agent`, `trace.parents` (upstream trace ids), and `trace.trainable`.
A topology run persists **one instance record per `results.jsonl` line**, traces nested (an
instance's rewards are only final once the whole instance is done); `AgentGraph.load(dict)`
reads one back without the originating packages (task-specific fields ride in
`task.model_extra`). The links themselves are plain `Trace` fields, so the graph also
reconstructs from any flat trace dump (one instance = one connected component). Navigation —
`graph.roots()` / `graph.children(trace, agent=...)` / `graph.by_agent(name)` — is what
cross-agent scoring lives on, and `graph.error` records a crash in `go` itself. A `Trace` is
the per-agent view of one episode; the agent graph is the global view of the interaction.

## The built-in judge topologies

Judging as a config-only pattern, two tiers, one verdict contract (a `SCORE: <0-10>` line,
recorded on the *solver's* trace as a weighted reward, `--topology.weight`):

- **`llm-judge`** — a `solver` (any taskset, via `--topology.taskset.id`) and a non-trainable
  `judge`, **fixed** to the in-process `direct` harness (an episode ≈ one API call). `go`
  peels the judge's inputs off the finished solver episode — the seed task's framing, its
  ground truth (an `answer` field, when the taskset carries one), and the solver's final
  message — into a `JudgeTask` minted from the trace. Give the judge its own model
  (`--topology.judge.model`) or client routing; swapping its harness is refused and points
  here:
- **`agentic-judge`** — same shape, but the judge is a real agent: the solver's **entire
  serialized trace** is uploaded into the judge's own runtime (by the judge task's `setup`
  hook), and the judge investigates it with its tools before committing. Its harness is
  configurable (`--topology.judge.harness.id ...`, bash+edit `default` by default), as is its
  assignment (`--topology.prompt`, with `{path}` = where the trace landed).

For a verdict baked into a task's own grading, call [`vf.Judge`](#judges) from the task's
`@reward` instead — that's the cheap utility tier; these topologies are the tier where the
judge is itself an agent.

Not yet supported under a topology: `--server` (env-server serving), `--resume`, and the
`--rich` dashboard.

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

A task class that sets `NEEDS_CONTAINER` (or a task row with an `image`) refuses the subprocess
runtime — pass `docker` / `prime` / `modal`.

---

# CLI reference

Four commands, all `uv run <cmd>`: **`eval`** (run + score a model), **`validate`**
(model-free setup/gold checks), **`debug`** (setup + one shell action), **`init`**
(scaffold). They share a resolution layer:

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
| topology | `--topology.id` (replaces the taskset × harness pair), `--topology.taskset.id <taskset>` (seeds; typed knobs as `--topology.taskset.*`), `--topology.<agent>.*` (per-agent `harness` / `model` / `client` / `sampling` / `trainable`) |
| counts | `-n`/`--num-tasks` (all), `-r`/`--num-rollouts` (1; `@group_reward` needs ≥2; topology instances per seed under `--topology.id`), `-s`/`--shuffle` |
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

**What it writes** (into `--output-dir`, default `outputs/<taskset>--<model>--<harness>/<uuid>`;
a topology run uses `outputs/<topology>--<model>/<uuid>`): `config.toml` (the resolved config,
re-runnable via `@ config.toml`), `results.jsonl` (one full trace per line — one full *instance*
record per line, traces nested, for a topology run — appended as each finishes, so it's durable
mid-run), and `eval.log`.

**`--dry-run`** writes `config.toml` and exits (resolve + validate, no run). **`--resume
<output-dir>`** re-runs only the missing/errored rollouts of a previous run (it reloads that run's
`config.toml`, so it takes no other args).

## `validate`

Model-free checks that a task is sound. By default each task gets two independent
runtimes — a **gold** check (the task's `setup` then its `validate` hook: the gold patch
makes the tests pass, the verifier accepts the gold answer) and a **setup-only** check —
reported as one aggregate row. Restrict to one check with `--only-gold` / `--only-setup`.
No model, no harness.

```bash
uv run validate gsm8k-v1 -n 20 --runtime.type subprocess
uv run validate swebench-v1 -n 1 --runtime.type prime --only-setup
uv run validate swebench-v1 -n 1 --runtime.type prime --only-gold
```

| flag | default | meaning |
| --- | --- | --- |
| `<taskset-id>` / `--taskset.id` | — | taskset to validate |
| `--runtime.type` | `docker` | runtime for `setup` + `validate` (a gold check often needs the task's container) |
| `--only-setup` / `--only-gold` | off | run just the setup-only or just the gold check (default: both) |
| `--timeout.setup` / `--timeout.total` | None | per-task wall-clock caps for `setup` and the `validate` hook |
| `-n`/`--num-tasks`, `-s`/`--shuffle`, `-c`/`--max-concurrent` (128) | | task selection + concurrency |
| `-v`/`--verbose`, `--no-rich` | | logging / disable the dashboard |

Each task is provisioned, set up, validated, and torn down independently; a raised error is
captured as a result row (one bad task is data, not a crash) with reason `valid` / `invalid` /
`timeout` / `error`. **Fire-and-forget — nothing is written to disk**; results show live. Note the
default runtime is **docker** (unlike eval's subprocess), and a subprocess runtime against
`NEEDS_CONTAINER` / image-bearing tasks aborts with a clear error.

## `debug`

Set up each selected task, run one explicit shell action, and save the trace. This is for
inspecting task state without a model: it does not apply gold patches, call `validate`, run
`finalize`, or score rewards.

```bash
uv run debug swebench-v1 -n 1 --runtime.type prime --command 'pwd; git status --short | head'
uv run debug swebench-v1 -n 1 --runtime.type prime --script-path ./inspect.sh
```

| flag | default | meaning |
| --- | --- | --- |
| `<taskset-id>` / `--taskset.id` | — | taskset to debug |
| `--command` / `--script-path` | — | exactly one inline command or host script to upload and execute |
| `--runtime.type` | `docker` | runtime for setup + the debug action |
| `--timeout.setup` / `--timeout.total` | None | per-task wall-clock caps for `setup` and the debug action |
| `-n`/`--num-tasks`, `-s`/`--shuffle`, `-c`/`--max-concurrent` (128) | | task selection + concurrency |
| `-o`/`--output-dir` | fresh debug run dir | where `config.toml` and `results.jsonl` are written |

Each saved trace has command/script metadata, exit status, elapsed time, timeout/error fields,
and the full stdout/stderr under `trace.info["debug"]`.

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
