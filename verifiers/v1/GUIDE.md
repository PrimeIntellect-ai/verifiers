# verifiers.v1 — user guide

How to build and run a v1 environment end-to-end. For the one-page tour of what v1 is and
why, see the [README](README.md); this guide is the longer-form how-to. `import verifiers.v1 as vf`.

## Mental model

A v1 environment is three decoupled pieces, each selected by `id` and configured by typed
config. You'll work with them in very different proportions:

- **Taskset** — the data and the scoring: it produces typed `Task`s and owns every `@reward` /
  `@metric`, plus any tools and a user simulator (*what* the model is asked and *how* it's
  graded). **This is what you author** — for almost every environment it's the only piece you
  write.
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
uv run eval gsm8k-v1 -n 5 -r 3   # 5 tasks, 3 rollouts each; default harness, docker runtime
uv run eval -h                   # typed help (lists local tasksets + harnesses)
```

Everything below has a CLI flag *and* a TOML equivalent (`uv run eval @ config.toml`); the
flag names are the dotted config path (`--harness.runtime.type docker`).

## Authoring a taskset

A taskset is a package selected by `id`. Copy the closest `environments/<name>_v1` and edit.
Minimal shape:

```python
import verifiers.v1 as vf


class ReverseTask(vf.Task):
    answer: str                     # your own fields, alongside vf.Task's (instruction, system_prompt, ...)


class ReverseConfig(vf.TasksetConfig):
    num_tasks: int = 100            # knobs surface as --taskset.num-tasks


class ReverseTaskset(vf.Taskset[ReverseTask, ReverseConfig]):
    def load_tasks(self) -> list[ReverseTask]:
        return [
            ReverseTask(idx=i, instruction=f"Reverse: {w}", answer=w[::-1])
            for i, w in enumerate(WORDS[: self.config.num_tasks])
        ]

    @vf.reward()
    async def exact_match(self, task: ReverseTask, trace: vf.Trace) -> float:
        return float(trace.assistant_messages[-1].content.strip() == task.answer)


__all__ = ["ReverseTaskset"]   # vf resolves the taskset by finding this Taskset subclass
```

### Scoring

Rewards and metrics are decorated `async` methods; the framework injects whichever of `task`
/ `trace` / `runtime` you name as parameters.

```python
@vf.reward(weight=1.0)                 # summed into trace.reward
async def correct(self, task, trace) -> float: ...

@vf.metric()                           # recorded, not summed (return float or a dict to merge)
async def num_turns(self, trace) -> int: ...

@vf.group_reward(weight=1.0)           # scores a task's N rollouts together
async def best_of_n(self, traces: list[vf.Trace]) -> list[float]: ...

@vf.stop()                             # extra stop condition, (self, trace) -> bool
async def saw_answer(self, trace) -> bool: ...
```

Higher `priority` runs first; `weight` controls aggregation. To score with a dependency the
eval process shouldn't have (e.g. `math-verify`), run it as a uv script *in the rollout's
runtime* — the dep never touches the eval process:

```python
VERIFY = (Path(__file__).parent / "verify.py").read_text()   # PEP 723 header declares its deps

@vf.reward()
async def verified(self, task, trace, runtime) -> float:
    r = await runtime.run_uv_script(VERIFY, args=[task.answer, trace.assistant_messages[-1].content])
    return float(r.stdout.strip() == "1.0")
```

### Lifecycle hooks

A rollout runs `setup → harness → finalize → scoring`, each independently timeout-bounded
(`--timeout.{setup,rollout,finalize,scoring}`). A taskset can hook any stage (all `async`):

- `setup(task, runtime)` — per-task prep before the harness runs (clone a repo, start a service).
- `finalize(task, trace, runtime)` — after the harness, before scoring (apply a diff, snapshot state, scrape runtime artifacts into `trace.info`).
- `validate(task, runtime)` — model-free gold check (does the reference solution pass?), run by `uv run validate`.

### Runtime access

Most hooks run *with the live runtime* and can execute in it, so the whole rollout shares one
isolated environment. Who gets the `runtime`:

| hook | signature | runtime |
| --- | --- | --- |
| `setup` | `(task, runtime)` | ✓ — prep it before the harness runs (the trace doesn't exist yet) |
| `finalize` | `(task, trace, runtime)` | ✓ — act on the finished trace + runtime, before scoring |
| `validate` | `(task, runtime)` | ✓ — model-free gold check |
| `@reward` / `@metric` / `@stop` | inject any of `task` / `trace` / `runtime` | ✓ |
| `@group_reward` | `(traces[, task])` | ✗ — runs after the per-rollout runtimes are gone |

On a `runtime` you can call: `run(argv, env)` (exec to completion → exit code + stdout/stderr),
`run_uv_script(src, args, env)` (a PEP 723 script with inline deps), `run_background(argv, env,
log)` (a long-lived server), `read(path)` / `write(path, data)` (workspace files), and
`expose(port)` (a URL reaching a port inside the runtime).

A SWE taskset is the canonical case: `setup` provisions the repo, the agent edits it during the
rollout, and a `@reward` runs the tests in the *same* runtime:

```python
class SWETaskset(vf.Taskset[SWETask, SWEConfig]):
    NEEDS_CONTAINER = True   # this taskset needs an isolated container/sandbox runtime

    async def setup(self, task: SWETask, runtime: vf.Runtime) -> None:
        # prep the runtime before the harness runs: clone + check out the base commit
        await runtime.run(["git", "clone", task.repo_url, "/repo"], {})
        await runtime.run(["git", "-C", "/repo", "checkout", task.base_commit], {})

    async def finalize(self, task: SWETask, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # scrape the agent's diff off the live runtime into trace.info; it persists to results.jsonl
        diff = await runtime.run(["git", "-C", "/repo", "diff"], {})
        trace.info["diff"] = diff.stdout

    @vf.reward()
    async def tests_pass(self, task: SWETask, trace: vf.Trace, runtime: vf.Runtime) -> float:
        # the agent edited /repo during the rollout; run the task's tests in that same runtime
        result = await runtime.run(["bash", "-lc", task.test_cmd], {})
        return 1.0 if result.exit_code == 0 else 0.0
```

`trace.info` is a free-form, JSON-serializable dict for anything that isn't a reward or metric —
runtime artifacts (the diff above, captured logs, command output) you want persisted with the
trace for inspection. Like the rewards/metrics it rides along to `results.jsonl`; use `metrics`
for numbers that aggregate, `trace.info` for everything else.

`trace.state` is the complementary **transient** store: a typed, mutable `vf.State` shared across the
rollout's tool servers, user simulator, and scoring — the one place per-rollout *runtime* state lives
(counters, game progress, your own end-of-trajectory flag). Unlike `info` it is **never** persisted to
disk or sent over the wire. A `@vf.tool` / `respond` reads+writes it as `self.state` (synced over the
interception server per call, so tools and the user sim see each other's writes); `@reward` /
`@metric` / `finalize` read+write `trace.state` directly. The base `vf.State` is empty — subclass it
to declare typed fields and parameterize the taskset (`vf.Taskset[Task, Config, MyState]`) plus any
stateful server (`vf.Toolset[Config, MyState]` / `vf.User[Config, MyState]`); it defaults to the base.
To **end a trajectory from state**, set your own flag and declare a `@vf.stop` over it (the framework
has no built-in end signal):

```python
class GameState(vf.State):
    game_over: bool = False

class GameUser(vf.User[vf.UserConfig, GameState]):
    async def respond(self, message: str) -> vf.Messages:
        ...
        if finished:
            self.state.game_over = True   # the @vf.stop below ends the rollout
        return [...]

class GameTaskset(vf.Taskset[GameTask, GameConfig, GameState]):
    @vf.stop
    async def game_over(self, trace) -> bool:   # stop reason is this method's name
        return trace.state.game_over
```

> **Concurrency — `self.state` is last-write-wins.** Each `@vf.tool` / `respond` call syncs the
> *whole* state as a read-modify-write: pull `self.state` from the host, run, push it back. Tool calls
> a harness runs **concurrently** (several `tool_calls` in one assistant turn) therefore race — each
> reads the same starting state and the last push wins, so concurrent increments/appends are lost.
> Tools the harness runs **sequentially** compose correctly. Keep shared-state mutations on the
> sequential path (or accumulate per-key on the host if you need parallel-safe writes). The taskset
> and its servers must also share **one** `State` subclass — a server that pushes a mismatching shape
> is rejected (the rollout fails legibly).

Since `@group_reward` has no runtime, fold any runtime-derived signal (here, pass/fail) into a
per-rollout `@reward`/`@metric` first, then compare those across the task's rollouts.

### Tools and user simulators

Both a tool server and a user simulator are **vf-native classes** (not raw MCP, no FastMCP
boilerplate) authored from a config — the same shape as a taskset:

- A **tool server** is a `vf.Toolset[ConfigT]` with `@vf.tool` methods (the model sees
  `<TOOL_PREFIX>_<method>`; the docstring is the description). A taskset exposes a task's tools via
  `tools(task) -> list[vf.Toolset]`.
- A **user simulator** is a `vf.User[ConfigT]` with one `async def respond(message) -> Messages` hook
  (the framework calls it after each assistant turn for the next user message(s); end the trajectory
  by setting a `self.state` flag a taskset `@vf.stop` checks — see above). A taskset supplies one via
  `user(task) -> vf.User | None`. If a task carries no prompt (`instruction=None`), the simulator also
  **opens the conversation**: the framework calls `respond("")` once before the first model turn and
  seeds its reply as the initial user message.

A taskset may expose **both** at once (tools the model calls *and* a user sim driving the turns) —
they're served together each rollout; a harness just needs to support both.

**Where they live.** Each server is its own self-launching module under the env package's
`servers/`, ending with `if __name__ == "__main__": <Server>.run()`; the framework launches it with
`python -m <env>.servers.<name>` (host: ambient; sandbox: the env package is uploaded + installed
first). Build state as plain `self.x` attributes in `async def setup(self)` (task-agnostic, runs for
every server) or `async def setup_task(self, task)` (per-rollout — **skipped for a `shared`
server**). Those attrs are server-local; for per-rollout state **shared** with the other servers and
scoring use the typed `self.state` (`trace.state`, above). Fixed data lives in module constants, not
config.

**Placement** is config on the server's config (a `vf.ToolsetConfig` / `vf.UserConfig` field on the
taskset's own config), so it's per-server and CLI-tunable (`--taskset.tools.runtime.type docker`,
`--taskset.tools.shared true`, `--taskset.user.colocated true`):

| placement | how |
| --- | --- |
| own host runtime (default) | its own `subprocess` runtime on the host, reached over the host network |
| own per-rollout runtime | `runtime = {type = "docker"/"prime"}`, reached over a tunnel |
| colocated | `colocated = true` — inside the harness's runtime (reached in-sandbox, no tunnel) |
| shared (tools only) | `shared = true` — one instance built once for the whole eval |
| remote (tools only) | an existing server, by `url` |

### Learn from the examples

The `*_v1` tasksets under `environments/` are the reference library — each shows one pattern:

| example | pattern |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1`, `aime24-v1`, `math-env-v1` | single-turn + in-runtime scoring (a `@reward` uv script) |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | multi-turn, stateful, driven by a `vf.User` simulator |
| `glossary-v1` | the simplest tool server (own host runtime) |
| `wikispeedia-v1` | a stateful tool server (global `setup` + per-task `setup_task`) |
| `wiki-search-v1` | a shared tool server (built once) + an LLM judge |
| `deepwiki-v1` | an existing remote tool server, by URL |
| `color-codeword-v1` | a multimodal (image) task |
| `scaleswe-v1`, `swelego-v1`, `r2e-gym-v1` | containerized SWE tasks (rlm harness, prime runtime) |
| `wordle-v1`, `terminal-bench-2-v1` | thin configs over the shipped `textarena-v1` / `harbor-v1` integrations |

## Harnesses

Built-ins, selected with `--harness.id`:

| id | what it is |
| --- | --- |
| `default` | a tiny OpenAI chat loop (bash tool opt-in via `--harness.enable-bash`) |
| `rlm` | the RLM CLI agent |
| `codex` | the Codex CLI (Responses dialect + SSE relay) |

```bash
uv run eval gsm8k-v1 -n 1                    # default harness
uv run eval gsm8k-v1 -n 1 --harness.id rlm   # same taskset, different driver
```

**Capability flags** gate which tasksets a harness can run, so an incompatible pairing fails
fast at load instead of mis-running: `SUPPORTS_TASK_TOOLS`, `SUPPORTS_USER_SIM`,
`SUPPORTS_MESSAGE_INSTRUCTION`, `APPENDS_SYSTEM_PROMPT` (class vars on the harness).

### Authoring a harness

You rarely need this — a custom harness is for a rollout loop the built-ins can't express
(context compaction, subagents, a bespoke agent CLI). Define a `HarnessConfig` (its `id` plus
any knobs, which surface as `--harness.*`), subclass `vf.Harness[ConfigT]`, declare the
capability flags, and implement `launch` — it drives the model however it likes and returns the
program's result (the base `run` wraps it and errors on a non-zero exit). Export the harness
class via `__all__`.

A harness never builds the trace itself: it just points *a program* at `endpoint` (authorized
with `secret`), and the interception server records every call. The program can be any
executable the runtime can run — an agent CLI, a binary, a script — **as long as it makes its
model requests in one of the supported dialects** (chat-completions, Responses, ...); that's
the whole contract. For a self-contained chat loop it's usually a single-file uv script
(`runtime.run_uv_script`, so the harness needs only `uv` in the runtime); otherwise launch your
binary with `runtime.run(...)`. `resolve_prompt(trace.task)` gives the `(system, instruction)`
to seed it, and `mcp_urls` are the task's tool servers.

```python
import verifiers.v1 as vf

PROGRAM = (Path(__file__).parent / "program.py").read_text()  # a uv script, deps = ["openai"]


class MyHarnessConfig(vf.HarnessConfig):
    id: str = "my-harness"
    enable_bash: bool = False        # a harness-specific knob; surfaces as --harness.enable-bash


class MyHarness(vf.Harness[MyHarnessConfig]):
    SUPPORTS_TASK_TOOLS = True
    SUPPORTS_USER_SIM = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls) -> vf.ProgramResult:
        system, instruction = self.resolve_prompt(trace.task)
        env = {"OPENAI_BASE_URL": endpoint, "OPENAI_API_KEY": secret,
               "OPENAI_MODEL": ctx.model, "SYSTEM_PROMPT": system or "",
               "ENABLE_BASH": "1" if self.config.enable_bash else "0"}
        return await runtime.run_uv_script(PROGRAM, args=[instruction], env=env)


__all__ = ["MyHarness"]   # vf resolves the harness by finding this Harness subclass
```

Copy `environments/compact` (a context-rewrite loop) as a starting point.

## Runtimes

The same `Runtime` contract backs the harness (`--harness.runtime`), a task's tools
(`--taskset.tools.runtime`), and the user simulator:

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # local process
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # local container (eval default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type prime       # remote prime sandbox
uv run eval gsm8k-v1 -n 1 --harness.runtime.type modal       # remote modal sandbox
```

## Evals

```bash
uv run eval gsm8k-v1 -n 5 -r 3 \
  --max-turns 8 --max-total-tokens 8192 \                          # per-rollout budgets
  --retries.model.max-retries 3 --retries.runtime.max-retries 3 \  # retry one failed call
  --retries.rollout.max-retries 3 --retries.rollout.include ProgramError \  # retry a whole rollout, by error type
  --timeout.rollout 600 --timeout.scoring 120                      # per-stage wall-clock caps (seconds)
```

Common aliases: `-m`/`--model`, `-n`/`--num-tasks`, `-r`/`--num-rollouts`,
`-c`/`--max-concurrent`, `-s`/`--shuffle`, `-o`/`--output-dir`, `-v`/`--verbose`,
`--no-rich` (disable the live dashboard).

- **Sampling** — set provider-neutral generation knobs under `sampling`, for example
  `--sampling.temperature 0 --sampling.max-tokens 2048 --sampling.reasoning-effort medium`,
  or:

  ```toml
  [sampling]
  temperature = 0
  max_tokens = 2048
  reasoning_effort = "medium"
  ```

  The active dialect maps the string field `reasoning_effort` to the top-level
  `reasoning_effort` field for chat-completions, `reasoning.effort` for Responses, or
  `output_config.effort` for Anthropic Messages.
- **Configs** — a saved run is `uv run eval @ config.toml` (the taskset/harness `id`s live in
  the file); CLI flags still override. `--dry-run` writes the resolved `config.toml` without
  running. Logs are teed to `<output_dir>/eval.log`.
- **Resume** — `uv run eval --resume <output-dir>` re-runs only the missing/errored rollouts
  of a previous run.
- **Clients** — eval (default) is a plain chat-completions relay. `--client.type train`
  tokenizes client-side so each node carries the exact `token_ids` / `mask` / `logprobs`
  (needs a vLLM engine via `--client.base-url`).
- **Validate** — `uv run validate gsm8k-v1` runs each taskset's `validate` hook (model-free
  gold check), no model needed.

## Training

prime-rl consumes the same env over the env-server, so a training env is the eval config in
TOML form. In a prime-rl config:

```toml
[[orchestrator.train.env]]
name    = "gsm8k"
taskset = { id = "math-env-v1", dataset_name = "..." }              # any v1 taskset id
harness = { id = "default", enable_bash = false, runtime = { type = "subprocess" } }
timeout = { scoring = 10 }                                          # per-stage cap (default: no limit)
# pool  = { type = "elastic", max_workers = 8, multiplex = 128 }    # env-server pool (default elastic, self-sizing)
```

`[orchestrator.renderer]` is required (set `name = "auto"` or a specific renderer) — the
renderer tokenizes rollouts into training samples. 

## Backwards compatibility

A classic v0 `verifiers.load_environment` env runs through the v1 CLIs via the legacy bridge —
its rollouts mapped to v1 `Trace`s. Use `--id` instead of a `taskset`:

```bash
uv run eval --id reverse-text -n 2                                  # eval a v0 env
uv run eval --id reverse-text --args.num_train_examples 50          # v0 construction args
```
