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
  `codex`); you only write your own if you need a custom rollout loop. Any taskset runs under
  any harness.
- **Runtime** — *where* the harness (and the taskset's tools / user simulator) executes:
  `subprocess` / `docker` / `prime` / `modal`. **You never write one** — runtimes ship with the
  framework behind one `Runtime` contract and compose with any taskset/harness; you just choose
  where code runs.

The output of a rollout is a `Trace` — a delta-native message graph (one node per message)
whose root→leaf paths are `Branch`es, each a ready training sample (token ids + logprobs +
trainable mask). The same `Trace` is what eval scores, the dashboard renders, and prime-rl
trains on.

## Install & first run

```bash
uv sync                          # core + shipped packages + examples
uv run eval gsm8k-v1 -n 5 -r 3   # 5 tasks, 3 rollouts each; default harness, docker runtime
uv run eval -h                   # typed help (lists local tasksets + harnesses)
```

Everything below has a CLI flag *and* a TOML equivalent (`uv run eval @ config.toml`); the
flag names are the dotted config path (`--harness.runtime.type docker`).

## Authoring a taskset

A taskset is a package selected by `id`. Copy the closest `examples/tasksets/<name>` and edit.
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


def load_taskset(config: ReverseConfig) -> ReverseTaskset:
    return ReverseTaskset(config=config)
```

`Task` is a frozen pydantic model. Beyond your own fields it carries `instruction: str |
Messages` (a single prompt or a seeded conversation — images go here as content parts),
`system_prompt`, `image` (forces a container runtime), `workdir`, `resources`
(`cpu`/`memory`/`gpu`/`disk`), and per-task timeout overrides.

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
- `finalize(task, trace, runtime)` — after the harness, before scoring (apply a diff, snapshot state).
- `validate(task, runtime)` — model-free gold check (does the reference solution pass?), run by `uv run validate`.

### Tools and user simulators

A taskset exposes a task's tools via `tools(task) -> list[vf.Tools]` (each an MCP server — a
single-file uv script, so it runs in any runtime). Placement is config on `taskset.tools`:

| placement | how |
| --- | --- |
| colocated (default) | in the harness's own runtime, reached on localhost |
| own per-rollout runtime | `taskset.tools.runtime = {...}`, reached over a tunnel |
| shared | one instance built once for the whole eval |
| remote | an existing server, by URL |

A multi-turn, stateful task drives the conversation with a user simulator —
`user(task) -> vf.User | None` (structurally a tool server; the framework calls it after each
assistant turn for the next user message + a done flag). `UserConfig.colocated` controls
whether it runs in the harness's runtime or its own.

### Learn from the examples

`examples/tasksets/` is the reference library — each shows one pattern:

| example | pattern |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1`, `aime24-v1`, `math-env-v1` | single-turn + in-runtime scoring (a `@reward` uv script) |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | multi-turn, stateful, driven by a `vf.User` simulator |
| `glossary-v1` | a colocated tool server |
| `wikispeedia-v1` | a tool server in its own per-rollout runtime |
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
| `compact` | rewrites context each turn → one branch per turn (handles subagents) |
| `codex` | the Codex CLI (Responses dialect + SSE relay) |

```bash
uv run eval gsm8k-v1 -n 1                    # default harness
uv run eval gsm8k-v1 -n 1 --harness.id rlm   # same taskset, different driver
```

**Capability flags** gate which tasksets a harness can run; an incompatible pairing fails
fast at load instead of mis-running. A custom harness declares them as class vars:
`SUPPORTS_TASK_TOOLS`, `SUPPORTS_USER_SIM`, `SUPPORTS_MESSAGE_INSTRUCTION`,
`APPENDS_SYSTEM_PROMPT`. To write one, subclass `vf.Harness[ConfigT]`, set the flags, and
implement `run(...)` (typically launching a program with `runtime.run_uv_script`); export
`load_harness(config)`. See `examples/harnesses/compact`.

## Runtimes

The same `Runtime` contract backs the harness (`--harness.runtime`), a task's tools
(`--taskset.tools.runtime`), and the user simulator:

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # local process
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # local container (eval default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type prime       # remote Prime sandbox
uv run eval gsm8k-v1 -n 1 --harness.runtime.type modal       # remote Modal sandbox
```

Remote resources are named after the rollout id (greppable in `prime sandbox list` /
`modal`), and the framework guarantees cleanup even on interrupt. Per-runtime knobs include
unified resources (`cpu` / `memory` / `gpu` like `"A100:2"` / `disk` / `timeout`), prime
`labels`, and creation-rate limits (`creates_per_sec` / `creates_per_min`).

## Running evals

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
- **Serve** — `uv run serve gsm8k-v1` serves the same env over ZMQ as an env server that an
  orchestrator (or any `EnvClient`) drives by task index.

## Training with prime-rl

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
renderer tokenizes rollouts into training samples. Multimodal training works end to end
(see `configs/v1/multimodal_color_codeword.toml`); pixel tensors ride the trace to the
trainer but are kept out of the on-disk rollout dumps.

## Backwards compatibility

A classic v0 `verifiers.load_environment` env runs through the v1 CLIs via the legacy bridge —
its rollouts mapped to v1 `Trace`s. Use `--id` instead of a `taskset`:

```bash
uv run eval --id reverse-text -n 2                                  # eval a v0 env
uv run serve --id reverse-text                                      # serve it over ZMQ
uv run eval --id reverse-text --args.num_train_examples 50          # v0 construction args
```

## Ids and the Hub

A taskset/harness/env `id` is `name`, `org/name`, or `org/name@version`. A bare local name
must be importable; an `org/...` id installs on demand from the Environments Hub. Built-in
ids (`default`, `rlm`, `harbor-v1`, ...) resolve to the shipped packages.
