# verifiers.v1

The next version of [verifiers](https://github.com/PrimeIntellect-ai/verifiers) —
**agentic-native, with a composable taskset × harness × runtime core**. A clean-slate,
heavily-typed rewrite that carries forward the proven high-level abstractions, with a 
tighter type contract. `import verifiers.v1 as vf`.

## Highlights

- **Composable taskset × harness** — a taskset (data + scoring) is fully decoupled from the
  harness (the program driving the rollout); any taskset runs under any harness
  (`default` / `rlm` / `codex` / your own)
- **Swappable runtime** — the harness, tools, and user simulators all run behind one
  `Runtime` contract, in `subprocess` / `docker` / `prime` / `modal` / ...
- **First-class branching rollouts** — a rollout isn't assumed linear: context compaction and
  subagents are native. Each branch (a root→leaf path through the trace graph) is its own
  training sample, so a compacting or multi-agent rollout trains end to end.
- **Fully typed** — pydantic end-to-end (`Task` / `Trace` / configs); no loose
  `dict` / `object` / `cast`.
- **Minimal & pythonic** — the high-level abstractions without the implementation bulk;
  plain classes + decorators (`@vf.reward` / `@vf.metric` / ...).
- **Training-ready traces** — exact token ids + logprobs straight from an agentic rollout
  (renderer client); one training sample per branch, recovered for compaction / subagents.
- **Hub-native + v0-compatible** — ids install on demand from the Environments Hub, and
  classic v0 envs run through the same CLIs via a bridge.

## Install

```bash
uv sync   # core + the shipped packages + examples (eval, serve, all runtimes)
```

## Quickstart

```bash
uv run eval gsm8k-v1 -n 5 -r 3   # single-turn math; default harness; docker runtime
uv run eval -h                   # typed help (+ the local example tasksets/harnesses)
```

Everything is typed config, so the advanced knobs — budgets, retries, and
wall-clock timeouts — are all framework-enforced and apply to any environment: 

```bash
uv run eval gsm8k-v1 -n 5 -r 3 \
  --max-turns 8 --max-total-tokens 8192 \
  --retries.model.max-retries 3 --retries.runtime.max-retries 3 \
  --retries.rollout.max-retries 3 --retries.rollout.include ProgramError \
  --timeout.setup 120 --timeout.rollout 600 --timeout.finalize 120 --timeout.scoring 120
```

Common knobs have short aliases:

| alias | long               | meaning                       | default                      |
| ----- | ------------------ | ----------------------------- | ---------------------------- |
| `-m`  | `--model`          | model id                      | `deepseek/deepseek-v4-flash` |
| `-n`  | `--num-tasks`      | how many tasks to evaluate    | all tasks                    |
| `-s`  | `--shuffle`        | shuffle before the `-n` slice | off                          |
| `-r`  | `--num-rollouts`   | rollouts per task             | `1`                          |
| `-c`  | `--max-concurrent` | max rollouts in flight        | `128`                        |
| `-v`  | `--verbose`        | debug logging                 | off (info)                   |
| `-o`  | `--output-dir`     | where to write results        | a fresh per-run dir          |
|       | `--no-rich`        | disable the live dashboard    | dashboard on                 |

## Tasksets & harnesses

Tasksets (data + scoring) and harnesses (the rollout driver) are Python packages 
and live in two places:

- **`packages/`** — shipped, installed by default. Commonly-used **harnesses** (`default`,
  `rlm`, `codex`, ...) and **taskset integrations** that wrap a whole benchmark family (`harbor-v1` — 
  the agentic-benchmark registry; `textarena-v1` — TextArena games).
- **`examples/`** — small reference implementations to copy when **authoring your own**,
  split by kind into `examples/tasksets/` and `examples/harnesses/`. Each shows one pattern.

Taskset examples (`examples/tasksets/`):

| example | pattern it shows |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1`, `aime24-v1`, `math-env-v1` | single-turn + in-runtime scoring |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | a multi-turn, stateful task driven by a `vf.User` simulator |
| `glossary-v1` | a custom **colocated** tool server |
| `wikispeedia-v1` | a tool server in its **own per-rollout** runtime |
| `wiki-search-v1` | a **shared** tool server (built once for the eval) + an LLM judge |
| `deepwiki-v1` | an **existing remote** tool server, by URL |
| `wordle-v1` | configuring the vendored `textarena-v1` integration |
| `terminal-bench-2-v1` | configuring the vendored `harbor-v1` integration |

Harness examples (`examples/harnesses/`):

| example | pattern it shows |
| --- | --- |
| `compact` | context compaction → branching trajectories |

## Patterns

### Swappable harness

The program that drives the rollout — same taskset, different driver:

```bash
uv run eval gsm8k-v1 -n 1                     # default: a tiny OpenAI chat loop (bash tool opt-in)
uv run eval gsm8k-v1 -n 1 --harness.id rlm    # the rlm harness
uv run eval gsm8k-v1 -n 1 --harness.id codex  # the codex harness
```

### Swappable runtime

*Where* code runs, behind one `Runtime` contract — the same contract backs the harness
(`--harness.runtime`), a task's own tool servers (`--taskset.tools.runtime`), and the user
simulator (`--taskset.user.runtime`) — all structurally MCP servers in a runtime:

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # local process
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # local container (default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type prime       # remote prime sandbox (requires auth)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type modal       # remote modal sandbox (requires auth)
```

The framework manages each runtime's full lifecycle — provisioning through
guaranteed cleanup of its resources, even on exit/interrupt.

### Tools

A taskset may expose task-specific tools beyond the tools shipping natively with
the harness as MCP servers. Its placement (separate runtime or colocated with
harness) is configurable on `taskset.tools` and reachability is handled resolved
automatically. Tools only run under a harness with `SUPPORTS_TASK_TOOLS` (the `default`
harness has it; `rlm` doesn't) — an incompatible pairing is refused at load. The tool examples
each show one placement:

```bash
uv run eval glossary-v1 -n 1     # colocated — in the harness's own runtime, localhost (default)
uv run eval wikispeedia-v1 -n 1 --taskset.min-dist 2 --taskset.max-dist 2  # its own per-rollout runtime
uv run eval wiki-search-v1 -n 1  # shared — one instance built once for the whole eval
uv run eval deepwiki-v1 -n 1     # an existing remote server, by URL
```

### User simulator

A stateful, multi-turn task can drive the *user* side of the conversation itself: a taskset's
`user(task)` returns a `vf.User` — structurally a tool server, but the framework drives it,
calling it after each assistant turn for the next user message(s) plus a done flag, then
re-prompting. The harness never knows — it just sees another user turn — but it must support
one (`SUPPORTS_USER_SIM`; the `default` harness has it, `rlm` doesn't). Placement is config on
`taskset.user` — colocated in the harness's runtime by default, or its own via
`--taskset.user.runtime`:

```bash
uv run eval alphabet-sort-v1 -n 1   # stateful multi-turn — the user sim injects each next turn
uv run eval wordle-v1 -n 1          # a TextArena game, driven by the same user-sim machinery
```

### Branching trajectories

A rollout isn't always linear. The `compact` harness rewrites its context every turn — a
fresh `[system, user]` carrying its running notes plus the last tool output — so each turn
is its own *branch*. Branches fall out of the message graph — each leaf's root→leaf path is
one branch, exposed by `trace.branches` / `num_branches` (a linear harness is one branch;
the compact harness is one per turn — it also handles subagents):

```bash
uv run eval wiki-search-v1 -n 1 --harness.id compact  # fresh prompt each turn → num_branches == turns
```

### Clients

The client sits *behind* the interception server, so the harness only ever speaks plain
chat-completions:

```bash
uv run eval gsm8k-v1 -n 1                          # eval (default): relay, text in / text out
uv run eval gsm8k-v1 -n 1 --client.type train \      # train: client-side tokenization →
  --client.base-url http://localhost:8000/v1          # token-in/out traces (needs a vLLM engine)
```

With `train`, each graph node carries the exact tokens the engine saw — `token_ids`
plus a per-token trainable `mask` and `logprobs` — so concatenating a branch's nodes is a
ready training sample, straight from an agentic rollout with zero agent changes. (When the
engine returns ids on the response itself, the openai client picks them up too — no
renderer required.)

### Budgets

Per-rollout budgets are framework-enforced and checked between turns, so they hold for any
harness: a cap on model turns (`--max-turns`) and three on tokens — `--max-input-tokens`,
`--max-output-tokens`, `--max-total-tokens` (prompt, completion, and the sum). Hitting a cap
cleanly truncates the rollout (`trace.is_truncated`) instead of erroring.

Alongside them, retries at two granularities: per-call (model + runtime, default 3 retries —
reruns just the failed call, keeping the rollout's progress) and whole-rollout (default 1
retry). A retry count of 0 turns a layer off.

```bash
uv run eval gsm8k-v1 -n 1 --max-turns 8                  # cap model turns
uv run eval gsm8k-v1 -n 1 --max-total-tokens 8192        # cap prompt+completion tokens
                                                          # (also --max-input-tokens / --max-output-tokens)
uv run eval gsm8k-v1 -n 1 --retries.model.max-retries 5 --retries.runtime.max-retries 5  # per-call retries
uv run eval gsm8k-v1 -n 1 --retries.rollout.max-retries 3 --retries.rollout.include ProgramError  # whole-rollout, by exception type
```

### First-class Harbor support

Common agentic benchmarks run out of the box: the shipped `harbor-v1` taskset (installed by
default) pulls tasks straight from the Harbor registry via the `harbor` CLI
(`uv tool install harbor`), each in its own declared, pullable
container image — e.g. Terminal-Bench 2 (the `terminal-bench-2-v1` example just pins this):

```bash
uv run eval harbor-v1 --taskset.dataset terminal-bench/terminal-bench-2 -n 10 --harness.enable-bash true
```

Tasks that define their environment with a `Dockerfile` rather than a pullable image (e.g.
SWE-bench) are rejected at load — building Dockerfiles isn't supported here (a locally-built
image isn't pullable by a remote sandbox) — rather than silently scored against a wrong
default image.

### Typed eval CLI

Every CLI flag has a TOML equivalent, and a saved config runs with just `@ file.toml` — the
taskset / harness `id` is read from the file, so no positional id is needed:

```bash
uv run eval @ configs/gsm8k.toml          # ids + knobs from the file
uv run eval @ configs/gsm8k.toml -n 1    # CLI flags still override the file
```

This is the same TOML-driven shape prime-rl consumes (ids live in the config, resolved by
`EnvConfig`). The other CLI is `uv run serve` — the same env, served over ZMQ as an env
server the orchestrator (or any `EnvClient`) drives by task index.

### Backwards compatibility

The v0 framework is untouched — the classic `verifiers` API and its entrypoints (`vf-eval`,
...) keep working exactly as before; v1 lives alongside it as `verifiers.v1`. On top of
that, a v0 `verifiers.load_environment` env runs through the v1 CLIs too, via the legacy
bridge — its rollouts mapped to v1 `Trace`s. Set `--id` (instead of a `taskset`) on either
`eval` or `serve`:

```bash
uv run eval --id reverse-text -n 2     # eval a v0 env
uv run serve --id reverse-text         # serve a v0 env over ZMQ (the orchestrator can't tell v0 from v1)
uv run eval --id reverse-text --args.num_train_examples 50 \
  --extra-env-kwargs.max-total-completion-tokens 256   # construction + post-load kwargs
```
