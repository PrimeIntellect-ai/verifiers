# verifiers.v1

The next version of [verifiers](https://github.com/PrimeIntellect-ai/verifiers) — a
clean-slate, heavily-typed, minimal core that carries forward the proven high-level
abstractions and on-disk output. Everything is pydantic-typed; `import verifiers.v1 as vf`.

## Highlights

- **Much more minimal** — the high-level abstractions, without the implementation bulk.
- **Fully typed** — pydantic end-to-end (`Task` / `Trace` / configs); no loose
  `dict` / `object` / `cast`.
- **Pythonic API preserved** — plain classes + decorators (`@vf.reward` / `@vf.metric` / ...).
- **Runtime-agnostic execution** — harnesses / tools run in swappable runtimes (`subprocess` /
  `docker` / `prime` / `modal`).
- **Hub-native + v0-compatible** — ids install on demand from the Environments Hub, and
  classic v0 envs run through the same CLI via a bridge.

## Install

```bash
uv sync   # core + the shipped plugins + examples (eval, serve, all runtimes)
```

## Quickstart

```bash
uv run eval gsm8k-v1 --n 5 --r 3   # single-turn math; default harness; docker runtime
uv run eval -h                      # typed help (+ the local example tasksets/harnesses)
```

Common knobs have short aliases:

| alias  | long               | meaning                                    |
| ------ | ------------------ | ------------------------------------------ |
| `--m`  | `--model`          | model id                                   |
| `--n`  | `--num-tasks`      | how many tasks to evaluate                 |
| `--s`  | `--shuffle`        | shuffle before the `--n` slice             |
| `--r`  | `--num-rollouts`   | rollouts per task                          |
| `--c`  | `--max-concurrent` | max rollouts in flight                     |
| `--v`  | `--verbose`        | debug logging                              |
| `--o`  | `--output-dir`     | where to write results                     |
|        | `--no-rich`        | disable the live dashboard (on by default) |

## Patterns

### Swappable harness

The program that drives the rollout — same taskset, different driver:

```bash
uv run eval gsm8k-v1 --n 1                   # default: a tiny OpenAI chat loop (bash tool opt-in)
uv run eval gsm8k-v1 --n 1 --harness.id rlm  # the rlm harness
```

### Swappable runtime

*Where* code runs, behind one `Runtime` contract — the same contract backs both the harness
(`--harness.runtime`) and a task's own tool servers (`--taskset.tools.runtime`):

```bash
uv run eval gsm8k-v1 --n 1 --harness.runtime.type subprocess  # harness: local process
uv run eval gsm8k-v1 --n 1 --harness.runtime.type docker      # harness: local container (default)
uv run eval gsm8k-v1 --n 1 --harness.runtime.type prime       # harness: remote Prime sandbox
uv run eval gsm8k-v1 --n 1 --harness.runtime.type modal       # harness: remote Modal sandbox
```

Remote sandboxes are named after the rollout id (greppable in `prime sandbox list` /
`modal`), and every runtime is freed on exit/interrupt (an atexit backstop catches a
signal that cut a rollout's teardown short).

### Tools

A task declares tool servers (`tool_servers`); **placement** is config on
`taskset.tools` and reachability (localhost / tunnel / native sandbox expose) is resolved
automatically. A tool server is a single-file uv script (only runtime dep: `uv`), so a
colocated or own-runtime tool runs in any runtime. The tool examples each show one
placement:

```bash
uv run eval glossary-v1 --n 1     # colocated — in the harness's own runtime, localhost (default)
uv run eval wikispeedia-v1 --n 1 --taskset.min-dist 2 --taskset.max-dist 2  # its own per-rollout runtime
uv run eval wiki-search-v1 --n 1  # shared — one instance built once for the whole eval
uv run eval deepwiki-v1 --n 1     # an existing remote server, by URL
```

### Scoring

Rewards / metrics are decorated methods on the taskset:

```bash
uv run eval gsm8k-v1 --n 1            # runtime scoring: a @vf.reward runs a math-verify uv script
                                     # IN the rollout's runtime (its deps never touch the eval process)
uv run eval code-golf-v1 --n 1 --r 2  # group rewards: a @vf.group_reward scores N rollouts together
```

### Branching trajectories

A rollout isn't always linear. The `compact` harness rewrites its context every turn — a
fresh `[system, user]` carrying its running notes plus the last tool output — so each turn
is its own *branch*. `branching` recovers them from the flat trajectory and
`trace.branches` / `num_branches` expose it (a linear harness is one branch; the compact
harness is one per turn — it also handles subagents):

```bash
uv run eval wiki-search-v1 --n 1 --harness.id compact  # fresh prompt each turn → num_branches == turns
```

### Clients

The client sits *behind* the interception server, so the harness only ever speaks plain
chat-completions:

```bash
uv run eval gsm8k-v1 --n 1                          # openai (default): text in / text out
uv run eval gsm8k-v1 --n 1 --client.type renderers \  # renderers: client-side tokenization →
  --client.base-url http://localhost:8000/v1          # token-in/out traces (needs a vLLM engine)
```

With `renderers`, each `trace.trajectory[i].tokens` carries the exact `prompt_ids` /
`completion_ids` / `completion_logprobs` the engine saw — training-ready token data
straight from an agentic rollout, with zero agent changes. (When the engine returns ids on
the response itself, the openai client picks them up too — no renderer required.)

### Limits & retries

Framework-enforced budgets, applied between turns (so they hold for any harness), plus
native whole-rollout retries:

```bash
uv run eval gsm8k-v1 --n 1 --max-turns 8                  # cap model turns
uv run eval gsm8k-v1 --n 1 --max-total-tokens 8192        # cap prompt+completion tokens
                                                          # (also --max-input-tokens / --max-output-tokens)
uv run eval gsm8k-v1 --n 1 --retry.attempts 3 --retry.include ProgramError  # retry by exception type
```

### Installable ids (the Hub)

An id is `name` (local), `org/name`, or `org/name@version`. A hub id is installed on demand
(the same path as `prime env install`) before it loads — for the taskset, the harness, or a
v0 env:

```bash
uv run eval org/my-taskset@1.2.0 --n 1   # installed from the Environments Hub, then run
```

### v0 backwards-compat

A classic v0 `verifiers.load_environment` env runs through the same CLI via the legacy
bridge — its rollouts mapped to v1 `Trace`s. Set `--id` (instead of a `taskset`):

```bash
uv run eval --id reverse-text --n 2                       # a local v0 env
uv run eval --id reverse-text --args.num_train_examples 50 \
  --extra-env-kwargs.max-total-completion-tokens 256       # construction + post-load kwargs
```

### User simulation

A taskset can drive the *user* side of a multi-turn conversation. The interception server
injects user turns from a `vf.User`, so the harness is unaware it's talking to a simulator:

```bash
uv run eval @ configs/textarena.toml --n 1   # a user simulator plays the game (TextArena Wordle-v0)
```

### Typed eval CLI

Every CLI flag has a TOML equivalent, and a saved config runs with just `@ file.toml` — the
taskset / harness `id` is read from the file, so no positional id is needed:

```bash
uv run eval @ configs/gsm8k.toml          # ids + knobs from the file
uv run eval @ configs/gsm8k.toml --n 1    # CLI flags still override the file
```

This is the same TOML-driven shape prime-rl consumes (ids live in the config, resolved by
`EnvConfig`). The other CLI is `uv run serve` — the same env, served over ZMQ as an env
server the orchestrator (or any `EnvClient`) drives by task index.

## Open TODOs

- **Iterable taskset** — let `load_tasks` stream tasks instead of returning a `list`
  (system-prompt diversity, privileged information, replay buffers, ...).
- **Multi-agent** — more than one agent per rollout.
- **More agent adapters** — other CLI agents (Claude Code, Codex, ...).
- **Trainer layer (credit assignment)** — an `Episode` (a task's scored rollouts) is the
  largest unit the *evaluator* knows about.
