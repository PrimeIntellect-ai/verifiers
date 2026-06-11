# verifiers.v1

The next version of [verifiers](https://github.com/PrimeIntellect-ai/verifiers) — a
clean-slate, heavily-typed, minimal core that carries forward the proven high-level
abstractions and on-disk output. Everything is pydantic-typed; `import verifiers.v1 as vf`.

## Highlights

- **Composable taskset × harness** — a taskset (data + scoring) is fully decoupled from the
  harness (the program that drives the rollout); any taskset runs under any harness
  (`default` / `rlm` / `compact` / your own), selected by id.
- **Swappable runtime** — the harness, its tools, and the user simulator all run behind one
  `Runtime` contract, in `subprocess` / `docker` / `prime` / `modal`.
- **Fully typed** — pydantic end-to-end (`Task` / `Trace` / configs); no loose
  `dict` / `object` / `cast`.
- **Minimal & pythonic** — the high-level abstractions without the implementation bulk;
  plain classes + decorators (`@vf.reward` / `@vf.metric` / ...).
- **Training-ready traces** — exact token ids + logprobs straight from an agentic rollout
  (renderer client); one training sample per branch, recovered for compaction / subagents.
- **Delta-native trace graph** — each message is stored once as a node linked to its
  predecessor, so a trace's size is linear in turns, not quadratic; branches fall out of
  walking the graph, and a training sample is a cheap concat of node tokens along a path.
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

Everything is typed config, so the advanced knobs — per-rollout budgets, retries, and
wall-clock timeouts — are all CLI flags (or TOML):

```bash
uv run eval gsm8k-v1 -n 5 -r 3 \
  --max-turns 8 --max-total-tokens 8192 \        # per-rollout budgets (also --max-{input,output}-tokens)
  --retries.model.max-attempts 3 --retries.runtime.max-attempts 3 \  # retry a single model/runtime call
  --retries.rollout.max-attempts 3 --retries.rollout.include ProgramError \  # retry a whole rollout, by exception type
  --timeout.rollout 600 --timeout.scoring 120        # wall-clock caps, in seconds
```

Common knobs have short aliases:

| alias | long               | meaning                                    |
| ----- | ------------------ | ------------------------------------------ |
| `-m`  | `--model`          | model id                                   |
| `-n`  | `--num-tasks`      | how many tasks to evaluate                 |
| `-s`  | `--shuffle`        | shuffle before the `-n` slice              |
| `-r`  | `--num-rollouts`   | rollouts per task                          |
| `-c`  | `--max-concurrent` | max rollouts in flight                     |
| `-v`  | `--verbose`        | debug logging                              |
| `-o`  | `--output-dir`     | where to write results                     |
|       | `--no-rich`        | disable the live dashboard (on by default) |

## Tasksets & harnesses

Tasksets (data + scoring) and harnesses (the rollout driver) are packages selected by `id`,
and live in two places:

- **`packages/`** — shipped, installed by default. Commonly-used **harnesses** (`default`,
  `rlm`) and **taskset integrations** that wrap a whole benchmark family (`harbor-v1` — the
  agentic-benchmark registry; `textarena-v1` — TextArena games). Use them by id.
- **`examples/`** — small reference implementations to copy when **authoring your own**,
  split by kind into `examples/tasksets/` and `examples/harnesses/`. Each shows one pattern.

Taskset examples (`examples/tasksets/`):

| example | pattern it shows |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1`, `aime24-v1`, `math-env-v1` | single-turn + in-runtime scoring (a `@reward` uv script) |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | a multi-turn, stateful task driven by a `vf.User` simulator |
| `glossary-v1` | a custom **colocated** tool server |
| `wikispeedia-v1` | a tool server in its **own per-rollout** runtime |
| `wiki-search-v1` | a **shared** tool server (built once for the eval) + an LLM judge |
| `deepwiki-v1` | an **existing remote** tool server, by URL |
| `wordle-v1` | configuring the vendored `textarena-v1` integration (user simulator) |
| `terminal-bench-2-v1` | configuring the vendored `harbor-v1` integration |

Harness examples (`examples/harnesses/`):

| example | pattern it shows |
| --- | --- |
| `compact` | context compaction → branching trajectories |

## Patterns

### Swappable harness

The program that drives the rollout — same taskset, different driver:

```bash
uv run eval gsm8k-v1 -n 1                   # default: a tiny OpenAI chat loop (bash tool opt-in)
uv run eval gsm8k-v1 -n 1 --harness.id rlm  # the rlm harness
```

### Swappable runtime

*Where* code runs, behind one `Runtime` contract — the same contract backs the harness
(`--harness.runtime`), a task's own tool servers (`--taskset.tools.runtime`), and the user
simulator (all structurally MCP servers in a runtime):

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # harness: local process
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # harness: local container (default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type prime       # harness: remote Prime sandbox
uv run eval gsm8k-v1 -n 1 --harness.runtime.type modal       # harness: remote Modal sandbox
```

Remote sandboxes are named after the rollout id (greppable in `prime sandbox list` /
`modal`), and the framework manages each runtime's full lifecycle — provisioning through
guaranteed cleanup of its resources, even on exit/interrupt.

### Tools

A taskset exposes a task's tools via `tools` (MCP servers launched in the runtime);
**placement** is config on `taskset.tools` and reachability (localhost / tunnel / native
sandbox expose) is resolved automatically. A tool server is a single-file uv script (only runtime dep: `uv`), so a
colocated or own-runtime tool runs in any runtime. The tool examples each show one
placement:

```bash
uv run eval glossary-v1 -n 1     # colocated — in the harness's own runtime, localhost (default)
uv run eval wikispeedia-v1 -n 1 --taskset.min-dist 2 --taskset.max-dist 2  # its own per-rollout runtime
uv run eval wiki-search-v1 -n 1  # shared — one instance built once for the whole eval
uv run eval deepwiki-v1 -n 1     # an existing remote server, by URL
```

### Scoring

Rewards / metrics are decorated methods on the taskset:

```bash
uv run eval gsm8k-v1 -n 1            # runtime scoring: a @vf.reward runs a math-verify uv script
                                     # IN the rollout's runtime (its deps never touch the eval process)
uv run eval code-golf-v1 -n 1 -r 2  # group rewards: a @vf.group_reward scores N rollouts together
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
uv run eval gsm8k-v1 -n 1                          # openai (default): text in / text out
uv run eval gsm8k-v1 -n 1 --client.type renderers \  # renderers: client-side tokenization →
  --client.base-url http://localhost:8000/v1          # token-in/out traces (needs a vLLM engine)
```

With `renderers`, each graph node carries the exact tokens the engine saw — `token_ids`
plus a per-token trainable `mask` and `logprobs` — so concatenating a branch's nodes is a
ready training sample, straight from an agentic rollout with zero agent changes. (When the
engine returns ids on the response itself, the openai client picks them up too — no
renderer required.)

### Limits & retries

Framework-enforced budgets, applied between turns (so they hold for any harness), plus
retries at two granularities: per-call (model + runtime, default 3 attempts — reruns just
the failed call, keeping the rollout's progress) and whole-rollout (default off):

```bash
uv run eval gsm8k-v1 -n 1 --max-turns 8                  # cap model turns
uv run eval gsm8k-v1 -n 1 --max-total-tokens 8192        # cap prompt+completion tokens
                                                          # (also --max-input-tokens / --max-output-tokens)
uv run eval gsm8k-v1 -n 1 --retries.model.max-attempts 5 --retries.runtime.max-attempts 5  # per-call retries
uv run eval gsm8k-v1 -n 1 --retries.rollout.max-attempts 3 --retries.rollout.include ProgramError  # whole-rollout, by exception type
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
