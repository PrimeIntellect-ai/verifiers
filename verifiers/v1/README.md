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
uv sync   # core + the shipped packages + examples
```

## Quickstart

```bash
uv run init my-task-v1           # scaffold a new environment (--add-tool/--add-user/--add-harness)
uv run eval gsm8k-v1 -n 5 -r 3   # single-turn math; default harness; subprocess runtime
uv run validate gsm8k-v1 -n 5    # model-free: run each task's gold check (the `validate` hook)
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
  `bash`, `rlm`, `codex`, ...) and **taskset integrations** that wrap a whole benchmark family (`harbor-v1` — 
  the agentic-benchmark registry; `textarena-v1` — TextArena games).
- **`environments/`** — small reference implementations to copy when **authoring your own**
  (the `*_v1` tasksets and the `compact` harness), co-located with the standalone v0
  environments. Each shows one pattern.

Taskset examples (the `*_v1` packages under `environments/`):

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

Harness examples (under `environments/`):

| example | pattern it shows |
| --- | --- |
| `compact` | context compaction → branching trajectories |

## Patterns

### Swappable harness

The program that drives the rollout — same taskset, different driver:

```bash
uv run eval gsm8k-v1 -n 1                     # default: bare agent (MCP tools only)
uv run eval gsm8k-v1 -n 1 --harness.id rlm    # the rlm harness
uv run eval gsm8k-v1 -n 1 --harness.id codex  # the codex harness
```

The same drivers on an agentic terminal task — harbor's `hello-world`. The task acts on a
filesystem, so run it under a containerized runtime: `docker` locally, or a remote `prime` /
`modal` sandbox (not the default `subprocess`). 

```bash
uv run eval harbor-v1 -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id bash            # bash-only agent
uv run eval harbor-v1 -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id mini-swe-agent  # the mini-swe-agent CLI
uv run eval harbor-v1 -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id rlm             # the rlm CLI agent
uv run eval harbor-v1 -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id codex           # the codex CLI agent
```

### Swappable runtime

Where code runs, behind one `Runtime` contract — the same contract backs the harness
(`--harness.runtime`), a task's own tool servers (`--taskset.tools.runtime`), and the user
simulator (`--taskset.user.runtime`) — all structurally MCP servers in a runtime:

```bash
uv run eval gsm8k-v1 -n 1 --harness.runtime.type subprocess  # local process (default)
uv run eval gsm8k-v1 -n 1 --harness.runtime.type docker      # local container (requires local docker)
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

### User

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

Real agents rarely keep one ever-growing context — and the two patterns they use both break
a linear trace:

- **Compaction** — when context grows too long, the agent summarizes the history into notes
  and continues from a fresh prompt; each compaction is a new context window.
- **Subagents** — the agent spawns a child to work a subtask on its own context, then folds
  the result back.

v1 handles both natively because the trace is a *graph*, not a list: each fresh context
window (a compaction) or child run (a subagent) is just another **branch** — a root→leaf
path, surfaced as `trace.branches` / `trace.num_branches` (a linear rollout is one branch).
And every branch is an independent training sample, so a compacting or multi-agent rollout
trains end to end with no agent-side changes.

The `compact` example harness is the deliberate stress test: it rewrites its prompt every
turn — a fresh `[system, user]` with only its carried-over notes plus the last tool output —
so each turn becomes its own branch:

```bash
uv run eval wiki-search-v1 -n 1 --harness.id compact  # fresh prompt each turn → num_branches == turns
```

### Clients

The model sits *behind* the interception server: a harness just points an OpenAI- or
Anthropic-style SDK at a localhost endpoint, and the framework intercepts every call. A
**dialect** layer route-detects the wire format the harness speaks — chat-completions,
Responses, or Anthropic messages (streaming and reasoning preserved) — so an off-the-shelf
agent or CLI integrates unchanged whatever SDK it's built on .

Behind that endpoint sit two **clients**, switched with `--client.type`:

```bash
uv run eval gsm8k-v1 -n 1                            # eval (default): a 1:1 relay, text in / text out
uv run eval gsm8k-v1 -n 1 --client.type train \      # train: client-side tokenization via the renderer (requires vllm)
  --client.base-url http://localhost:8000/v1
```

`eval` is the default; `train` is only needed for the prime-rl training
integration (it tokenizes client-side so each branch comes back as a ready
training sample).

### Budgets

The framework enforces every rollout's resource limits itself — between turns and around each
stage — so they hold for any harness or task. Three kinds:

**Caps.** A limit on model turns (`--max-turns`) and three on tokens — `--max-input-tokens`,
`--max-output-tokens`, `--max-total-tokens` (prompt, completion, and the sum), checked between
turns. Hitting a cap cleanly truncates the rollout (`trace.is_truncated`) instead of erroring.

**Timeouts.** Wall-clock caps that bound each rollout stage independently — `--timeout.setup`,
`--timeout.rollout` (the harness run), `--timeout.finalize`, `--timeout.scoring` (seconds;
default no limit). A `rollout` timeout scores what the harness produced so far (like a turn
cap); `setup` / `finalize` / `scoring` timeouts error the rollout.

**Retries.** Two granularities — per-call (model + runtime, default 3, reruns just the failed
call and keeps the rollout's progress) and whole-rollout (default 1). A retry count of 0 turns
a layer off.

```bash
uv run eval gsm8k-v1 -n 1 --max-turns 8                  # cap model turns
uv run eval gsm8k-v1 -n 1 --max-total-tokens 8192        # cap prompt+completion tokens
                                                          # (also --max-input-tokens / --max-output-tokens)

uv run eval gsm8k-v1 -n 1 --timeout.rollout 600 --timeout.scoring 120  # per-stage wall-clock caps (s)

uv run eval gsm8k-v1 -n 1 --retries.model.max-retries 5 --retries.runtime.max-retries 5  # per-call retries
uv run eval gsm8k-v1 -n 1 --retries.rollout.max-retries 3 --retries.rollout.include ProgramError  # whole-rollout, by exception type
```

**Errors.** Expected rollout failures persist on `trace.errors` with their boundary intact:
provider/auth/schema failures use the `Provider*Error` types, agent implementation or exit
failures use `HarnessError`, task-tool construction/server failures use `ToolError`, and
other runtime process/tunnel failures use `ProgramError`. Terminal sandbox failures are
`SandboxTimeoutError` or `SandboxOutOfMemoryError`, both under `SandboxError`. MCP `isError`
tool results remain in-band so the model can recover.

### Integrations

Some tasksets wrap a whole benchmark family rather than a single task — shipped, installed by
default. For example, `textarena-v1` (TextArena games) and `harbor-v1` (the agentic-
benchmark registry). Harbor is the showcase: it pulls tasks straight from the Harbor registry
via the `harbor` CLI (`uv tool install harbor`), each in its own declared, pullable container
image — e.g. Terminal-Bench 2:

```bash
uv run eval harbor-v1 --taskset.dataset terminal-bench/terminal-bench-2 -n 10 --harness.id rlm
```

## Backwards compatibility

The v0 framework is untouched — the classic `verifiers` API and its entrypoints (`vf-eval`,
...) keep working exactly as before; v1 lives alongside it as `verifiers.v1`. On top of
that, a v0 `verifiers.load_environment` env runs through the v1 CLIs too, via the legacy
bridge — its rollouts mapped to v1 `Trace`s. Set `--id` (instead of a `taskset`) on either
`eval` or `serve`:

```bash
uv run eval --id reverse-text -n 2     # eval a v0 env
uv run eval --id reverse-text --args.num_train_examples 50 \
  --extra-env-kwargs.max-total-completion-tokens 256   # construction + post-load kwargs
```
