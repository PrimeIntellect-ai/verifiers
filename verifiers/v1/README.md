# verifiers.v1

The next version of [verifiers](https://github.com/PrimeIntellect-ai/verifiers) —
**agentic-native, with a composable taskset × harness × runtime core**. A clean-slate,
heavily-typed rewrite that carries forward the proven high-level abstractions, with a 
tighter type contract. `import verifiers.v1 as vf`.

## Highlights

- **Task-first** — the task is the unit of work: data on the (frozen, serializable)
  instance, behavior — rewards, stops, lifecycle hooks, tools — on the class. A taskset is
  just the factory (point it at a dataset); a topology mints the same first-class tasks
  dynamically. Any task runs under any harness (`default` / `rlm` / `codex` / your own)
- **Swappable runtime** — the harness, tools, and user simulators all run behind one
  `Runtime` contract, in `subprocess` / `docker` / `prime` / `modal` / ...
- **First-class branching rollouts** — a rollout isn't assumed linear: context compaction and
  subagents are native. Each branch (a root→leaf path through the trace graph) is its own
  training sample, so a compacting or multi-agent rollout trains end to end.
- **Multi-agent topologies** — compose agents over episodes: a `Topology` turns one agent's
  trace into the next agent's task, fans solvers out, and flows rewards backwards (a proposer
  scored by its solvers' pass rate, a solver scored by an agentic judge). Each instance
  serializes as an *agent graph*: an instance record with its parent-linked traces nested.
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

Harbor's optional Python package and CLI require Python 3.12+:

```bash
uv sync --python 3.12 --extra harbor
```

## Quickstart

```bash
uv run init my-task-v1           # scaffold a new environment (--add-tool/--add-user/--add-harness)
uv run eval gsm8k-v1 -n 5 -r 3   # single-turn math; default harness; subprocess runtime
uv run validate gsm8k-v1 -n 5    # model-free gold + setup-only checks (--only-gold / --only-setup)
uv run debug gsm8k-v1 -n 1 --command 'pwd'  # setup, run one shell action, save traces
uv run eval -h                   # typed help (+ the local example tasksets/harnesses)
```

Everything is typed config, so the advanced knobs — budgets, retries, and
wall-clock timeouts — are all framework-enforced and apply to any environment: 

```bash
uv run eval gsm8k-v1 -n 5 -r 3 \
  --max-turns 8 --max-total-tokens 8192 \
  --retries.rollout.max-retries 3 --retries.rollout.include SandboxError \
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

Taskset examples (the `*_v1` packages under `environments/`):

| example | pattern it shows |
| --- | --- |
| `reverse-text-v1` | the minimal single-turn taskset |
| `gsm8k-v1` | single-turn + in-runtime scoring |
| `code-golf-v1` | group rewards (`@group_reward` over a task's N rollouts) |
| `alphabet-sort-v1` | a multi-turn, stateful task driven by a `vf.User` simulator |
| `glossary-v1` | a custom **colocated** tool server |
| `wiki-search-v1` | a **shared** tool server (lazily built once per run) + a `vf.Judge` reward |
| `deepwiki-v1` | an **existing remote** tool server, by URL |
| `wordle-v1` | configuring the vendored `textarena` integration |
| `proposer-solver-v1` | a multi-agent **topology**: proposer → n solvers (fan-out), deferred rewards |
| `writer-editors-v1` | a multi-agent **topology**: rounds + fan-in (writer → n editors → revision), one `vf.Judge` verdict rewarding every trace |
| `chess-v1` | **sessions**: two live episodes play chess against each other; host-side board as referee |
| `debate-v1` | **sessions, N-ary**: n concurrent debater episodes argue and peer-vote |

Harness examples (under `environments/`):

| example | pattern it shows |
| --- | --- |
| `compact` | context compaction → branching trajectories |

## Patterns

### Swappable harness

The program that drives the rollout — same taskset, different driver:

```bash
uv run eval gsm8k-v1 -n 1                     # default harness (fallback): a bash + edit agent
uv run eval gsm8k-v1 -n 1 --harness.id null   # bare chat loop, no local tools (MCP tools only)
uv run eval gsm8k-v1 -n 1 --harness.id rlm    # the rlm harness
uv run eval gsm8k-v1 -n 1 --harness.id codex  # the codex harness
```

The same drivers on an agentic terminal task — harbor's `hello-world`. The task acts on a
filesystem, so run it under a containerized runtime: `docker` locally, or a remote `prime` /
`modal` sandbox (not the default `subprocess`). 

```bash
uv run eval harbor -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id default         # the bash + edit agent
uv run eval harbor -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id mini-swe-agent  # the mini-swe-agent CLI
uv run eval harbor -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id rlm             # the rlm CLI agent
uv run eval harbor -n 1 --taskset.ignore-dockerfile --harness.runtime.type docker --harness.id codex           # the codex CLI agent
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

### Multi-agent topologies

A *topology* composes episodes — one agent consuming one task and producing one trace — into
a multi-agent interaction, as plain imperative code (`go`): loops are rounds, gathers are
fan-out. An agent is pure routing (harness × model/client), so a non-trainable judge can run
a stronger model on a plain relay while the solver trains; tasks — carrying their own rewards
and hooks — are minted by the seed factory (`--topology.taskset.id`) or in `go` itself. Each
instance persists as one *agent graph* record: parent-linked traces, nested.

```bash
# any taskset, LLM-judged: solver → judge, verdict lands on the solver's trace
# (the judge is fixed to the in-process `direct` harness — an episode ≈ one API call)
uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
# judge as a real agent: the solver's whole trace is uploaded into the judge's runtime
# and the judge (bash+edit `default` harness, configurable) investigates it with tools
uv run eval --topology.id agentic-judge --topology.taskset.id gsm8k-v1 \
  --topology.judge.model deepseek/deepseek-v4 -n 4
# proposer writes questions, n solvers race them, proposer scored by their pass rate
uv run eval --topology.id proposer-solver-v1 -n 3
# rounds + fan-in: writer drafts, n editors critique, the writer revises; one judge call
# compares first draft to final and the same reward lands on every trace
uv run eval --topology.id writer-editors-v1 -n 3
# back-and-forth: two live sessions play chess (each agent is the other's user turn)
uv run eval --topology.id chess-v1 -n 1 --max-turns 64 --timeout.rollout 900 --sampling.max-tokens 4096
```

### Tools

A task may expose its own tools (`Task.load_tools`); placement is the taskset's `tools` config knob, so it stays CLI-tunable. Tools go beyond the tools shipping natively with
the harness as MCP servers. Its placement (separate runtime or colocated with
harness) is configurable on `taskset.tools` and reachability is handled resolved
automatically. Tools only run under a harness with `SUPPORTS_MCP` (the `default`
harness has it; `rlm` doesn't) — an incompatible pairing is refused at load. The tool examples
each show one placement:

```bash
uv run eval glossary-v1 -n 1     # own host runtime (the default placement)
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

uv run eval gsm8k-v1 -n 1 --retries.rollout.max-retries 3 --retries.rollout.include SandboxError  # whole-rollout, by exception type (per-call model/runtime retries are owned by the SDKs)
```

### Integrations

Some tasksets wrap a whole benchmark family rather than a single task — shipped, installed by
default. For example, `textarena` (TextArena games) and `harbor` (the agentic-
benchmark registry). Harbor is the showcase: it pulls tasks straight from the Harbor registry,
each in its own declared, pullable container image — e.g. Terminal-Bench 2:

```bash
uv run eval harbor --taskset.dataset terminal-bench/terminal-bench-2 -n 10 --harness.id rlm
```

Harbor registry selectors pass through as taskset config. For the
`research-environments` registry:

```bash
uv run eval harbor --taskset.dataset general-agent@2026-06-25 \
  --taskset.repo PrimeIntellect-ai/research-environments@REF_OR_SHA

uv run eval harbor --taskset.dataset general-agent@2026-06-25 \
  --taskset.registry-path ./registry.json

uv run eval harbor --taskset.dataset general-agent@2026-06-25 \
  --taskset.registry-url https://raw.githubusercontent.com/PrimeIntellect-ai/research-environments/REF_OR_SHA/registry.json
```

## Backwards compatibility

The v0 framework is untouched — the classic `verifiers` API and its entrypoints (`vf-eval`,
...) keep working exactly as before; v1 lives alongside it as `verifiers.v1`. On top of
that, a v0 `verifiers.load_environment` env runs through the v1 CLIs too, via the legacy
bridge — its rollouts mapped to v1 `Trace`s. Set `--id` (instead of a `taskset`) on either
`eval` or `serve`:

```bash
uv run eval --id reverse-text -n 2     # eval a v0 env
uv run eval --id reverse-text --args.dataset_split train \
  --extra-env-kwargs.max-total-completion-tokens 256   # construction + post-load kwargs
```
