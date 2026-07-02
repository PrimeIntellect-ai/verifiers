# Training

`verifiers.v1` is designed to produce training-ready traces. The supported native integration is
`prime-rl`: its orchestrator starts the same v1 environment server used by `uv run eval --server`,
requests rollout groups by task index, and trains from the returned `vf.Trace` branches.
The [runtime architecture guide](runtime-architecture.md) shows the orchestrator, environment
workers, interception server, harness sandboxes, inference engine, and trainer as separate
process and network boundaries.

## Why v1 traces are training-ready

A v1 trace is a message graph. Each node can carry:

- the exact token IDs seen or sampled by inference;
- a trainability mask;
- sampled-token log probabilities;
- reasoning, tool calls, multimodal data, and usage;
- rewards, metrics, errors, and timing.

Each root-to-leaf branch is a training sample. Context compaction and subagents therefore remain
representable without flattening the rollout into a lossy transcript.

Training uses the renderer client rather than the plain eval relay. The renderer keeps prior token
prefixes stable across turns and returns exact token data from the inference engine.

## Validate before training

First check the task contract without a model, where the taskset provides a gold validation hook:

```bash
uv run validate my-task-v1 -n 20 --runtime.type subprocess
```

Then run the actual model and inspect traces:

```bash
uv run eval my-task-v1 -m openai/gpt-5-mini -n 20 -r 4 --shuffle
```

Do not begin RL from import tests alone. Verify that:

- successful and unsuccessful samples receive the intended reward;
- tool and user-simulator turns are present in the trace;
- runtime failures are distinguishable from valid zero-reward completions;
- task difficulty produces mixed outcomes for the intended base model;
- every rollout has a clear stop condition and bounded resource use.

## Install `prime-rl`

Use the current `prime-rl` setup instructions. A quick setup is:

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

Manual setup:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git submodule update --init --recursive
uv sync --all-extras
```

`prime-rl` requires NVIDIA GPUs. Its repository is the source of truth for supported models,
distributed layouts, and launch infrastructure.

## V1 environment configuration

Native v1 environments use `taskset` and `harness` objects under both training and periodic eval
sections:

```toml
max_steps = 100
seq_len = 4096

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[orchestrator]
training_mode = "rl"
batch_size = 512
group_size = 16

[orchestrator.renderer]
name = "auto"

[orchestrator.train.sampling]
temperature = 1.0
max_completion_tokens = 2048

[[orchestrator.train.env]]
name = "gsm8k-train"
taskset = { id = "gsm8k-v1", split = "train" }
harness = { id = "default", runtime = { type = "subprocess" } }
timeout = { rollout = 600, scoring = 120 }

[orchestrator.eval]
interval = 20

[orchestrator.eval.sampling]
temperature = 0.0
max_completion_tokens = 2048

[[orchestrator.eval.env]]
name = "gsm8k-test"
taskset = { id = "gsm8k-v1", split = "test" }
harness = { id = "default", runtime = { type = "subprocess" } }
num_examples = 256
group_size = 4

[trainer]

[inference]
```

The environment entries inherit `vf.EnvServerConfig`, so the same v1 fields used by eval are
available:

| Field | Purpose |
| --- | --- |
| `taskset` | Taskset ID and typed taskset-specific fields. |
| `harness` | Harness ID, runtime, environment variables, and disabled harness tools. |
| `timeout` | Per-stage setup, rollout, finalize, and scoring wall-clock limits. |
| `retries.rollout` | Whole-rollout retry policy by captured error type. |
| `max_turns` / token limits | Framework-enforced trajectory budgets. |
| `multiplex` | Rollouts per interception server/tunnel. |
| `pool` | Static or elastic environment worker pool. |
| `address` | Connect to an already-running v1 env server instead of spawning one. |

Do not put native v1 taskset fields in `args`; that mapping is only for bridged v0
`load_environment()` packages.

## Launch

From the `prime-rl` checkout:

```bash
uv run rl @ /path/to/rl.toml --dry-run
uv run rl @ /path/to/rl.toml
```

`--dry-run` resolves the merged configuration and writes process-specific configs without starting
training. Use it after changing taskset, harness, runtime, renderer, or distributed settings.

For split-process or distributed operation, use the dedicated `prime-rl` entrypoints (`inference`,
`trainer`, and `orchestrator`) documented in that repository. Do not replace its inference
entrypoint with a plain `vllm serve`; the Prime entrypoint adds the weight-update and token-generation
interfaces training needs.

## Group size and rewards

`orchestrator.group_size` is the number of rollouts sampled for each task before group-relative
advantage calculation. A per-environment `group_size` overrides it.

Start with at least `8` for group-relative RL; `16` is a common baseline. The useful value depends
on task cost and difficulty:

- Larger groups are more likely to contain mixed rewards and give a useful within-task signal.
- Smaller groups reduce rollout cost and latency but collapse more often to all-success or
  all-failure.
- `batch_size` should accommodate whole groups and be chosen with the trainer's packing and memory
  constraints in mind.

A taskset's `@vf.group_reward` is different: it is environment scoring that compares the sampled
traces before the trainer computes advantages. If a taskset defines one, every rollout for that task
must be served as one group; v1 and `prime-rl` preserve this automatically.

## Difficulty filtering

The most important training diagnostic is the distribution of reward within each rollout group.
If almost every group is all-zero, the task is too hard or broken. If almost every group is
all-one, it is too easy.

`prime-rl` can drop collapsed groups with online difficulty filtering. This exact `0.0`/`1.0`
criterion is naturally suited to binary rewards. For continuous rewards, inspect distributions and
choose task sampling or an explicit filter that matches the reward semantics rather than assuming
binary thresholds are meaningful.

Use a held-out v1 taskset config for periodic eval. A different split is usually just another
instance of the same taskset:

```toml
[[orchestrator.train.env]]
name = "math-train"
taskset = { id = "math-v1", split = "train" }
harness = { id = "default", runtime = { type = "subprocess" } }

[[orchestrator.eval.env]]
name = "math-test"
taskset = { id = "math-v1", split = "test" }
harness = { id = "default", runtime = { type = "subprocess" } }
num_examples = 200
group_size = 4
```

## Multi-environment training

Add multiple `[[orchestrator.train.env]]` tables. Give each a unique `name`. If one environment
should be sampled more often, set `ratio` on every training environment; ratios are relative and
normalized:

```toml
[[orchestrator.train.env]]
name = "math"
ratio = 0.75
taskset = { id = "gsm8k-v1", split = "train" }
harness = { id = "default", runtime = { type = "subprocess" } }

[[orchestrator.train.env]]
name = "tools"
ratio = 0.25
taskset = { id = "wiki-search-v1" }
harness = { id = "default", runtime = { type = "subprocess" } }
```

Environment-specific sampling can be placed under that environment table; otherwise it inherits
`[orchestrator.train.sampling]` or `[orchestrator.eval.sampling]`.

## Agentic and sandbox training

Keep the environment boundary identical between eval and training. Change only the model client
and surrounding trainer configuration:

- Taskset setup, tools, user simulation, finalization, and scoring remain in the taskset.
- The harness still sends all model calls through the interception endpoint.
- Container-requiring tasks set `NEEDS_CONTAINER` or a per-task `image`.
- Select Docker, Prime, or Modal under `harness.runtime`; do not make the taskset call a sandbox
  provider directly.
- Set explicit stage timeouts and resource requests before scaling concurrency.

Test the trainer's server path locally before a long run:

```bash
uv run eval my-agent-task-v1 -n 10 -r 4 --server --no-rich \
  --pool.type elastic --pool.max-workers 4
```

## Renderers and branching

Use `[orchestrator.renderer] name = "auto"` unless a model or fine-tune needs an explicit renderer.
A renderer is not just a tokenizer: it preserves exact prior token prefixes across multi-turn
requests. This prevents chat-template normalization, BPE drift, or stripped reasoning blocks from
silently changing the sampled sequence.

If the harness rewrites message history—context compaction or a subagent—the trace branches at the
message level. Each branch remains a separate training sample. Renderer-level token divergence is
also represented as a branch rather than corrupting the prefix invariant.

## Failure diagnosis

Separate three failure classes before tuning training:

1. **Task failure** — rollout completed and valid scoring returned a low reward.
2. **Rollout error** — `trace.error` identifies a provider, harness, toolset, user, sandbox,
   taskset, or interception boundary.
3. **Training instability** — valid traces reach the trainer, but loss, KL, gradients, or weights
   become unstable.

Changing learning rate cannot fix an environment error. Changing retries cannot fix a deterministic
reward bug. Inspect saved traces and environment-worker logs first.

## Legacy v0 environments

`prime-rl` still accepts a flat `id` and optional `args` for an existing v0 package. It runs that
package through the v1 trace bridge:

```toml
[[orchestrator.train.env]]
id = "legacy-reverse-text"
args = { split = "train" }
```

New training environments should use the native `taskset` and `harness` shape.

## Hosted Training

The current native v1 configuration surface is the open-source `prime-rl` orchestrator. The
Hosted Training CLI's public environment schema still accepts flat v0 `id`/`args` entries and does
not expose typed v1 `taskset`/`harness` fields. Do not present a native v1 Hosted Training command
as supported until that schema is updated.

## Prompt optimization

The existing `prime gepa run` / `vf-gepa` path is built around v0 rollout records and
`load_environment()` packages. It is not a native v1 taskset/harness optimizer. For v1, treat prompt
changes as typed taskset or harness config, run reproducible evals, and compare the resulting
traces. Do not route a v1 taskset through GEPA unless an adapter explicitly supports `vf.Trace`.
