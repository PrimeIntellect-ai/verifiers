---
name: train-with-environments
description: Configure and diagnose prime-rl training with native verifiers.v1 tasksets, harnesses, runtimes, and trace branches. Use for RL experiment setup, environment-server configuration, group sizing, difficulty filtering, renderer selection, periodic eval, or rollout/training failure diagnosis.
---

# Train With Environments

## Goal

Move a validated v1 environment into `prime-rl` without changing its task, harness, runtime, or
scoring semantics, and diagnose environment failures separately from training instability.

## Supported native path

The current native v1 training integration is open-source `prime-rl`. It consumes `vf.Trace` over
the v1 environment-server protocol.

Hosted Training's public CLI schema still exposes flat v0 `id`/`args` environments, not typed v1
`taskset`/`harness` fields. Do not present a native v1 hosted command as supported until that schema
changes.

## First-run gate

Before configuring RL:

```bash
uv run validate my-task-v1 -n 20 --runtime.type subprocess
uv run eval my-task-v1 -m openai/gpt-5-mini -n 20 -r 4 --shuffle
uv run eval my-task-v1 -m openai/gpt-5-mini -n 20 -r 4 --shuffle \
  --server --no-rich --pool.type elastic --pool.max-workers 2
```

Require:

- correct prompt/tool/user traces;
- intended reward on known successes and failures;
- mixed reward on the target base model;
- bounded turn/token/time/resource use;
- low infrastructure error and truncation rates;
- successful environment-server path.

Do not start training to discover whether scoring works.

## Native `prime-rl` environment shape

```toml
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
name = "my-task-train"
taskset = { id = "my-task-v1", split = "train" }
harness = { id = "default", runtime = { type = "subprocess" } }
timeout = { rollout = 600, scoring = 120 }
max_turns = 8
max_total_tokens = 32768

[orchestrator.eval]
interval = 20

[[orchestrator.eval.env]]
name = "my-task-test"
taskset = { id = "my-task-v1", split = "test" }
harness = { id = "default", runtime = { type = "subprocess" } }
num_examples = 200
group_size = 4
```

Native env entries inherit the v1 server config. Use:

- `taskset` for taskset ID and typed fields;
- `harness` for harness ID and runtime;
- root environment fields for framework turn/token limits, stage timeouts, retries, interception
  multiplexing, and worker pool;
- `sampling` for an environment-specific override;
- `ratio` for multi-environment sampling weight;
- `address` only for an externally managed env server.

Never put native v1 fields under `args`. That is the legacy bridge.

## Dry-run and launch

From the current `prime-rl` checkout:

```bash
uv run rl @ /path/to/rl.toml --dry-run
uv run rl @ /path/to/rl.toml
```

Use the repository's current install and model-support docs. `prime-rl` requires NVIDIA GPUs. Use
its `inference` entrypoint rather than plain `vllm serve`; training needs token generation and
weight-update interfaces supplied by that entrypoint.

## Group and batch sizing

`orchestrator.group_size` is rollouts per task for group-relative advantage; a train environment
can override it.

- Start at `8` or more; `16` is a common baseline.
- Larger groups improve the chance of mixed outcomes but increase rollout cost and group latency.
- If groups are mostly all-zero, fix difficulty, model fit, or environment behavior before
  increasing optimizer effort.
- If groups are mostly all-one, increase difficulty or sample a harder task distribution.
- Choose `batch_size` with whole groups, packing, sequence length, and GPU memory in mind.

A taskset `@vf.group_reward` is scoring, not trainer advantage. The env server automatically keeps
that task's rollouts together before returning traces.

## Difficulty filtering

Online difficulty filtering can drop exact all-zero and all-one groups. It fits binary rewards.
For continuous rewards, inspect distributions and use a filter whose thresholds match the signal;
do not assume equality with `0` and `1` captures easy/hard collapse.

Monitor per environment:

- reward distribution and solve-none/solve-all rate;
- effective batch size after filtering;
- errored and truncated rollout rates;
- completion length and turns;
- policy staleness/off-policy age;
- mismatch KL and trainer stability metrics.

## Renderers and branches

Use `[orchestrator.renderer] name = "auto"` unless the model/fine-tune needs an explicit renderer.
The renderer maintains exact token-prefix identity between turns and puts token IDs, masks, and
logprobs on trace nodes.

V1 traces may branch because of context compaction, subagents, or token-level renderer divergence.
Each branch is a training sample. Do not flatten branches into one transcript or reconstruct tokens
from text.

## Agentic runtimes

Keep eval and training composition identical:

```toml
[[orchestrator.train.env]]
taskset = { id = "harbor", dataset = "terminal-bench/terminal-bench-2" }
harness = { id = "rlm", runtime = { type = "prime", cpu = 4, memory = 8 } }
timeout = { setup = 600, rollout = 3600, finalize = 600, scoring = 600 }
pool = { type = "elastic", max_workers = 8, multiplex = 64 }
```

Verify current field names from the selected taskset/harness help or source; do not copy guessed
dataset knobs.

For remote runtimes:

- set explicit timeouts and resource limits;
- use `multiplex` to control interception tunnel count;
- cap pool workers and sandbox creation rate;
- ensure tools/user placement is reachable and correctly isolated;
- make teardown observable before scaling.

## Multi-environment training

Every environment needs a unique `name`. If weighted sampling is used, set `ratio` on all training
environments:

```toml
[[orchestrator.train.env]]
name = "math"
ratio = 3
taskset = { id = "gsm8k-v1", split = "train" }
harness = { id = "default", runtime = { type = "subprocess" } }

[[orchestrator.train.env]]
name = "tools"
ratio = 1
taskset = { id = "wiki-search-v1" }
harness = { id = "default", runtime = { type = "subprocess" } }
```

Ratios are relative. Keep environment-specific rewards interpretable; a scalar reward scale mismatch
can dominate training even when sampling ratios look balanced.

## Failure diagnosis

Classify before changing hyperparameters:

1. **Valid low reward** — model/task difficulty or prompt/tool behavior.
2. **ProviderError** — inference endpoint, model, capacity, or request dialect.
3. **HarnessError** — program installation, launch, or nonzero exit.
4. **ToolsetError/UserError** — server startup, reachability, or call behavior.
5. **SandboxError/TunnelError** — runtime capacity, lifecycle, filesystem, network, or rate limits.
6. **TasksetError** — setup/finalize/scoring implementation.
7. **Trainer instability** — valid branches arrive, then loss/KL/gradient/weight behavior fails.

Retries may help transient provider/sandbox/tunnel failures. They do not fix deterministic taskset
or harness errors. Learning-rate changes do not fix invalid traces.

## Quality rules

- Never train on an uninspected reward.
- Keep held-out periodic eval configured from the start.
- Preserve exact taskset/harness/runtime config between baseline eval and training.
- Use model-free validation for dataset and verifier regressions.
- Bound every remote stage.
- Do not direct harness model calls upstream; they must pass through interception.
- Do not modify reward semantics merely to create more gradient signal without user approval.
- Do not publish a package or start a costly run without the user's requested external action and
  visibility/compute choice.

## Legacy v0

A real v0 environment can still be bridged with a flat entry:

```toml
[[orchestrator.train.env]]
id = "legacy-env"
args = { split = "train" }
```

Label it legacy. New work should use `taskset` and `harness`.

## Deliverable

Report:

1. Prime-RL version/source and resolved config.
2. Taskset, harness, runtime, renderer, group, batch, and eval settings.
3. Pretraining validation/eval evidence.
4. Expected resource/cost envelope and external dependencies.
5. Monitoring and stop criteria.
6. Any environment, rollout, or trainer failures with their correct boundary classification.
