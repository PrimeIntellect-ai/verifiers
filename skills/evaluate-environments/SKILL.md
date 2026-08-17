---
name: evaluate-environments
description: Run and evaluate verifiers tasksets. Set up the necessary config files and observe the runs and their results.
---

# Evaluate Tasksets

## Goal

Set up an evaluation for a taskset in the correct way to reproduce results from others or evaluate a model and harness combination on a given taskset.

## Canonical path

Use the prime CLI

```bash
prime eval run <MY_ENV>
```

## Core workflow

1. Resolve and validate config without model calls:

```bash
prime eval run <MY_ENV> --dry-run
```

2. Run model-free gold validation when the taskset implements `validate`:

```bash
prime eval validate <MY_ENV> --runtime.type subprocess
```

3. Do a small run to see whether it works correctly:

```bash
prime eval run <MY_ENV> -m deepseek/deepseek-v4-flash -n 3 -r 1
```

4. Inspect successful, zero-reward, and errored traces.
5. Scale only after task loading, harness capability, runtime lifecycle, and scoring are correct.

When the user requests a full run, do not restrict the number of tasks. Ask for the appropriate harness to use (if not specified)

## IDs and plugin resolution

A plugin id names an installed package (e.g. `my-taskset`); verifiers imports it and never installs anything itself.

The leading ID is shorthand for `--env.taskset.id`. A harness belongs to an agent — `--env.agent.harness.*` on the single-agent env, `--env.<agent>.harness.*` on a multi-agent one (there is no run-level `--harness.*`):

```bash
prime eval run my-task-v1 --env.agent.harness.id codex --env.agent.runtime.type prime
```

The env — the control flow between agents — owns the whole `[env]` block. Empty `--env.id`
keeps the taskset's own story (its exported `Env` subclass, else the single-agent
env); `--env.id` pairs a reusable env with any taskset, its knobs typed under `--env.*`:

```bash
prime eval run my-task-v1 --env.id best-of-n --env.n 8      # pass@k / rejection sampling
prime eval run my-task-v1 --env.id agentic-judge \
  --env.judge.runtime.type docker                           # a judge agent verifies each attempt in a sandbox
```

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Research the relevant first party documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.

## Config discovery

The CLI help is generated from the current config classes. Include the taskset and env ids you plan to use before `--help` so their concrete config fields are loaded:

```bash
uv run eval my-task-v1 \
  --env.id best-of-n \
  --help
```

For implementation details and defaults, start at `verifiers/v1/configs/cli/eval.py` and follow its fields into `verifiers/v1/configs/`. Client configs live in `verifiers/v1/configs/client.py`, sampling in `verifiers/v1/types.py`, and runtime- and harness-specific configs next to their implementations in `verifiers/v1/runtimes/` and `verifiers/v1/harnesses/`. Custom taskset and env config fields live next to those implementations.

## Typed taskset overrides

Taskset settings:

```bash
prime eval run my-task-v1 --env.taskset.split test --env.taskset.difficulty hard
```

Harness and runtime settings:

```bash
prime eval run my-task-v1 \
  --env.agent.harness.id rlm \
  --env.agent.runtime.type docker \
  --env.agent.runtime.cpu 4 \
  --env.agent.runtime.memory 8
```

Sampling:

```bash
prime eval run my-task-v1 \
  --sampling.temperature 0.7 \
  --sampling.top-p 0.95 \
  --sampling.max-tokens 2048 \
  --sampling.reasoning-effort medium
```

Always research the correct sampling parameters first. This is one of the most important settings, so make sure to find the correct values. For open models, you can find them on Hugging Face in the README and/or in the generation config.

Your parameter selection or settings should leave room for full runs, and you should not restrict things like tokens or number of turns unless specified by the user.

Leave optional settings unset unless the user asks for them. Always confirm the harness, runtime, and sampling parameters before running an evaluation.

## Reproducible TOML

You can also use a TOML:

```toml
model = "openai/gpt-5-mini"

[env.taskset]
id = "my-task-v1"
split = "test"

[env.agent]
runtime = { type = "subprocess" }

[env.agent.harness]
id = "bash"

[sampling]
temperature = 0.7
```

```bash
prime eval run @ configs/my-eval.toml
```

## Retries

Whole-rollout retry is opt-in. That means if something fails in the rollout, the whole rollout is retried. This is very useful for large-scale runs. You can also restrict certain errors from the retries:

```bash
prime eval run my-task-v1 \
  --env.agent.retries.max-retries 2 \
  --env.agent.retries.include SandboxError ProviderError \
  --env.agent.retries.exclude TaskError
```

## Output and resume

Default output:

```text
outputs/<env>--<model>--<harness>/<uuid>/
├── config.toml
├── traces.jsonl
└── eval.log
```

Set an exact path with `-o`. `traces.jsonl` is one **episode** per line — the episode's traces plus their shared standing — appended after each episode finishes, so an episode is durable whole or not at all (a torn last line is the whole episode redone on resume).

Resume in place:

```bash
prime eval run --resume /path/to/run
```

## Trace inspection

For each representative sample inspect:

- `task` and prompt fields;
- `branches`, assistant messages, tool messages, and stop condition;
- named `rewards`, aggregate `reward`, and `metrics`;
- persisted `info` artifacts;
- `error`/`errors` and boundary type;
- per-call `calls` records (model, sampling, finish reason, usage, timing, error) linked to the graph;
- usage and stage timing;
- token/mask/logprob fields when using the training client.

Classify outcomes:

1. Valid completion and correct reward.
2. Valid completion with low reward (model/task outcome).
3. Truncated completion (budget outcome).
4. Captured rollout error (provider, harness, tool, user, runtime, task, or interception).

Do not average these categories together without reporting failure rate.

## Metrics interpretation

- Binary rewards support solve rate and pass@k-style analysis.
- Continuous rewards need distributions, quantiles, and per-task/group comparisons.
- Group rewards must be interpreted with their comparison rule and group size.
- Always inspect samples before attributing a delta to model quality.
- Keep taskset, harness, runtime, sampling, and selected task indices fixed across variants.
- Do not overinterpret a tiny smoke run.
