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

- `my-taskset` resolves an importable local package.
- `owner/name` installs a Hub package on demand.
- `owner/name@version` pins a Hub version.

The leading ID is shorthand for `--env.taskset.id`. A harness belongs to a seat —
`--env.agent.harness.*` on the single-agent env, `--env.<role>.harness.*` on a
multi-agent role (there is no run-level `--harness.*`):

```bash
prime eval run owner/name --env.agent.harness.id codex --env.agent.harness.runtime.type prime
```

The env — the control flow between agents — owns the whole `[env]` block. Empty `--env.id`
keeps the taskset's own story (its exported `Environment` subclass, else the single-agent
env); `--env.id` pairs a reusable env with any taskset, its knobs typed under `--env.*`:

```bash
prime eval run my-task-v1 --env.id best-of-n --env.n 8      # pass@k / rejection sampling
prime eval run my-task-v1 --env.id agentic-judge \
  --env.judge.harness.runtime.type docker                   # a judge agent verifies each attempt in a sandbox
```

When specifying Hub tasksets, always include the owner to resolve them correctly.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Research the relevant first party documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.

## Typed taskset overrides

Taskset settings:

```bash
prime eval run my-task-v1 --env.taskset.split test --env.taskset.difficulty hard
```

Harness and runtime settings:

```bash
prime eval run my-task-v1 \
  --env.agent.harness.id rlm \
  --env.agent.harness.runtime.type docker \
  --env.agent.harness.runtime.cpu 4 \
  --env.agent.harness.runtime.memory 8
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

For all parameters, look up the [reference](references/REFERENCE.md). Leaving things out when the user does not want them is a sane default. However, you should always ask for the harness to use, the runtime to use, as well as the sampling parameters.

## Reproducible TOML

You can also use a TOML: 

```toml
model = "openai/gpt-5-mini"

[env.taskset]
id = "my-task-v1"
split = "test"

[env.agent.harness]
id = "default"
runtime = { type = "subprocess" }

[sampling]
temperature = 0.7
```

```bash
prime eval run @ configs/my-eval.toml
```

For all parameters, look up the [reference](references/REFERENCE.md). Leaving things out when the user does not want them is a sane default. However, you should always ask for the harness to use, the runtime to use, as well as the sampling parameters.

## Retries

Whole-rollout retry is opt-in. That means if something fails in the rollout, the whole rollout is retried. This is very useful for large-scale runs. You can also restrict certain errors from the retries:

```bash
prime eval run my-task-v1 \
  --env.retries.rollout.max-retries 2 \
  --env.retries.rollout.include SandboxError ProviderError \
  --env.retries.rollout.exclude TaskError
```

## Output and resume

Default output:

```text
outputs/<taskset>--<model>--<harness>/<uuid>/
├── config.toml
├── traces.jsonl
└── eval.log
```

Set an exact path with `-o`. Results append as each trace finishes.

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

## Legacy bridge

For a real v0 package only:

```bash
prime eval run --id legacy-env --args.split test -n 5
```

Label the result as bridged v0. Do not present `args` as the v1 config contract.
