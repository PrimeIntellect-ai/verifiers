---
name: evaluate-environments
description: Run and evaluate verifiers environments. Set up the necessary config files and observe the runs and their results.
---

# Evaluate Environments

## Goal

Set up an evaluation for an environment in the correct way to reproduce results from others or evaluate a model and harness combination on a given environment.

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

2. Run model-free gold validation when the taskset implements it:

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

- `my-environment` resolves an importable local package.
- `owner/name` installs a Hub package on demand.
- `owner/name@version` pins a Hub version.

The leading ID is shorthand for `--taskset.id`. Select a harness independently:

```bash
prime eval run owner/name --harness.id codex --harness.runtime.type prime
```

When specifying environments, always include the owner to resolve it correctly.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Research the relevant first party documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.

## Typed environment overrides

Taskset settings:

```bash
prime eval run my-task-v1 --taskset.split test --taskset.difficulty hard
```

Harness and runtime settings:

```bash
prime eval run my-task-v1 \
  --harness.id rlm \
  --harness.runtime.type docker \
  --harness.runtime.cpu 4 \
  --harness.runtime.memory 8
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

[taskset]
id = "my-task-v1"
split = "test"

[harness]
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
  --retries.rollout.max-retries 2 \
  --retries.rollout.include SandboxError ProviderError \
  --retries.rollout.exclude TasksetError
```

## Output and resume

Default output:

```text
outputs/<taskset>--<model>--<harness>/<uuid>/
├── config.toml
├── results.jsonl
└── eval.log
```

Set an exact path with `-o`. Results append as each trace finishes.

Resume in place:

```bash
prime eval run --resume /path/to/run
```

## Export SFT data

A finished run's saved traces can be reshaped into an SFT dataset for prime-rl's `uv run sft`:

```bash
uv run export-sft /path/to/run --min-reward 1.0
```

Emits a `messages` column (OpenAI chat wire shape) plus a `tool_defs` column (the tools advertised to the model), one row per branch. Generation-errored traces always drop; `--min-reward` / `--drop-truncated` select further. Writes `<run>/sft/train.parquet` (readable via `load_dataset`, i.e. prime-rl's `data.name`) or pushes to the Hugging Face Hub with `--push <repo-id>`.

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
4. Captured rollout error (provider, harness, tool, user, runtime, taskset, or interception).

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
