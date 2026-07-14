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

## External candidate optimization (Weco)

Weco rewrites a candidate artifact and re-runs an eval command; `weco-eval` runs one fixed
v1 evaluation of the taskset + harness and ends stdout with a parseable `reward: <mean>`
line. Any errored rollout fails the step (non-zero exit, no metric lines) rather than
scoring a partial eval; `--retries.rollout.*` absorbs transient errors. The candidate must
be a *declarative* file the taskset or harness actually loads on each evaluation — a
prompt, template, or config, never Python the taskset imports; verifiers neither receives
nor manages Weco's source paths. Author the optimizer guidance (what may change, what must
remain) yourself in an instructions *file* — inline text breaks on leading `-` or
filename-shaped content, a missing path silently becomes literal instruction text (check
it exists), and instructions must never come from candidate or task output:

```bash
weco run --source <candidate-artifact> \
  --eval-command "uv run weco-eval <MY_ENV> -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

When the candidate is the taskset's system prompt, use the built-in override and seeding
(each run snapshots the evaluated prompt to `<run-dir>/system_prompt.txt` and points its
saved config at the snapshot; an explicit `-o` gains a per-run leaf so candidates never
overwrite each other):

```bash
uv run weco-eval <MY_ENV> --system-prompt-path prompt.txt --init-prompt -n 20
weco run --source prompt.txt \
  --eval-command "uv run weco-eval <MY_ENV> --system-prompt-path prompt.txt -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

Headless runs are bounded and noninteractive with an explicit `--steps` budget (Weco defaults to 100), an
`--eval-timeout`, and `--apply-change` (without it `weco run` ends on an interactive
confirmation). Keep the eval command, scoring implementations, reference answers,
taskset/harness/sampling config, and task selection fixed across candidates and outside
`--source`; evaluate the winner on a held-out selection. Declarative candidates only: code
candidates run in the host `weco-eval` process (harness sandboxing does not cover taskset
import) and could tamper with scorers or data in memory, which even a container around the
evaluator can't make valid. Expect data egress — `weco run` uploads source contents,
instructions, the eval command, each step's eval output, and passed `--api-key` keys to
the Weco service. Native v1 tasksets only; use the existing legacy workflow for v0
environments.

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
  --retries.rollout.exclude TaskError
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
