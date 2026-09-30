---
name: evaluate-environments
description: Configure and run evaluations with verifiers. Use to select tasksets, models, harnesses, and runtimes, resume a run, or inspect its results.
---

# Evaluate Environments

Run the evaluation the user requested. Preserve the dataset, model, harness,
sampling settings, and run size they specified. Ask only about missing choices
that would affect the result.

## Check the setup

1. Check the installed package version and exact dataset, split, and filters.
2. Check config and imports without calling a model:

   ```bash
   uv run vf-eval <MY_ENV> --dry-run
   ```

3. Check a known solution and the untouched task on the intended runtime:

   ```bash
   uv run vf-validate <MY_ENV> --runtime.type docker -n 1
   ```

4. Run a small sample with the requested settings:

   ```bash
   uv run vf-eval <MY_ENV> -m <MODEL> -n 3 -r 1 -c 1
   ```

5. Inspect successful, zero-reward, and failed traces. Once setup and scoring work, continue to the requested run size.

A dry run does not check dataset loading, images, or credentials. Gold validation
reported as `unchecked` has no model-free verdict. Container tasks cannot use
`subprocess` for validation. Use [debug-environments](../debug-environments/SKILL.md)
for setup failures or replay.

Validation runs gold and noop checks in separate runtimes. Gold checks the
reference solution; noop scores the untouched task and flags rewards of at least
1.0. Use `--only-gold`, `--only-noop`, or `--only-setup` for one check. Noop
scoring includes configured judges and can call models.

## Choose the taskset and environment

An ID names an installed Python package. It does not install anything.
`owner/name@version` imports the installed `name` package without enforcing the
version, so check the installed distribution.

The leading ID is shorthand for `--env.taskset.id`:

```bash
uv run vf-eval my-task-v1 --env.agent.harness.id codex --env.agent.runtime.type prime
```

Without `--env.id`, verifiers uses the package's exported `Env`, or
`SingleAgentEnv` if none is exported. Choose another environment explicitly:

```bash
uv run vf-eval my-task-v1 --env.id best-of-n --env.n 8
uv run vf-eval my-task-v1 --env.id agentic-judge --env.judge.runtime.type docker
```

Agent settings live under `env.<agent>.*`. The default single-agent environment
uses `env.agent.*`; there is no top-level `--harness.*` setting.

## Configure the run

Include the intended taskset, environment, and harness when reading CLI help.
This loads their specific options:

```bash
uv run vf-eval my-task-v1 --env.id best-of-n --help
```

Check the selected environment's role names before using `env.<role>.*`.
For defaults, read `verifiers/v1/configs/cli/eval.py` and follow its config types.
Runtime and harness settings live beside their implementations.

Use TOML for a repeatable command:

```toml
model = "openai/gpt-5-mini"

[env.taskset]
id = "my-task-v1"
split = "test"

[env.agent.harness]
id = "bash"

[env.agent.runtime]
type = "subprocess"

[sampling]
temperature = 0.7
```

```bash
uv run vf-eval @ configs/my-eval.toml --dry-run
uv run vf-eval @ configs/my-eval.toml
```

CLI flags override TOML. Use dotted names such as `--env.taskset.split test`,
`--env.agent.runtime.cpu 4`, or `--sampling.temperature 0.7`.

Use `--select.*` for task selection: include or exclude indices, IDs, keys, or
names, then shuffle, skip, and limit. `-n` sets `select.limit`; `-s` enables
`select.shuffle`. See [selecting tasks](../../docs/v1/tasksets.md#selecting-tasks).

Check model authors' recommended sampling settings when the user has not supplied
them. For open models, start with the model card and generation config. Leave
optional settings unset unless needed. Do not impose token or turn limits that
cut short the requested evaluation.

The default runtime is Prime; default concurrency is 128. Set `-c` deliberately
for a small check or limited sandbox capacity. Agent token and turn limits cover
the whole run; `sampling.max_tokens` limits each response. See
[runtimes](../../docs/v1/runtimes.md).

## Endpoints, credentials, and tools

The default client uses Prime Inference with `PRIME_API_KEY` or the active Prime
CLI credentials. For another provider, set `client.base_url` and
`client.api_key_var`. The latter is an environment variable name, not its secret
value. Check the provider supports the API used by the harness.

Task judges have their own model, client, and sampling settings. Configure
`env.taskset.task.judge.*` or `env.taskset.task.judges`, as the task exposes them.
Changing the evaluated model does not change the judge.

Uploads are enabled by default. Use `--no-push` to keep results local. Keep secrets
out of TOML and task data. Harness `forward_env` passes named environment variables
to the agent; keep judge-only credentials on the evaluator.

Most harnesses have `disabled_tools`. Check the implementation and its official
docs for exact names and support:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

Disabling one tool does not restrict other ways to access files or the network.
Use runtime restrictions where the task requires them.

## Retries

Whole-agent retries are off by default (`max_retries = 0`). Each retry starts a
fresh rollout. Use `env.agent.retries` for one agent or `env.retries` for a whole
episode, including environments that own shared resources such as Harbor Compose.
Provider SDK retries are separate.

Ordered rules override the default budget for matching errors. With a zero default, only explicitly enabled errors retry. For example:

```toml
[env.agent.retries]
max_retries = 0

[[env.agent.retries.rules]]
type = "ProviderError"
status_code = [429, "5xx"]
max_retries = 3

[[env.agent.retries.rules]]
type = "SandboxError"
message = 'temporarily unavailable|connection reset'
max_retries = 2
```

Fields within a rule must all match. `type` matches the exact recorded exception name, `status_code` matches any listed status or status class, and `message` is a regex search (plain text matches a substring). Invalid regexes fail config validation. Omitted match fields match anything. Each rule must explicitly provide `max_retries`; an omitted budget fails validation.

The first matching rule wins for each error; zero retries excludes it, and exhausted rules never fall through. Unmatched errors share the default `max_retries` budget. Each retry consumes only the matching rule's budget, or the default budget when no rule matches. Budgets persist across attempts of that agent rollout or episode; the default is not a global cap. An empty rules list uses only the default budget. When an attempt captures multiple errors, the first eligible error triggers the retry; a denied error does not veto other errors. Successful traces' recovered errors do not trigger episode retries.

## Output and resume

Results go to `output_dir / run.dir`. `-o` sets `output_dir`, which defaults to
`outputs`. A run contains:

```text
outputs/<env>--<model>--<harness>--<short-id>/
├── configs/resolved/eval.json    # full config
├── configs/eval.toml             # input TOML, if supplied
├── logs/attempt_1/eval.log
├── logs/latest                  # link to the current attempt
└── traces.jsonl                 # one episode per line
```

Resume with the saved config:

```bash
uv run vf-eval @ <run-dir>/configs/resolved/eval.json --resume
```

Resume keeps complete, successful episodes and reruns missing, failed, or
malformed ones. It rejects config changes. Keep package, code, and dataset
revisions fixed too; the config does not pin their contents. `--clean` deletes
the selected run directory. Use a new directory to preserve existing results.

## Inspect and report results

Read each line of `traces.jsonl` as a `WireEpisode`, then inspect its traces.
An episode can contain several traces or fail before producing any. See
[trace inspection](../../docs/v1/debugging.md) for a reader.

For representative results, check:

- Task data, prompt, assistant replies, and tool results.
- Stop condition, errors, stage timing, and per-model-call records in `calls`.
- Named rewards, weights, metrics, and saved evidence in `info`.
- Judge requests and verdicts in `info["judge_calls"]`.
- Token, mask, and log-probability fields when using the training client.

`trace.reward` sums `score * weight`. `None` means unscored, not zero.
`trace.state` is not saved. Replay can rerun scoring that needs only saved data
and can call judges, but cannot reproduce sandbox or cross-agent scoring.

Report correct answers, wrong answers, runs cut short by limits, and execution
errors separately. Include the failure rate when reporting aggregate scores.
Inspect examples before attributing a score change to model quality.

Use solve rate or pass@k for binary rewards. For continuous rewards, inspect the
distribution and compare tasks or groups. For scores that compare several
attempts, state the comparison rule and group size. Keep task selection, harness,
runtime, and sampling fixed across comparisons. A tiny check shows whether the
run works; it does not establish model quality.
