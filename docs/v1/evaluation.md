# Evaluation

To evaluate an installed taskset, use `vf-eval`:

```bash
uv run vf-eval primeintellect/terminal-bench-2
```

The taskset must already be installed. An ID such as `owner/name@version` loads
the installed `name` package; it does not download or enforce that version.
Include the taskset and harness IDs in `--help` to see their specific options:

```bash
uv run vf-eval my-task --env.agent.harness.id codex --help
```

You can also use `.toml` files for configuration:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[env.taskset]
id = "primeintellect/terminal-bench-2"

[env.agent.harness]
id = "codex"
version = "0.116.0"

[env.agent.runtime]
type = "docker"
```

Check the config with `uv run vf-eval @ config.toml --dry-run`, then run it with
`uv run vf-eval @ config.toml`.

CLI flags override TOML values. Use dotted names, such as `--sampling.temperature 0.5`.

Results go to `output_dir / run.dir`, defaulting to
`outputs/<env>--<model>--<harness>--<short-id>/`. If you select an `env.id`, it is
included before the taskset name. The run contains:

```text
configs/eval.toml           # launch TOML, when supplied
configs/resolved/eval.json  # full resolved config, usable with @
traces.jsonl               # one complete episode per line
logs/attempt_1/eval.log     # run and worker logs for this launch
logs/latest                # symlink to the latest attempt
```

## Model endpoints

The default client uses Prime Inference and `PRIME_API_KEY`, falling back to the
active Prime CLI credentials. To use another compatible endpoint:

```toml
[client]
base_url = "http://localhost:8000/v1"
api_key_var = "MODEL_API_KEY"
```

Put the variable's name in config and its secret value in your environment.
The endpoint must support the API used by the harness. Each agent can override
`model`, `client`, and `sampling` under
`env.<agent>`. Task judges have independent client settings; see
[judges](tasksets.md#using-judges).

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — model settings, such as `sampling.temperature`
- `env.taskset.id` — pick the taskset (or the positional `eval <taskset-id>`)
- `env.agent.harness.id` — pick the agent's harness (`[env.agent.harness]` in TOML)
- `select` — [which tasks to evaluate](tasksets.md#selecting-tasks). `-n` sets `select.limit`; `-s` enables `select.shuffle`.
- `num_rollouts` — rollouts per task
- `max_concurrent` / `-c` — episodes running at once (default 128); lower it for a small check or limited sandbox capacity
- `verbose` — log at debug instead of info
- `rich` — the live dashboard (default); `--no-rich` streams logs to the console and
  prints each trace as JSON at the end
- `rich.show_logs` — show live logs, including worker logs, in place of the dashboard's rollout rows
- `push` — upload results to Prime Intellect as the run progresses (default); `--no-push` keeps them local

The platform uses `run.attach <evaluation-id>` to send results to an evaluation
it already created. Local runs do not need to set it.

## Resuming evaluations

Re-run the saved resolved config with `--resume`:

```bash
uv run vf-eval @ <run-dir>/configs/resolved/eval.json --resume
```

Complete, successful episodes are kept. Missing, failed, or malformed episodes
run again in the same directory. The config must match the saved run.

Keep the installed taskset, code, and dataset versions fixed too. Tasks are
matched by a hash of their data; matching config alone does not ensure matching
datasets. `--clean` deletes the selected run directory. Use a new directory to
keep the previous results.

## Timeouts and retries

Retries of the whole agent run are off by default. To enable them:

```toml
[env.agent.retries]
max_retries = 0

[[env.agent.retries.rules]]
type = "ProviderError"
status_code = [429, "5xx"]
max_retries = 2

[[env.agent.retries.rules]]
type = "SandboxError"
max_retries = 2
```

Rules are checked in order; all fields in a rule must match. The first matching
rule sets that error's retry budget. Each rule has its own budget across attempts;
unmatched errors use the default `max_retries` budget. A zero or exhausted budget
does not fall through to later rules, but another error can still trigger a retry.
A retry starts a new attempt from the beginning. Provider SDK retries are separate.
Environments that manage shared resources, such as Harbor Compose, use
`env.retries`. See [runtimes](runtimes.md)
for stage timeouts, token limits, and network policy, and
[trace inspection](debugging.md) for validation, failure diagnosis, and replay.

## Disabling tools

Most harnesses accept a `disabled_tools` list:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

Check the harness's docs for exact tool names and whether it supports disabling them.

## Skills

Harnesses that support `SKILL.md` files, such as Claude Code and Codex, accept a
`skills` list. Give a local skill folder to upload it. Use `{runtime = "..."}`
to copy skills already inside the runtime into the program's skill directory:

```toml
[env.agent.harness]
skills = [{runtime = "/opt/skills"}, "path/to/my-skill"]
```

Tasks can also set `TaskData.skills`, for example
`skills=[{"runtime": "/opt/skills"}]`. Task skills are installed first, then
harness skills. Later files overwrite earlier files with the same path. Each run
gets its own copy. Installation happens after task setup; a missing source
directory fails the run.

The run fails if skills are configured but the harness does not support them.
