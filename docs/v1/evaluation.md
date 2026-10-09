# Evaluation

To evaluate any taskset, use the `eval` entrypoint:

```bash
uv run vf-eval primeintellect/terminal-bench-2
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

Validate the config by using `uv run vf-eval @ config.toml --dry-run`. To run the evaluation, use `uv run vf-eval @ config.toml`.

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

The output from evaluations are written into `outputs/<env>--<model>--<harness>/<uuid>/` by default, where `<env>` is the taskset, prefixed by the paired env id when `--env.id` sets one (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the episodes in `traces.jsonl`, as well as logs of the run and workers in `logs/attempt_<n>/eval.log` — one directory per launch attempt (a resume starts a new one), with `logs/latest` pointing at the current attempt.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `env.taskset.id` — pick the taskset (or the positional `eval <taskset-id>`)
- `env.agent.harness.id` — pick the agent's harness (`[env.agent.harness]` in TOML)
- `select` — which tasks to evaluate (see [Selecting tasks](tasksets.md#selecting-tasks)).
  `-n` sets `select.limit`: not setting it means all tasks, and an infinite taskset (a
  procedural generator, e.g. `wordle`) requires it. `-s` sets `select.shuffle` (fixed seed)
- `num_rollouts` — rollouts per task
- `verbose` — log at debug instead of info
- `rich` — the live dashboard (default); `--no-rich` streams logs to the console and
  prints each trace as JSON at the end
- `rich.show_logs` — replace the dashboard's per-rollout rows with a live tail of the
  attempt's logs (`logs/latest/eval.log`), the env workers' lines included
- `push` — upload the run to the Prime Intellect platform as it goes (default; `--no-push`
  keeps it local). `run.attach <evaluation-id>` streams into a run the platform already
  created instead of opening a new one — this is how a hosted evaluation's sandbox runs
  the same command; it is not something a local run sets by hand

## Network access

The `prime`, `docker` and `modal` runtimes can enforce an egress policy on the agent's box. A policy is one `NetworkPolicyConfig` with two lists of destinations:

- `allow`: what the box may reach. Each entry is a host pattern (`github.com`, `*.github.com`, with `fnmatch` wildcards) or a URL origin (`https://api.github.com:443`, whose scheme and port must then match). `["*"]` is unrestricted and `[]` reaches nothing beyond the framework's own routes (the model endpoint and tool servers).
- `block`: what the box may not reach. A bare domain blocks its subdomains too; `*.example.com` blocks the subdomains but not the domain itself; `*` blocks everything beyond the framework, which is the same as `allow = []`.

A request is permitted when it matches no `block` entry and some `allow` entry. A concrete allowlist cannot be combined with a non-empty blocklist (the config is rejected); use `allow = ["*"]` with `block` for a denylist, or `allow` alone for an allowlist.

A box's policy comes from two sources:

- The task. Its data declares it (`TaskData.network`; a closed-book benchmark declares `allow = []`), and the run's task config replaces that declaration: `[env.taskset.task.network]` in TOML or `--env.taskset.task.network.allow` on the CLI.
- The runtime (`[env.agent.runtime]`). Its `allow`/`block` are additive restrictions on top of the task's resolved policy: the two allowlists intersect (only destinations both permit survive, and `[]` on either side leaves nothing) and the two blocklists union. The runtime can therefore only narrow a task's policy, never widen it; widening is what the task config above is for.

Before a rollout starts, verifiers appends a note about the restriction to the system prompt, so the model treats a failing connection as intended. `network_notice = false` under `[env.agent]` disables it.

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `traces.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.

## Skills

Harnesses whose program supports SKILL.md skills natively (e.g. Claude Code, Codex) take a `skills` list. A local skill folder is uploaded to `<skills dir>/<folder name>`. A `{runtime = "..."}` entry copies the contents of a directory already inside the runtime into the program's skill discovery directory:

```toml
[env.agent.harness]
skills = [{runtime = "/opt/skills"}, "path/to/my-skill"]
```

Tasks can supply the same sources through `TaskData.skills`, for example `skills=[{"runtime": "/opt/skills"}]`. Task sources are installed first, followed by harness sources; later files override matching earlier files. Each run gets its own installed skills. Sources are resolved after task setup, and missing source directories fail the run.

Setting `skills` on a task or harness without native harness skill support fails up front.
