# Evaluation

To evaluate any taskset, use the `eval` entrypoint:

```bash
uv run eval primeintellect/terminal-bench-2
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

Validate the config by using `uv run eval @ config.toml --dry-run`. To run the evaluation, use `uv run eval @ config.toml`.

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

The E2B runtime requires the E2B Python SDK and an API key: install it alongside verifiers with `uv add "e2b>=2.35.0"`, set `E2B_API_KEY`, and select it with `type = "e2b"` under `[env.agent.runtime]`. The first run with a given image, workdir, CPU, and memory combination builds a cached E2B template (images must be Debian-based; CPU must be 1 or an even whole number, memory an even whole number of MB); later sandboxes start from the cache. `timeout` bounds the sandbox lifetime (default one hour) — raise it above your longest rollout. Sandbox creation is paced by `creates_per_sec`, whose default of 1/s fits E2B's base tier — Pro allows ~5/s and enterprise plans more, so raise it to your plan's rate for high-concurrency runs. Like the other remote runtimes, E2B sandboxes reach the host-side interception server through a tunnel, so the eval or env-server host also needs Prime credentials (`prime login`).

The output from evaluations are written into `outputs/<env>--<model>--<harness>/<uuid>/` by default, where `<env>` is the taskset, prefixed by the paired env id when `--env.id` sets one (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the episodes in `traces.jsonl`, as well as logs of the run and workers in `eval.log`.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `env.taskset.id` — pick the taskset (or the positional `eval <taskset-id>`)
- `env.agent.harness.id` — pick the agent's harness (`[env.agent.harness]` in TOML)
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks; an
  infinite taskset (a procedural generator, e.g. `wordle`) requires it
- `num_rollouts` — rollouts per task
- `verbose` — log at debug instead of info
- `shuffle` — samples the task order (fixed seed); an error on an infinite taskset

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

Harnesses whose program supports SKILL.md skills natively (e.g. Claude Code, Codex) take a `skills` list of local skill folders, each uploaded into the program's skill discovery directory in the agent's runtime as `<skills dir>/<folder name>`:

```toml
[env.agent.harness]
skills = ["path/to/my-skill"]
```

Setting `skills` on a harness without native skill support fails up front.
