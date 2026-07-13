# Evaluation

To evaluate any environment, use the `eval` entrypoint:

```bash
uv run eval primeintellect/terminal-bench-2
```

You can also use `.toml` files for configuration:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[taskset]
id = "primeintellect/terminal-bench-2"

[harness]
id = "codex"
version = "0.116.0"
```

Validate the config by using `uv run eval @ config.toml --dry-run`. To run the evaluation, use `uv run eval @ config.toml`.

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

Evaluation output is written under `outputs/<taskset>--<model>--<harness>/<uuid>/` by
default. The directory contains `config.toml`, one completed trace per line in
`traces.jsonl`, and logs in `eval.log`. Explicit `--topology.id` runs share the
in-process and `--server` runners (no `--resume` / platform push yet); they write flat
traces dug out of each finished graph.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `taskset.id` / `harness.id` — pick the taskset and harness
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks
- `num_rollouts` — independent invocations per seed task
- `max_concurrent` — caps agent runs in-process or graph requests through the server
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed)

## Resuming evaluations

`--resume <output-dir>` independently re-runs graph invocations a previous run left
missing or errored, appending to that run's `traces.jsonl`. It reloads the saved
`config.toml` verbatim, so it takes no other arguments.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.
