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

The output from evaluations are written into `outputs/<taskset>--<model>--<harness>/<uuid>/` by default (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the traces in `results.jsonl`, as well as logs of the run and workers in `eval.log`.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `taskset.id` / `harness.id` — pick the taskset and harness
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks
- `num_rollouts` — rollouts per task
- `max_concurrent` — caps how many rollouts are in flight at once
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed)

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `results.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone.

## Exporting SFT data

`uv run export-sft <run-dir>` reshapes a finished run's saved traces into an SFT dataset that [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)'s `uv run sft` consumes directly: a `messages` column (OpenAI chat wire shape) plus a `tool_defs` column (the tools advertised to the model, from `trace.tool_defs`). One row per branch; generation-errored traces are always dropped (a scoring-only error keeps its finished transcript). `--min-reward 1.0` keeps only solved rollouts, `--drop-truncated` drops budget-cut ones. Writes `<run-dir>/sft/train.parquet` (readable via `load_dataset`, i.e. prime-rl's `data.name`), or pushes to the Hugging Face Hub with `--push <repo-id>`.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.
