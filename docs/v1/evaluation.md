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
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks; an
  infinite taskset (a procedural generator, e.g. `wordle-v1`) requires it
- `num_rollouts` — rollouts per task
- `max_concurrent` — caps how many rollouts are in flight at once
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed); a no-op on an infinite taskset

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `results.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.


## Rendering images as text

Textify can replace every base64 image sent to the model — task prompts, tool results, and
user-simulator turns — with deterministic ASCII or braille at the interception server. It is
disabled by default; ASCII is the default mode when enabled:

```toml
[textify]
enabled = true
width = 160
```

Or on the CLI: `--textify.enabled true --textify.width 160`. Set
`--textify.mode braille` for braille; `height`, `char_aspect`, `gamma`, `invert`, `ramp`,
`threshold`, and `max_chars` tune rendering. Set `threshold = "otsu"` for deterministic
global Otsu binarization in either ASCII or braille mode. `invert` defaults to auto, which maps the white
backgrounds common in diagrams and documents to spaces. Each image is limited to 40,000 output
characters by default (25 megapixels decoded), so extreme aspect ratios cannot explode a prompt.
