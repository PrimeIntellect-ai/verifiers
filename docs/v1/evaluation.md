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
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks; an
  infinite taskset (a procedural generator, e.g. `wordle-v1`) requires it
- `num_rollouts` — rollouts per task
- `max_concurrent` — caps how many rollouts are in flight at once
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed); a no-op on an infinite taskset

## Optimizing a system prompt

For a native v1 taskset, seed a prompt file from the taskset's own baseline system prompt.
Seeding scans the same task selection the eval scores (`-n`/`--shuffle`; an infinite taskset
requires `-n`) and is refused when those tasks carry differing system prompts — the override
below replaces all of them:

```bash
uv run weco-eval <taskset-id> --system-prompt-path prompt.txt --init-prompt -n 20
```

Then use `weco-eval` as the parseable scalar evaluator for Weco:

```bash
weco run --source prompt.txt \
  --eval-command "uv run weco-eval <taskset-id> --system-prompt-path prompt.txt -n 20" \
  --metric reward --goal maximize --apply-change \
  --additional-instructions "$(cat prompt.txt)"
```

`--additional-instructions` carries the seeded baseline prompt, so the optimizer keeps the
task's intent even after it has rewritten `prompt.txt`.

`--system-prompt-path` overrides native v1 task prompts. Legacy v0 evals reject it. Errored
rollouts score 0 in the reported mean, so flaky candidates can't win on their surviving
rollouts. `--apply-change` writes the winning prompt back to `prompt.txt` without the
interactive confirmation `weco run` otherwise ends with — required when running headless.

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `results.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone. When `--system-prompt-path` is used, the initial run copies the prompt to `system_prompt.txt` in the output directory and saves that snapshot in `config.toml`; changing the original file cannot mix prompt versions during resume.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.
