# GEPA Prompt Optimization

verifiers offers built in support for GEPA, an algorithm that optimizes a system prompt to maximize the downstream reward for a given taskset:

```bash
uv run gepa reverse-text-v1
```

`gepa` runs [GEPA](https://github.com/gepa-ai/gepa) where a number of rollouts are done then a teacher LLM reflects on the results to propose a better `Task.system_prompt` without any gradient based training. It runs against native v1 tasksets. Legacy v0 environments can be run with `vf-gepa`.

GEPA reuses the same `taskset` / `harness` / `client` / `sampling` config as eval, so the `.toml` config remains very similar:

```toml
model = "deepseek/deepseek-v4-flash"

[taskset]
id = "reverse-text-v1"

[harness]
id = "default"

[sampling]
temperature = 1.0
```

Validate the config by using `uv run gepa @ config.toml --dry-run`. To run GEPA, use `uv run gepa @ config.toml`. CLI arguments overwrite toml arguments when both are present.

## Common config values
- `model` / `-m` — model for the rollouts under optimization (default: `deepseek/deepseek-v4-flash`, same as eval)
- `reflection_model` / `reflection_client` - the model/endpoint that proposes new prompts (default: reuse `model` / `client`)
- `num_train` - tasks reserved for reflection minibatches (default: 100)
- `num_val` - tasks held out to score each prompt candidate for the pareto frontier (default: 50)
- `max_metric_calls` - total rollouts the run may spend (default: 500)
- `reflection_minibatch_size` - train tasks sampled per reflection step (default: 3)
- `initial_prompt` - seed system prompt (default: first task's `Task.system_prompt`)
- `reflection_columns` - extra per-trace fields (from the `trace.info`, else the task) to surface to the reflection model
- `max_concurrent` / `-c` - caps how many rollouts are in flight at once (default: 128)
- `shuffle` - shuffle tasks before the train/val split (fixed seed, so the split is reproducible)

The seed system prompt comes from `--initial-prompt`, or else the first task that sets `Task.system_prompt`. Tasksets that bake instructions into the user `prompt` instead (e.g. `gsm8k-v1`) need an explicit `--initial-prompt`.

## Output

The output from gepa runs are written into `outputs/<taskset>--<model>--<harness>/<uuid>/`, matching `eval`. The folder holds the used `config.toml`, every rollout's trace in `traces.jsonl`, and GEPA's own artifacts (`candidates.json`, `run_log.json`, ...). The best performing system prompt is printed to stdout when the run finishes.

## Limitations

- **Group-reward tasksets are not supported** -- GEPA scores one rollout per task, but `@group_reward` compares two or more, so tasksets like these are rejected upfront.

