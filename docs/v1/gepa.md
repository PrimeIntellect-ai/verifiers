# GEPA Prompt Optimization

verifiers offers built in support for [GEPA](https://github.com/gepa-ai/gepa), an algorithm that optimizes a system prompt to maximize the downstream reward for a given taskset:

```bash
uv run gepa reverse-text-v1
```

`gepa` runs GEPA where a number of rollouts are done before a teacher LLM reflects on the results to propose a better `Task.system_prompt` without any gradient based training. It runs against native v1 tasksets.

GEPA reuses the same `taskset` / `harness` / `client` / `sampling` config as eval, so the `.toml` config remains very similar:

```toml
model = "deepseek/deepseek-v4-flash"

[taskset]
id = "reverse-text-v1"

[harness]
id = "bash"

[sampling]
temperature = 1.0
```

Validate the config by using `uv run gepa @ config.toml --dry-run`. To run GEPA, use `uv run gepa @ config.toml`. CLI arguments overwrite toml arguments when both are present.

## Common config values

- `model` / `-m` — model for the rollouts under optimization (default: `deepseek/deepseek-v4-flash`, same as eval)
- `reflection_model` / `reflection_client` — model/endpoint that proposes new prompts (default: reuse `model` / `client`)
- `num_train` / `num_val` — train tasks for reflection minibatches and held-out val tasks for the pareto frontier (defaults: 100 / 50)
- `max_total_rollouts` — total rollouts the run may spend (default: 500)
- `max_concurrent` / `-c` — caps how many rollouts are in flight at once (default: 128)

## Output

Results go under `outputs/<taskset>--<model>--<harness>/<uuid>/`, matching `eval`. The best system prompt is printed when the run finishes.

## Limitations

**Tasksets** — GEPA optimizes `Task.system_prompt`, so the taskset must provide one. Tasksets that bake instructions into the user `prompt` instead (e.g. `gsm8k-v1`) are not supported out of the box.

**Harnesses** — any eval harness works. With `APPENDS_SYSTEM_PROMPT`, the optimized prompt is used as a system message but otherwise is folded into the user prompt.
