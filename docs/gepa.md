# GEPA Prompt Optimization

verifiers includes built-in support for [GEPA](https://github.com/gepa-ai/gepa), an algorithm that optimizes a system prompt to maximize downstream reward on a taskset without gradient-based training:

```bash
uv run vf-gepa reverse-text
```

`vf-gepa` alternates between running rollouts on the taskset and asking a reflection LLM to review the trajectories and propose improvements to `TaskData.system_prompt`.

GEPA reuses the same `env` (taskset + agent), `client`, and `sampling` configuration as `vf-eval`, so its `.toml` config looks almost identical:

```toml
model = "deepseek/deepseek-v4-flash"

[env.taskset]
id = "reverse-text"

[env.agent.harness]
id = "bash"

[sampling]
temperature = 1.0
```

Validate the config with `uv run vf-gepa @ config.toml --dry-run`, and run optimization with `uv run vf-gepa @ config.toml`. CLI flags override `.toml` values when both are present.

## Common config values

- `model` / `-m` — model for the rollouts under optimization (default: `deepseek/deepseek-v4-flash`, same as `vf-eval`)
- `reflection_model` / `reflection_client` — teacher model and endpoint that propose new prompts (default: reuse `model` and `client`)
- `select` — [task selection](tasksets.md#selecting-tasks) applied before splitting into train and validation sets (shuffling is off by default)
- `num_train` / `num_val` — train tasks for reflection minibatches and validation tasks for scoring candidates on the Pareto frontier (defaults: `100` / `50`)
- `max_total_rollouts` — total rollouts the run may spend (default: `500`)
- `max_concurrent` / `-c` — max episodes in flight at once (default: `128`)

Task selection runs before the train/val split: the first `num_train` selected tasks feed reflection minibatches, and the next `num_val` tasks score candidate prompts (`select.seed` controls task-selection shuffling, while `seed` controls the optimizer's own sampling). Because the validation slice guides prompt selection during optimization, evaluate the final prompt on a separate held-out slice of tasks.

## Output

Results are written to `output_dir / run.dir`, which defaults to `outputs/<env>--<model>--<harness>--<short-id>/` just like `vf-eval`. The resolved config is saved at `configs/resolved/gepa.json`, and when the run finishes, the best system prompt is printed to the console and written to `best_system_prompt.txt`.

You can pass that file directly back into evaluation or training via `--env.taskset.system-prompt`:

```bash
uv run vf-eval reverse-text \
  --env.taskset.system-prompt outputs/<run>/best_system_prompt.txt
```

## Limitations

**Tasksets** — GEPA optimizes `TaskData.system_prompt`. If a taskset bakes instructions into the user `prompt` instead of setting `system_prompt` (e.g. `gsm8k`), pass `--initial-prompt` to seed a starting system prompt explicitly.

**Environments** — Only `SingleAgentEnv` and its subclasses are supported. Multi-agent environments are rejected because GEPA optimizes a single agent's prompt against per-trace rewards.

**Harnesses** — Any evaluation harness works. Harnesses with `APPENDS_SYSTEM_PROMPT` pass the optimized prompt as a system message; otherwise it is folded into the user prompt.
