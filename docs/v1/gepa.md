# GEPA Prompt Optimization

[GEPA](https://github.com/gepa-ai/gepa) uses a model to improve a system prompt
based on task scores. Run it with:

```bash
uv run vf-gepa reverse-text
```

`vf-gepa` runs tasks, asks a model to review the results and propose better
prompts, then evaluates those prompts. It changes `TaskData.system_prompt`, not
the model's weights.

The [gepa skill](../../skills/gepa/SKILL.md) covers the full workflow, including
task selection, budgets, and comparison on tasks not used during optimization.

GEPA uses the same `env`, `client`, and `sampling` settings as evaluation:

```toml
model = "deepseek/deepseek-v4-flash"

[env.taskset]
id = "reverse-text"

[env.agent.harness]
id = "bash"

[sampling]
temperature = 1.0
```

Check the config with `uv run vf-gepa @ config.toml --dry-run`, then run it with
`uv run vf-gepa @ config.toml`. CLI flags override TOML values.

## Common config values

- `model` / `-m` — model that solves the tasks (default: `deepseek/deepseek-v4-flash`, same as eval)
- `reflection_model` / `reflection_client` — model/endpoint that proposes new prompts (default: reuse `model` / `client`)
- `select` — [tasks to split into the two groups](tasksets.md#selecting-tasks); shuffling is off by default
- `num_train` / `num_val` — tasks used to improve prompts and separate tasks used to compare them (defaults: 100 / 50)
- `max_total_rollouts` — total agent runs allowed (default: 500)
- `max_concurrent` / `-c` — episodes running at once (default: 128)

## Output

Results go to `output_dir / run.dir`, defaulting to
`outputs/<env>--<model>--<harness>--<short-id>/`, as in eval. The resolved config is
saved at `configs/resolved/gepa.json`.
The best system prompt is printed when the run finishes and written to `best_system_prompt.txt` in that folder.

Use the saved prompt in evaluation or training:

```bash
uv run vf-eval reverse-text \
  --env.taskset.system-prompt outputs/<run>/best_system_prompt.txt
```

## Limitations

**Tasksets** — If the taskset has no `TaskData.system_prompt`, set
`--initial-prompt` to provide a starting prompt. Keep task splits and scoring
fixed when comparing prompts.

**Environments** — Only `SingleAgentEnv` and its subclasses are supported.
Environments that run several agents or score traces together are rejected.

**Harnesses** — Any evaluation harness works. A harness with
`APPENDS_SYSTEM_PROMPT` uses the result as a system message. Otherwise it is
added to the user prompt.
