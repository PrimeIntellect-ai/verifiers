---
name: gepa
description: Improve a taskset's system prompt with vf-gepa. Use to choose task splits and budgets, run prompt optimization, inspect candidates, and compare the winning prompt with a baseline.
---

# Optimize Prompts with GEPA

Use GEPA to improve `TaskData.system_prompt` against a task's reward. It changes
the prompt, not model weights. Keep the user's model, harness, dataset, and
scoring requirements fixed.

## Check the task first

1. Confirm that the package loads and a small evaluation produces sensible scores.
2. Inspect wrong answers and errors. Fix broken setup or scoring before optimizing against it.
3. Confirm the environment is `SingleAgentEnv` or a subclass. GEPA rejects other environments, including multi-agent runs.
4. Choose the starting prompt. GEPA uses `initial_prompt` when supplied, otherwise the first selected task with a system prompt. `--initial-prompt` takes text, not a file path.

A harness with `APPENDS_SYSTEM_PROMPT` passes the prompt separately; other
harnesses add it to the user prompt. Check that this works with the task's input
format. See [GEPA](../../docs/v1/gepa.md) and
[evaluate-environments](../evaluate-environments/SKILL.md).

## Reserve tasks for the final comparison

GEPA uses two groups: `num_train` tasks to propose improvements and `num_val`
tasks to select candidates. Its validation scores influence the winning prompt,
so reserve a third, untouched group for the final baseline comparison.

Use `select.include` and `select.exclude`, or the taskset's own split option, to
keep final-test tasks out of GEPA's input. There is no universal taskset `split`
field. Keep dataset versions, task IDs, and selection settings with the run, and
check for overlap or duplicate questions across groups.

GEPA applies `select`, then takes the first `num_train + num_val` tasks. Shuffling
is off by default; enable it with `select.shuffle` and choose its `select.seed`.
The top-level `seed` controls the optimizer, not task selection. Keep enough
tasks for both groups. Before shuffling an infinite taskset, bound it with closed
`select.include.idx` ranges. See [task selection](../../docs/v1/tasksets.md#selecting-tasks).

## Configure and run

Read the taskset's available options:

```bash
uv run vf-gepa my-task --env.agent.harness.id null --help
```

Use the requested settings and budget. This small example can be saved as
`configs/my-gepa.toml`:

```toml
model = "openai/gpt-5-mini"
num_train = 8
num_val = 8
max_total_rollouts = 64
max_concurrent = 1

[env.taskset]
id = "reverse-text"

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
```

`model` solves tasks; `reflection_model` proposes prompts. The reflection model
and client default to `model` and `client`, or can be set with `reflection_model`
and `reflection_client`. Task judges still have independent settings.

`max_total_rollouts` sets the optimizer's rollout budget, not a dollar limit.
Reflection, judging, and sandbox use also cost resources. Choose concurrency for
the available runtime capacity; the default is 128.

```bash
uv run vf-gepa @ configs/my-gepa.toml --dry-run
uv run vf-gepa @ configs/my-gepa.toml
```

The dry run checks config and the environment type, not dataset loading, task
count, setup, or credentials. Keep results enabled and use a fresh run directory.
The CLI refuses a directory with saved traces and has no evaluation-style
`--resume` option.

## Inspect the result

Read `configs/resolved/gepa.json`, `traces.jsonl`, and `best_system_prompt.txt`
in the output directory. Each line of `traces.jsonl` is an episode, and its task
data records the candidate prompt. See [trace inspection](../../docs/v1/debugging.md).

Inspect errors alongside rewards: an episode without scores can count as zero
in optimization. Check that improvements come from better answers rather than
grader shortcuts, answer memorization, or changes to the intended task.
Use [audit-envs](../audit-envs/SKILL.md) if the reward looks unreliable.

By default, the reflection model receives the task's prompt text, final reply,
reward, and any error or stop condition. Add `reflection_columns` when it needs
specific grading details from `trace.info` or task fields. Check what those
fields contain before sending them; the default record is not the full trace.

## Compare with the baseline

Run the starting prompt and the winner on the same untouched tasks with the
same model, harness, runtime, sampling, and scoring. Put those settings in an
evaluation config. If `initial_prompt` supplied the starting text, use that text
for the baseline too.

Apply the winner with:

```bash
uv run vf-eval @ configs/test-eval.toml \
  --env.taskset.system-prompt outputs/<gepa-run>/best_system_prompt.txt
```

Keep baseline and candidate outputs in separate run directories. Report scores,
task counts, failures, and the optimization budget. Repeat comparisons when
sampling or judge variation could explain the difference. If no untouched tasks
were tested, report a validation improvement, not a demonstrated gain on new tasks.
