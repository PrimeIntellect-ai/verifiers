"""Train/val split and upfront validation for a GEPA run.

v1 tasksets have no generic train/val split concept (`TasksetConfig` has no `split` field;
individual tasksets define ad hoc ones inconsistently), so GEPA carves one out of the tasks
`Taskset.select` hands it (the shared fixed-seed shuffle, reproducible across runs like every
other entrypoint): two disjoint slices.
"""

from verifiers.v1.task import Task


def split_tasks(
    tasks: list[Task], num_train: int, num_val: int
) -> tuple[list[Task], list[Task]]:
    """The selected tasks, split into disjoint `(train, val)` slices (`train` feeds reflection
    minibatches; `val` scores each candidate for the pareto frontier)."""
    if num_train + num_val > len(tasks):
        raise ValueError(
            f"requested {num_train} train + {num_val} val tasks, but the taskset only "
            f"loaded {len(tasks)}"
        )
    return tasks[:num_train], tasks[num_train : num_train + num_val]


def resolve_gepa_seed_prompt(tasks: list[Task], initial_prompt: str | None) -> str:
    """The system prompt GEPA starts optimizing from: `initial_prompt` if given, else the first
    task that sets `system_prompt`. Some tasksets (e.g. `gsm8k-v1`) bake instructions into
    `prompt` rather than `system_prompt` and can't be optimized this way — pass `--initial-prompt`
    to seed one explicitly.

    How the resolved prompt reaches the model at rollout time — a real system message vs. folded
    into the user prompt — is the harness's call: `Harness.resolve_prompt` owns that policy and
    warns when it folds, so it isn't re-policed here."""
    if initial_prompt is not None:
        return initial_prompt
    for task in tasks:
        if task.data.system_prompt is not None:
            return task.data.system_prompt
    raise ValueError(
        "no task in this taskset sets Task.system_prompt — some tasksets bake instructions "
        "directly into `prompt` instead (e.g. gsm8k-v1) and can't be optimized this way. Pass "
        "--initial-prompt to seed one explicitly, or pick a taskset whose load() sets "
        "system_prompt on its task data (e.g. reverse-text-v1, lean, textarena)."
    )
