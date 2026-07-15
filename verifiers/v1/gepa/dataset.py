"""Train/val split and upfront validation for a GEPA run.

v1 tasksets have no generic train/val split concept (`TasksetConfig` has no `split` field;
individual tasksets define ad hoc ones inconsistently), so GEPA carves one out of the tasks
`Taskset.select` hands it (the shared fixed-seed shuffle, reproducible across runs like every
other entrypoint): two disjoint slices.
"""

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig, resolve_system_prompt


def reject_group_reward_tasksets(tasks: list[Task]) -> None:
    """GEPA scores one rollout per task (`n=1`) to get the per-task scalar its pareto frontier
    needs, but `@group_reward`s compare a task's rollouts and need >=2 — so a group-reward
    taskset can't be optimized this way (`Environment.episode` would raise on the first batch).
    Reject it up front with a clear message instead of crashing mid-run. A taskset loads one task
    type, so the first task is representative (as `run_eval` also assumes)."""
    if tasks and discover_decorated(tasks[0], "group_reward"):
        raise ValueError(
            "this taskset defines @group_reward(s), which compare >=2 rollouts of a task; GEPA "
            "optimizes a single system prompt from per-task scalar scores (one rollout per task) "
            "and can't score group rewards. Optimize a taskset without @group_reward instead."
        )


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


def resolve_gepa_seed_prompt(config: TasksetConfig, tasks: list[Task]) -> str:
    """The config-layer system prompt GEPA starts from.

    Prefer `--taskset.system-prompt` / `--taskset.system-prompt-file` when set; otherwise
    bootstrap from the first bake-in `TaskData.system_prompt` on `tasks` (expected to be
    pre-overlay / `select(apply_config=False)`). Tasksets that only put instructions in
    `prompt` (e.g. `gsm8k-v1`) need an explicit config seed.

    How the prompt reaches the model — a real system message vs. folded into the user
    prompt — is the harness's call (`Harness.resolve_prompt` / `compose_system_prompt`).
    """
    if (override := resolve_system_prompt(config)) is not None:
        return override
    for task in tasks:
        if task.data.system_prompt is not None:
            return task.data.system_prompt
    raise ValueError(
        "no config system prompt and no task sets TaskData.system_prompt — some "
        "tasksets bake instructions into `prompt` instead (e.g. gsm8k-v1). Pass "
        "--taskset.system-prompt or --taskset.system-prompt-file as the GEPA seed, "
        "or pick a taskset that sets a bake-in system_prompt (e.g. reverse-text-v1, "
        "lean, textarena)."
    )
