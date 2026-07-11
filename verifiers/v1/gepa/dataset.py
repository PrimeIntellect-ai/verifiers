"""Train/val split and upfront validation for a GEPA run.

v1 tasksets have no generic train/val split concept (`TasksetConfig` has no `split` field;
individual tasksets define ad hoc ones inconsistently), so GEPA carves one out of
`taskset.load()` itself: an optional shuffle under a fixed seed (reproducible across runs,
like `run_eval`), then two disjoint slices.
"""

import logging
import random

from verifiers.v1.env import Environment
from verifiers.v1.task import Task

logger = logging.getLogger(__name__)

_SHUFFLE_SEED = 0  # fixed so the split is the same every run (reproducible)


def split_tasks(
    tasks: list[Task], num_train: int, num_val: int, shuffle: bool
) -> tuple[list[Task], list[Task]]:
    """The taskset's tasks, split into disjoint `(train, val)` slices (`train` feeds reflection
    minibatches; `val` scores each candidate for the pareto frontier)."""
    pool = list(tasks)
    if shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(pool)
    if num_train + num_val > len(pool):
        raise ValueError(
            f"requested {num_train} train + {num_val} val tasks, but the taskset only "
            f"loaded {len(pool)}"
        )
    return pool[:num_train], pool[num_train : num_train + num_val]


def resolve_gepa_seed_prompt(
    env: Environment, tasks: list[Task], initial_prompt: str | None
) -> str:
    """The system prompt GEPA starts optimizing from.

    Warns (doesn't block) if the configured harness has no system-message input
    (`APPENDS_SYSTEM_PROMPT` is False): optimization still works — the harness folds the
    candidate into the user prompt (`Harness.resolve_prompt`) — but the result isn't delivered
    as a real system message, so redeploy it the same way. Then resolves the seed:
    `initial_prompt` wins if given; else the first task that sets `system_prompt` (some
    tasksets, e.g. `gsm8k-v1`, bake instructions into `prompt` instead and can't be optimized
    this way — an explicit `--initial-prompt` is required for those)."""
    if not env.harness.APPENDS_SYSTEM_PROMPT:
        logger.warning(
            "harness %r has no system-message input (APPENDS_SYSTEM_PROMPT is False), so the "
            "candidate is folded into the user prompt rather than sent as a system message "
            "(Harness.resolve_prompt). Optimization still runs, but the result isn't used as a "
            "true system prompt — deploy it the same folded way, or use a harness that emits "
            "one (e.g. --harness.id default, null, rlm, or terminus_2).",
            env.harness.config.id,
        )
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
