from __future__ import annotations

from collections.abc import Mapping
from difflib import SequenceMatcher
from statistics import mean

import verifiers.v1 as vf


SYSTEM_PROMPT = """\
You are testing group-aware scoring. Each rollout receives one candidate answer.
Return the assigned candidate exactly.
"""


TASKS: list[dict[str, object]] = [
    {
        "task_id": "distributed-systems",
        "question": "Describe v1 verifiers in one short phrase.",
        "target": "composable tasksets and harnesses with group-aware scoring",
        "candidates": [
            {
                "id": "exact",
                "answer": "composable tasksets and harnesses with group-aware scoring",
            },
            {
                "id": "missing-group",
                "answer": "composable tasksets and harnesses with rollout scoring",
            },
            {
                "id": "too-short",
                "answer": "tasksets and harnesses",
            },
            {
                "id": "off-topic",
                "answer": "a single monolithic environment object",
            },
        ],
    },
    {
        "task_id": "runtime-boundary",
        "question": "Describe the v1 runtime boundary in one short phrase.",
        "target": "serializable task and state with hidden runtime handles",
        "candidates": [
            {
                "id": "exact",
                "answer": "serializable task and state with hidden runtime handles",
            },
            {
                "id": "leaky",
                "answer": "serializable task and state with public runtime handles",
            },
            {
                "id": "partial",
                "answer": "task and state with runtime handles",
            },
            {
                "id": "wrong",
                "answer": "global objects stored directly in every task",
            },
        ],
    },
]


class GroupRewardTaskset(vf.Taskset):
    async def init_group(
        self, task: vf.Task, num_rollouts: int
    ) -> tuple[list[vf.Task], list[vf.State]]:
        candidates = task.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("hello_group_reward_v1 tasks require candidates.")
        tasks: list[vf.Task] = []
        states: list[vf.State] = []
        for rollout_index in range(num_rollouts):
            candidate = candidates[rollout_index % len(candidates)]
            if not isinstance(candidate, Mapping):
                raise TypeError("candidate entries must be mappings.")
            candidate_id = str(candidate["id"])
            candidate_answer = str(candidate["answer"])
            group_task = vf.Task(
                {
                    **dict(task),
                    "candidate_id": candidate_id,
                    "candidate_answer": candidate_answer,
                    "rollout_index": rollout_index,
                    "prompt": [
                        {
                            "role": "user",
                            "content": (
                                f"Question: {task['question']}\n"
                                f"Assigned candidate id: {candidate_id}\n"
                                f"Assigned candidate answer: {candidate_answer}\n\n"
                                "Return the assigned candidate answer exactly."
                            ),
                        }
                    ],
                    "max_turns": 1,
                }
            ).freeze()
            state = vf.State.for_task(group_task)
            state["group_setup"] = {
                "base_task_id": task["task_id"],
                "num_rollouts": num_rollouts,
                "candidate_id": candidate_id,
            }
            tasks.append(group_task)
            states.append(state)
        return tasks, states


async def candidate_program(task: vf.Task, state: vf.State) -> vf.State:
    answer = str(task["candidate_answer"])
    state["answer"] = answer
    state["candidate_id"] = task["candidate_id"]
    state["completion"] = [{"role": "assistant", "content": answer}]
    state.stop("candidate_program")
    return state


@vf.metric
async def answer_length(task, state) -> float:
    _ = task
    return float(len(str(state.get("answer") or "")))


@vf.reward(weight=0.1)
async def rollout_similarity(task, state) -> float:
    return candidate_quality(str(task["target"]), str(state.get("answer") or ""))


@vf.update(stage="group", priority=10)
async def summarize_group(tasks, states) -> None:
    qualities = [
        candidate_quality(str(task["target"]), str(state.get("answer") or ""))
        for task, state in zip(tasks, states)
    ]
    ranks = dense_ranks(qualities)
    best_index = min(
        range(len(states)),
        key=lambda index: (
            -qualities[index],
            len(str(states[index].get("answer") or "")),
        ),
    )
    candidates = [
        {
            "candidate_id": str(state.get("candidate_id")),
            "quality": qualities[index],
            "rank": ranks[index],
        }
        for index, state in enumerate(states)
    ]
    for index, state in enumerate(states):
        state["group_summary"] = {
            "best_candidate_id": str(states[best_index].get("candidate_id")),
            "best_answer": str(states[best_index].get("answer") or ""),
            "quality": qualities[index],
            "rank": ranks[index],
            "group_size": len(states),
            "candidates": candidates,
        }


@vf.metric(stage="group")
async def group_quality(tasks, states) -> list[float]:
    return [
        candidate_quality(str(task["target"]), str(state.get("answer") or ""))
        for task, state in zip(tasks, states)
    ]


@vf.metric(stage="group")
async def group_rank(tasks, states) -> list[float]:
    _ = tasks
    qualities = [
        float(state.get("group_summary", {}).get("quality", 0.0)) for state in states
    ]
    return [float(rank) for rank in dense_ranks(qualities)]


@vf.reward(stage="group", weight=1.0)
async def relative_group_reward(tasks, states) -> list[float]:
    _ = tasks
    qualities = [
        float(state.get("group_summary", {}).get("quality", 0.0)) for state in states
    ]
    if not qualities:
        return []
    low = min(qualities)
    high = max(qualities)
    if high == low:
        return [0.5 for _ in qualities]
    return [(quality - low) / (high - low) for quality in qualities]


@vf.advantage
async def centered_group_advantage(tasks, states) -> list[float]:
    _ = tasks
    rewards = [
        float(state.get("metrics", {}).get("relative_group_reward", 0.0))
        for state in states
    ]
    baseline = mean(rewards) if rewards else 0.0
    return [reward - baseline for reward in rewards]


@vf.cleanup(stage="group")
async def mark_group_cleaned(tasks, states) -> None:
    _ = tasks
    for state in states:
        state["group_cleaned"] = True


def candidate_quality(target: str, answer: str) -> float:
    if not answer:
        return 0.0
    ratio = SequenceMatcher(None, answer.lower(), target.lower()).ratio()
    length_ratio = min(len(answer), len(target)) / max(len(answer), len(target))
    return round(0.8 * ratio + 0.2 * length_ratio, 6)


def dense_ranks(values: list[float]) -> list[int]:
    ordered = sorted(set(values), reverse=True)
    return [ordered.index(value) + 1 for value in values]


def source(num_examples: int = -1):
    rows = TASKS if num_examples < 0 else TASKS[:num_examples]
    for index, row in enumerate(rows):
        yield {
            **row,
            "example_id": index,
            "answer": row["target"],
            "prompt": [
                {
                    "role": "user",
                    "content": str(row["question"]),
                }
            ],
            "max_turns": 1,
        }


def load_taskset(num_examples: int = -1, config=None) -> GroupRewardTaskset:
    def load_rows():
        return source(num_examples=num_examples)

    return GroupRewardTaskset(
        source=load_rows,
        system_prompt=SYSTEM_PROMPT,
        metrics=[answer_length, group_quality, group_rank],
        rewards=[rollout_similarity, relative_group_reward],
        advantages=[centered_group_advantage],
        updates=[summarize_group],
        cleanups=[mark_group_cleaned],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(program=candidate_program, max_turns=1, config=config)


def load_environment(num_examples: int = -1, config=None) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(
            num_examples=num_examples,
            config=getattr(config, "taskset", None),
        ),
        harness=load_harness(getattr(config, "harness", None)),
    )


def load_v1_environment(num_examples: int = -1, config=None) -> vf.Env:
    return load_environment(num_examples=num_examples, config=config)
