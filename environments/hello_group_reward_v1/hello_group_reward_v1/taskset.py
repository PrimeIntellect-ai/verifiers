from difflib import SequenceMatcher

import verifiers.v1 as vf


SYSTEM_PROMPT = """\
You are testing group-aware scoring. Each rollout receives one candidate answer.
Return the assigned candidate exactly.
"""


class GroupRewardTasksetConfig(vf.TasksetConfig):
    system_prompt: str = SYSTEM_PROMPT
    num_examples: int = -1


class GroupRewardHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


class GroupRewardTask(vf.Task):
    question: str
    target: str
    answer: str
    near_answer: str
    partial_answer: str
    wrong_answer: str


class GroupRolloutTask(GroupRewardTask):
    parent_task_id: str
    candidate_id: str
    candidate_answer: str
    rollout_index: int


def group_reward_task(
    task_id: str,
    question: str,
    target: str,
    near: str,
    partial: str,
    wrong: str,
) -> vf.JsonData:
    return {
        "task_id": task_id,
        "question": question,
        "target": target,
        "prompt": question,
        "near_answer": near,
        "partial_answer": partial,
        "wrong_answer": wrong,
    }


TASKS: list[vf.JsonData] = [
    group_reward_task(
        "distributed-systems",
        "Describe v1 verifiers in one short phrase.",
        "composable tasksets and harnesses with group-aware scoring",
        "composable tasksets and harnesses with rollout scoring",
        "tasksets and harnesses",
        "a single monolithic environment object",
    ),
    group_reward_task(
        "runtime-boundary",
        "Describe the v1 runtime boundary in one short phrase.",
        "serializable task and state with hidden runtime handles",
        "serializable task and state with runtime handles",
        "task and state",
        "global objects stored directly in every task",
    ),
    group_reward_task(
        "toolset-scope",
        "Describe v1 toolset scope in one short phrase.",
        "rollout group and global tool lifetimes",
        "rollout and group tool lifetimes",
        "tool lifetimes",
        "static imports with no runtime handles",
    ),
    group_reward_task(
        "sandbox-sharing",
        "Describe sandbox sharing in one short phrase.",
        "borrowed sandbox handles shared across nested stages",
        "sandbox handles shared across stages",
        "shared sandbox handles",
        "new isolated machines for every function call",
    ),
    group_reward_task(
        "endpoint-controls",
        "Describe endpoint controls in one short phrase.",
        "nested programs inherit active model endpoint controls",
        "programs inherit model endpoint controls",
        "model endpoint controls",
        "hardcoded providers inside tasks",
    ),
    group_reward_task(
        "users",
        "Describe v1 users in one short phrase.",
        "task-owned follow-up messages between assistant turns",
        "follow-up messages between assistant turns",
        "follow-up messages",
        "metrics computed before any rollout starts",
    ),
    group_reward_task(
        "program-uploads",
        "Describe program uploads in one short phrase.",
        "task fields and files staged before harness execution",
        "files staged before harness execution",
        "staged files",
        "reward weights serialized into model prompts",
    ),
    group_reward_task(
        "cleanup-hooks",
        "Describe cleanup hooks in one short phrase.",
        "final artifact collection after rewards and metrics",
        "artifact collection after metrics",
        "artifact collection",
        "dataset filtering before import time",
    ),
    group_reward_task(
        "harbor-taskset",
        "Describe HarborTaskset in one short phrase.",
        "task directories converted into sandboxed rollout rows",
        "task directories converted into rollout rows",
        "task directories",
        "chat templates stored in every reward function",
    ),
    group_reward_task(
        "advantage-baseline",
        "Describe group advantages in one short phrase.",
        "rollout rewards centered against a group baseline",
        "rewards centered against a baseline",
        "centered rewards",
        "single responses scored without group context",
    ),
]


class GroupRewardTaskset(vf.Taskset[GroupRewardTasksetConfig]):
    task_type = GroupRewardTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return [
            GroupRewardTask.model_validate(record)
            for record in load_tasks(num_examples=self.config.num_examples)
        ]

    async def init_group(
        self, task: GroupRewardTask, num_rollouts: int
    ) -> tuple[list[GroupRolloutTask], list[vf.State]]:
        candidates = [
            ("exact", task.target),
            ("near", task.near_answer),
            ("partial", task.partial_answer),
            ("off-topic", task.wrong_answer),
        ]
        tasks: list[GroupRolloutTask] = []
        states: list[vf.State] = []
        for rollout_index in range(num_rollouts):
            candidate_id, candidate_answer = candidates[rollout_index % len(candidates)]
            task_data = task.to_record()
            task_data.pop("task_id", None)
            group_task = GroupRolloutTask.model_validate(
                {
                    **task_data,
                    "parent_task_id": task.task_id,
                    "candidate_id": candidate_id,
                    "candidate_answer": candidate_answer,
                    "rollout_index": rollout_index,
                    "prompt": (
                        f"Question: {task.question}\n"
                        f"Assigned candidate id: {candidate_id}\n"
                        f"Assigned candidate answer: {candidate_answer}\n\n"
                        "Return the assigned candidate answer exactly."
                    ),
                    "max_turns": 1,
                }
            )
            state = vf.State(task_id=group_task.task_id)
            state.scratch["group_setup"] = {
                "base_task_id": task.task_id,
                "num_rollouts": num_rollouts,
                "candidate_id": candidate_id,
            }
            tasks.append(group_task)
            states.append(state)
        return tasks, states

    @vf.metric
    async def answer_length(self, state: vf.State) -> float:
        return float(len(str(state.scratch.get("answer") or "")))

    @vf.reward(weight=0.1)
    async def rollout_similarity(
        self, task: GroupRolloutTask, state: vf.State
    ) -> float:
        return candidate_quality(task.target, str(state.scratch.get("answer") or ""))

    @vf.metric(stage="group")
    async def group_quality(
        self, tasks: list[GroupRolloutTask], states: list[vf.State]
    ) -> list[float]:
        return [
            candidate_quality(task.target, str(state.scratch.get("answer") or ""))
            for task, state in zip(tasks, states, strict=True)
        ]

    @vf.metric(stage="group")
    async def group_rank(
        self, tasks: list[GroupRolloutTask], states: list[vf.State]
    ) -> list[float]:
        qualities = [
            candidate_quality(task.target, str(state.scratch.get("answer") or ""))
            for task, state in zip(tasks, states, strict=True)
        ]
        return [float(rank) for rank in dense_ranks(qualities)]

    @vf.reward(stage="group", weight=1.0)
    async def relative_group_reward(
        self, tasks: list[GroupRolloutTask], states: list[vf.State]
    ) -> list[float]:
        qualities = [
            candidate_quality(task.target, str(state.scratch.get("answer") or ""))
            for task, state in zip(tasks, states, strict=True)
        ]
        if not qualities:
            return []
        low = min(qualities)
        high = max(qualities)
        if high == low:
            return [0.5 for _ in qualities]
        return [(quality - low) / (high - low) for quality in qualities]

    grpo = staticmethod(vf.advantages.grpo)


class GroupRewardHarness(vf.Harness[GroupRewardHarnessConfig]):
    async def _run(
        self,
        task: GroupRolloutTask,
        state: vf.State,
        *,
        ctx: vf.RolloutContext,
        runtime: vf.RuntimeSession | None = None,
        tools: vf.MCPToolRegistry | None = None,
        user: vf.MCPToolRegistry | None = None,
    ) -> None:
        _ = ctx, runtime, tools, user
        answer = task.candidate_answer
        message = vf.AssistantMessage(content=answer)
        state.scratch["answer"] = answer
        state.scratch["candidate_id"] = task.candidate_id
        state.add_turn(
            vf.Turn(prompt=self.initial_messages(task), completion=[message])
        )
        state.stop("candidate_program")


def candidate_quality(target: str, answer: str) -> float:
    if not answer:
        return 0.0
    ratio = SequenceMatcher(None, answer.lower(), target.lower()).ratio()
    length_ratio = min(len(answer), len(target)) / max(len(answer), len(target))
    return round(0.8 * ratio + 0.2 * length_ratio, 6)


def dense_ranks(values: list[float]) -> list[int]:
    ordered = sorted(set(values), reverse=True)
    return [ordered.index(value) + 1 for value in values]


def load_tasks(num_examples: int = -1):
    records = TASKS if num_examples < 0 else TASKS[:num_examples]
    for index, record in enumerate(records):
        yield {
            **record,
            "example_id": index,
            "answer": record["target"],
            "prompt": [
                {
                    "role": "user",
                    "content": str(record["question"]),
                }
            ],
            "max_turns": 1,
        }


def load_taskset(config: GroupRewardTasksetConfig) -> GroupRewardTaskset:
    return GroupRewardTaskset(config=config)


def load_harness(config: GroupRewardHarnessConfig) -> GroupRewardHarness:
    return GroupRewardHarness(config=config)
