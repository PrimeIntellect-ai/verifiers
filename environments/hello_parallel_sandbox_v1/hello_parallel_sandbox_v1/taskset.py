import asyncio

import verifiers.v1 as vf

SYSTEM_PROMPT = """Reply with the requested answer text only."""

TASKS: list[vf.JsonData] = [
    {
        "task_id": "exact-token",
        "answer": "prime-v1-shared-sandbox",
        "instruction": "Return exactly `prime-v1-shared-sandbox`.",
    },
    {
        "task_id": "reverse-token",
        "answer": "xobdnas-derahs",
        "instruction": "Return exactly the reverse of `shared-sandbox`.",
    },
    {
        "task_id": "joined-words",
        "answer": "taskset-harness-runtime",
        "instruction": "Return taskset, harness, and runtime joined by hyphens.",
    },
]


class ParallelSandboxTasksetConfig(vf.TasksetConfig):
    system_prompt: str = SYSTEM_PROMPT
    num_examples: int = -1


class ParallelSandboxHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


class ParallelSandboxTask(vf.Task):
    answer: str
    instruction: str


class ParallelSandboxTaskset(vf.Taskset[ParallelSandboxTasksetConfig]):
    task_type = ParallelSandboxTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        rows = (
            TASKS if self.config.num_examples < 0 else TASKS[: self.config.num_examples]
        )
        return [
            {
                **row,
                "row_id": index,
                "prompt": [{"role": "user", "content": str(row["instruction"])}],
                "max_turns": 1,
            }
            for index, row in enumerate(rows)
        ]

    @vf.update(priority=10)
    async def parallel_audit(self, task: ParallelSandboxTask, state: vf.State) -> None:
        response = assistant_text(state)
        file_audit, command_audit = await asyncio.gather(
            audit_exact_answer(task, response),
            audit_shape(task, response),
        )
        audits = [file_audit, command_audit]
        state.extras["parallel_audits"] = audits
        state.artifacts["parallel_audits"] = audits

    @vf.metric
    async def update_audits(self, state: vf.State) -> float:
        audits = state.extras.get("parallel_audits")
        return float(len(audits) if isinstance(audits, list) else 0)

    @vf.reward(weight=1.0)
    async def sandbox_stage_score(self, state: vf.State) -> float:
        audits = state.extras.get("parallel_audits")
        if not isinstance(audits, list) or not audits:
            return 0.0
        passed = [
            bool(audit.get("passed")) for audit in audits if isinstance(audit, dict)
        ]
        return sum(float(item) for item in passed) / len(passed)


async def audit_exact_answer(task: ParallelSandboxTask, response: str) -> vf.JsonData:
    await asyncio.sleep(0)
    expected = task.answer
    return {
        "name": "exact_answer",
        "passed": response.strip() == expected,
        "expected": expected,
        "observed": response.strip(),
    }


async def audit_shape(task: ParallelSandboxTask, response: str) -> vf.JsonData:
    await asyncio.sleep(0)
    expected = task.answer
    return {
        "name": "shape",
        "passed": bool(response.strip()) and "\n" not in response.strip(),
        "expected_length": len(expected),
        "observed_length": len(response.strip()),
    }


def assistant_text(state: vf.State) -> str:
    messages = [message for message in state.completion if message.role == "assistant"]
    return str(messages[-1].content or "") if messages else ""


def load_taskset(config: ParallelSandboxTasksetConfig) -> ParallelSandboxTaskset:
    return ParallelSandboxTaskset(config=config)


def load_harness(config: ParallelSandboxHarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)
