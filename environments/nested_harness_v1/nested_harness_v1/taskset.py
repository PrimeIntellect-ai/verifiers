import verifiers.v1 as vf

from .servers.toolset import NestedToolsetConfig

CHILD_PROMPT_GROUPS = [
    ["hello"],
    ["open", "source"],
    ["taskset", "harness"],
    ["runtime", "boundary"],
    ["sandbox", "lease"],
    ["toolset", "scope"],
    ["group", "reward"],
    ["endpoint", "proxy"],
    ["cleanup", "signals"],
    ["harbor", "tasks"],
]


class NestedTasksetConfig(vf.TasksetConfig):
    toolsets: list[vf.ToolsetConfig] = [NestedToolsetConfig()]


class NestedHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


class NestedTask(vf.Task):
    child_prompts: list[str]
    answer: str


class NestedTaskset(vf.Taskset[NestedTasksetConfig]):
    task_type = NestedTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Ask child harnesses to uppercase: "
                        + ", ".join(child_prompts)
                        + ".",
                    }
                ],
                "child_prompts": child_prompts,
                "answer": " ".join(prompt.upper() for prompt in child_prompts),
            }
            for child_prompts in CHILD_PROMPT_GROUPS
        ]

    @vf.metric
    async def child_calls(self, state: vf.State) -> float:
        answers = state.extras.get("child_answers")
        return float(len(answers) if isinstance(answers, list) else 0)

    @vf.reward(weight=1.0)
    async def exact_answer(self, task: NestedTask, state: vf.State) -> float:
        messages = vf.get_messages(state.completion or [], role="assistant")
        answer = str(messages[-1].content or "").strip() if messages else ""
        return float(answer == task.answer)


class NestedHarness(vf.Harness[NestedHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = NestedTask.model_validate(context.task.model_dump())
        state = context.state
        tools = context.tools
        if tools is None:
            raise ValueError("NestedHarness requires tools.")
        answers: list[str] = []
        for prompt in task.child_prompts:
            result = await tools.call("nested_call_harness", {"prompt": str(prompt)})
            response = result.response
            answer = str(response.messages[0].content) if response.messages else ""
            answers.append(answer)
        state.extras["child_answers"] = answers
        answer = " ".join(answers)
        message = vf.AssistantMessage(content=answer)
        state.add_turn(
            vf.Turn(prompt=self.initial_messages(task), completion=[message])
        )
        state.stop("nested_completed")


def load_taskset(config: NestedTasksetConfig) -> NestedTaskset:
    return NestedTaskset(config=config)


def load_harness(config: NestedHarnessConfig) -> NestedHarness:
    return NestedHarness(config=config)
