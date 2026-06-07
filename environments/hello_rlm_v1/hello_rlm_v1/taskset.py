import verifiers.v1 as vf
from harnesses import RLM, RLMConfig


def load_tasks(split: vf.TaskSplit = "train"):
    _ = split
    return [
        {
            "prompt": "Reply with exactly hello rlm.",
            "answer": "hello rlm",
        },
        {
            "prompt": "Reply with exactly taskset harness.",
            "answer": "taskset harness",
        },
        {
            "prompt": "Reply with exactly runtime boundary.",
            "answer": "runtime boundary",
        },
    ]


class HelloRLMTask(vf.Task):
    answer: str


class HelloRLMTasksetConfig(vf.TasksetConfig):
    pass


class HelloRLMTaskset(vf.Taskset[HelloRLMTasksetConfig]):
    task_type = HelloRLMTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return [HelloRLMTask.model_validate(record) for record in load_tasks(split)]

    @vf.reward(weight=1.0)
    async def exact_answer(self, task: HelloRLMTask, state: vf.State) -> float:
        command = state.artifacts.get("command")
        stdout = str(command.get("stdout") if isinstance(command, dict) else "")
        return float(task.answer.lower() in stdout.lower())


class HelloRLMHarnessConfig(RLMConfig):
    cwd: str | None = None


def load_taskset(config: HelloRLMTasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config=config)


def load_harness(config: HelloRLMHarnessConfig) -> RLM:
    return RLM(config=config)
