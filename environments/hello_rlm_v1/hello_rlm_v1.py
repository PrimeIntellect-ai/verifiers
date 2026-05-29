import verifiers as vf
from harnesses import RLM, RLMConfig


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    stdout = str(state.get("command", {}).get("stdout") or "")
    return float(str(task["answer"]).lower() in stdout.lower())


def load_tasks(split: vf.TaskSplit = "train"):
    _ = split
    return [
        {
            "question": "Reply with exactly hello rlm.",
            "answer": "hello rlm",
        },
        {
            "question": "Reply with exactly taskset harness.",
            "answer": "taskset harness",
        },
        {
            "question": "Reply with exactly runtime boundary.",
            "answer": "runtime boundary",
        },
        {
            "question": "Reply with exactly sandbox lease.",
            "answer": "sandbox lease",
        },
        {
            "question": "Reply with exactly toolset scope.",
            "answer": "toolset scope",
        },
        {
            "question": "Reply with exactly group reward.",
            "answer": "group reward",
        },
        {
            "question": "Reply with exactly endpoint proxy.",
            "answer": "endpoint proxy",
        },
        {
            "question": "Reply with exactly cleanup signal.",
            "answer": "cleanup signal",
        },
        {
            "question": "Reply with exactly harbor task.",
            "answer": "harbor task",
        },
        {
            "question": "Reply with exactly recursive model.",
            "answer": "recursive model",
        },
    ]


class HelloRLMTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_answer"]


class HelloRLMTaskset(vf.Taskset[HelloRLMTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(split)


def load_taskset(config: HelloRLMTasksetConfig) -> HelloRLMTaskset:
    return HelloRLMTaskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset_config = config.taskset
    harness_config = config.harness
    assert isinstance(taskset_config, HelloRLMTasksetConfig)
    assert isinstance(harness_config, RLMConfig)
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=load_harness(harness_config),
    )
