import verifiers as vf


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    stdout = str(state.get("command", {}).get("stdout") or "")
    return float(str(task["answer"]).lower() in stdout.lower())


def load_tasks():
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


class HelloRLMTaskset(vf.Taskset):
    pass


class HelloRLMTasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    rewards: list[str] = ["exact_answer"]


class HelloRLMEnvConfig(vf.EnvConfig):
    taskset: HelloRLMTasksetConfig = HelloRLMTasksetConfig()
    harness: vf.RLMConfig = vf.RLMConfig()


def load_environment(config: HelloRLMEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=HelloRLMTaskset(config=config.taskset),
        harness=vf.RLM(config=config.harness),
    )
