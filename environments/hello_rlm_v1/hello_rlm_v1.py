import verifiers as vf


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    stdout = str(state.get("command", {}).get("stdout") or "")
    return float(str(task["answer"]).lower() in stdout.lower())


def source():
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


def load_taskset(config: vf.TasksetConfig | None = None):
    return vf.Taskset(
        source=source,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config: vf.RLMConfig | None = None):
    return vf.RLM(config=config)


def load_environment(config: vf.EnvConfig):
    harness_config = None if config.harness is None else vf.RLMConfig(config.harness)
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=harness_config),
    )
