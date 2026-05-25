import verifiers as vf


class NestedHarnessConfig(vf.HarnessConfig):
    program: str | None = f"{__name__}:parent_program"
    toolsets: dict[str, dict[str, str]] = {"nested": {"fn": f"{__name__}:load_toolset"}}
    metrics: list[str] = [f"{__name__}:child_calls"]


async def child_program(task, state):
    state["answer"] = str(task["prompt"]).upper()
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


async def call_harness(prompt, harness, state):
    _ = state
    task = vf.Task({"prompt": prompt}).freeze()
    child_state = await harness.run(task)
    return child_state["answer"]


@vf.metric
async def child_calls(task, state) -> float:
    return float(len(state["child_answers"]))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state["answer"] == task["answer"])


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


def load_tasks():
    return [
        {
            "prompt": (
                "Ask child harnesses to uppercase: " + ", ".join(child_prompts) + "."
            ),
            "child_prompts": child_prompts,
            "answer": " ".join(prompt.upper() for prompt in child_prompts),
        }
        for child_prompts in CHILD_PROMPT_GROUPS
    ]


def load_child_harness():
    return vf.Harness(config=vf.HarnessConfig(program=f"{__name__}:child_program"))


def load_toolset(config: vf.ToolsetConfig | None = None):
    async def call_child_harness(prompt, state):
        return await call_harness(prompt, load_child_harness(), state)

    call_child_harness.__name__ = "call_harness"
    call_child_harness.__doc__ = call_harness.__doc__
    return vf.Toolset(
        tools=[call_child_harness],
        config=config,
    )


async def parent_program(task, state):
    tools = state.get_tools()
    answers = []
    for prompt in task["child_prompts"]:
        answer = await tools["call_harness"](prompt=prompt)
        answers.append(answer)
    state["child_answers"] = answers
    state["answer"] = " ".join(answers)
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


class NestedTaskset(vf.Taskset):
    pass


class NestedHarness(vf.Harness):
    pass


class NestedEnvConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig(
        tasks=f"{__name__}:load_tasks",
        rewards=[f"{__name__}:exact_answer"],
    )
    harness: NestedHarnessConfig = NestedHarnessConfig()


def load_environment(config: NestedEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=NestedTaskset(config=config.taskset),
        harness=NestedHarness(config=config.harness),
    )
