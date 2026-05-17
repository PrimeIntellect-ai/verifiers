import verifiers as vf

_child_harness: vf.Harness | None = None


class NestedHarnessConfig(vf.HarnessConfig):
    toolset: vf.ToolsetConfig | None = None


class NestedEnvConfig(vf.EnvConfig):
    harness: NestedHarnessConfig


async def child_program(task, state):
    state["answer"] = str(task["prompt"]).upper()
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


def child_harness() -> vf.Harness:
    global _child_harness
    if _child_harness is None:
        _child_harness = vf.Harness(program=child_program)
    return _child_harness


async def call_harness(prompt, state):
    _ = state
    harness = child_harness()
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


def source():
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


def load_toolset(config: vf.ToolsetConfig | None = None):
    return vf.Toolset(
        tools=[call_harness],
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


def load_taskset(config: vf.TasksetConfig):
    return vf.Taskset(
        source=source,
        rewards=[exact_answer],
        config=config,
    )


class NestedHarness(vf.Harness):
    def __init__(self, config: NestedHarnessConfig | None = None):
        config = NestedHarnessConfig(config)
        super().__init__(
            program=parent_program,
            toolsets=[load_toolset(config.toolset)],
            metrics=[child_calls],
            config=config,
        )


def load_harness(config: NestedHarnessConfig) -> NestedHarness:
    return NestedHarness(config=config)


def load_environment(config: NestedEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
