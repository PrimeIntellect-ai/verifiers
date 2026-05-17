import verifiers as vf


class NestedHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn=f"{__name__}:parent_program")
    metrics: list[vf.CallableConfig] = [vf.CallableConfig(fn=f"{__name__}:child_calls")]
    toolset: vf.ToolsetConfig | None = None


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


def load_child_harness():
    return vf.Harness(config=vf.HarnessConfig(program=f"{__name__}:child_program"))


def load_toolset(config: vf.ToolsetConfig | None = None):
    return vf.Toolset(
        tools=[call_harness],
        objects={"child_harness": load_child_harness},
        bindings={
            "call_harness.harness": "objects.child_harness",
        },
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


def load_taskset(config: vf.TasksetConfig = vf.TasksetConfig()):
    taskset_config = type(config).model_validate(
        {
            **config.model_dump(),
            "source": f"{__name__}:source",
            "rewards": [vf.CallableConfig(fn=f"{__name__}:exact_answer", weight=1.0)],
        }
    )
    return vf.Taskset(config=taskset_config)


def load_harness(config: NestedHarnessConfig = NestedHarnessConfig()):
    toolset = {"nested": {"fn": f"{__name__}:load_toolset"}}
    if config.toolset is not None:
        toolset["nested"]["config"] = config.toolset
    harness_config = NestedHarnessConfig.model_validate(
        {**config.model_dump(exclude_none=True), "toolsets": toolset}
    )
    base_config = {
        key: value
        for key, value in harness_config.model_dump().items()
        if key in vf.HarnessConfig.model_fields
    }
    return vf.Harness(config=vf.HarnessConfig.model_validate(base_config))


class NestedEnvConfig(vf.EnvConfig):
    taskset: vf.TasksetConfig = vf.TasksetConfig()
    harness: NestedHarnessConfig = NestedHarnessConfig()


def load_environment(config: NestedEnvConfig = NestedEnvConfig()):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
