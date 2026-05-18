import verifiers as vf


class NestedHarnessConfig(vf.HarnessConfig):
    toolset: vf.ToolsetConfig | None = None


async def child_program(task, state):
    state["answer"] = str(task["prompt"]).upper()
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


class ChildHarness(vf.Harness[vf.HarnessConfig]):
    _default_program = child_program


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
    return ChildHarness()


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


class NestedTaskset(vf.Taskset[vf.TasksetConfig]):
    _default_source = source
    _default_rewards = (exact_answer,)


class NestedHarness(vf.Harness[NestedHarnessConfig]):
    _default_program = parent_program
    _default_metrics = (child_calls,)


def load_taskset(config: vf.TasksetConfig = vf.TasksetConfig()):
    return NestedTaskset(config=config)


def load_harness(config: NestedHarnessConfig = NestedHarnessConfig()):
    harness = NestedHarness(config=config)
    if "toolsets" not in harness.config.model_fields_set:
        harness.add_toolset({"nested": load_toolset(config=config.toolset)})
    return harness


load_environment = vf.Env.loader(
    taskset=load_taskset,
    harness=load_harness,
    taskset_config=vf.TasksetConfig,
    harness_config=NestedHarnessConfig,
)
