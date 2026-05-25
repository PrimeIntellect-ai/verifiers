import verifiers as vf


async def ask_subagent(name: str, state) -> str:
    """Ask a child language-model harness to produce the greeting for one name."""
    harness = load_child_harness()
    task = vf.Task(
        {
            "name": name,
            "system_prompt": (
                "You are a child subagent. Reply with exactly "
                f"`hello {name}` and no extra text."
            ),
            "prompt": [
                {"role": "user", "content": f"Say hello to {name}."},
            ],
        }
    ).freeze()
    child_state = state.for_task(task, borrow="model")
    child_state = await harness.run(task, child_state)
    messages = vf.get_messages(child_state.get("completion") or [], role="assistant")
    answer = str(messages[-1].content or "").strip() if messages else ""
    state.setdefault("subagent_calls", []).append({"name": name, "answer": answer})
    return answer


@vf.metric
async def subagent_calls(task, state) -> float:
    return float(len(state.get("subagent_calls", [])))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    answer = str(messages[-1].content or "").strip() if messages else ""
    return float(answer == task["answer"])


NAME_GROUPS = [
    ["world"],
    ["prime", "verifiers"],
    ["taskset", "harness", "runtime"],
    ["sandbox"],
    ["alpha", "beta"],
    ["delta", "epsilon", "zeta"],
    ["tools", "users"],
    ["group", "reward", "advantage"],
    ["mcp", "search"],
    ["open", "superintelligence", "stack"],
]


def load_tasks():
    return [
        {
            "names": names,
            "prompt": [{"role": "user", "content": f"Names: {', '.join(names)}"}],
            "answer": ", ".join(f"hello {name}" for name in names),
        }
        for names in NAME_GROUPS
    ]


def load_child_harness():
    return vf.Harness(config=vf.HarnessConfig())


def load_toolset():
    return vf.Toolset(
        tools=[ask_subagent],
        scope="rollout",
    )


class SubagentTasksetConfig(vf.TasksetConfig):
    tasks: str = f"{__name__}:load_tasks"
    rewards: list[str] = [f"{__name__}:exact_answer"]
    system_prompt: str = (
        "You are a parent coordinator. You must call ask_subagent once for "
        "each requested name. After all tool results are available, join "
        "the child answers with ', ' and output only that final joined text."
    )


class SubagentHarnessConfig(vf.HarnessConfig):
    toolsets: dict[str, dict[str, str]] = {
        "subagent": {"fn": f"{__name__}:load_toolset"}
    }
    metrics: list[str] = [f"{__name__}:subagent_calls"]


class SubagentTaskset(vf.Taskset):
    pass


class SubagentHarness(vf.Harness):
    pass


class SubagentEnvConfig(vf.EnvConfig):
    taskset: SubagentTasksetConfig = SubagentTasksetConfig()
    harness: SubagentHarnessConfig = SubagentHarnessConfig()


def load_environment(config: SubagentEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=SubagentTaskset(config=config.taskset),
        harness=SubagentHarness(config=config.harness),
    )
