import verifiers as vf


async def child_program(
    task: vf.Task, state: vf.State
) -> dict[str, list[dict[str, str]]]:
    _ = state
    name = str(task["name"])
    return {"completion": [{"role": "assistant", "content": f"hello {name}"}]}


class ChildHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="child_program")


async def ask_subagent(name: str, state) -> str:
    """Ask a child harness to produce the greeting for one name."""
    harness = vf.Harness(config=ChildHarnessConfig())
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


def load_tasks(split: vf.TaskSplit = "train"):
    _ = split
    return [
        {
            "names": names,
            "prompt": [{"role": "user", "content": f"Names: {', '.join(names)}"}],
            "answer": ", ".join(f"hello {name}" for name in names),
        }
        for names in NAME_GROUPS
    ]


class SubagentTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_answer"]
    system_prompt: str = (
        "You are a parent coordinator. You must call ask_subagent once for "
        "each requested name. After all tool results are available, join "
        "the child answers with ', ' and output only that final joined text."
    )


class SubagentHarnessConfig(vf.HarnessConfig):
    metrics: list[str] = ["subagent_calls"]


class SubagentTaskset(vf.Taskset[SubagentTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(split)


class SubagentHarness(vf.Harness[SubagentHarnessConfig]):
    def load_toolsets(self, config: SubagentHarnessConfig) -> vf.Toolsets:
        _ = config
        return {"subagent": vf.Toolset(tools=[ask_subagent], scope="rollout")}


class SubagentEnvConfig(vf.EnvConfig):
    taskset: SubagentTasksetConfig = SubagentTasksetConfig()
    harness: SubagentHarnessConfig = SubagentHarnessConfig()


def load_environment(config: SubagentEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=SubagentTaskset(config=config.taskset),
        harness=SubagentHarness(config=config.harness),
    )
