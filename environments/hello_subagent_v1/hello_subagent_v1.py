import verifiers as vf


async def ask_subagent(name: str, harness, state) -> str:
    """Ask a child language-model harness to produce the greeting for one name."""
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


def source():
    return [
        {
            "names": names,
            "prompt": [{"role": "user", "content": f"Names: {', '.join(names)}"}],
            "answer": ", ".join(f"hello {name}" for name in names),
        }
        for names in NAME_GROUPS
    ]


def load_child_harness():
    return vf.Harness()


def load_toolset():
    return vf.Toolset(
        tools=[ask_subagent],
        objects={"harness": load_child_harness},
        bindings={"ask_subagent.harness": "objects.harness"},
        scope="rollout",
    )


def load_taskset(config: vf.TasksetConfig):
    return vf.Taskset(
        source=source,
        system_prompt=(
            "You are a parent coordinator. You must call ask_subagent once for "
            "each requested name. After all tool results are available, join "
            "the child answers with ', ' and output only that final joined text."
        ),
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config: vf.HarnessConfig):
    return vf.Harness(
        toolsets=[load_toolset()],
        metrics=[subagent_calls],
        config=config,
    )


def load_environment(config: vf.EnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
