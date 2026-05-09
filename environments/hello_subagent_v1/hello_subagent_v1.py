from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.utils.judge_utils import completion_text


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
    answer = completion_text(child_state.get("completion")).strip()
    state.setdefault("subagent_calls", []).append({"name": name, "answer": answer})
    return answer


@vf.metric
async def subagent_calls(task, state) -> float:
    return float(len(state.get("subagent_calls", [])))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(completion_text(state.get("completion")).strip() == task["answer"])


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
        bindings={"ask_subagent.harness": load_child_harness()},
        scope="rollout",
    )


def load_taskset(config: vf.TasksetConfig | None = None):
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


def load_harness(config: vf.HarnessConfig | None = None):
    return vf.Harness(
        toolsets=[load_toolset()],
        metrics=[subagent_calls],
        config=config,
    )


def load_environment(config: vf.EnvConfig | None = None):
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
