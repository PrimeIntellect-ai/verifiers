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
    child_state = await harness.run(task)
    answer = completion_text(child_state.get("completion")).strip()
    state.setdefault("subagent_calls", []).append({"name": name, "answer": answer})
    return answer


@vf.metric
async def subagent_calls(task, state) -> float:
    return float(len(state.get("subagent_calls", [])))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(completion_text(state.get("completion")).strip() == task["answer"])


def source():
    return [
        {
            "names": ["world"],
            "prompt": [{"role": "user", "content": "Names: world"}],
            "answer": "hello world",
        },
        {
            "names": ["prime", "verifiers"],
            "prompt": [{"role": "user", "content": "Names: prime, verifiers"}],
            "answer": "hello prime, hello verifiers",
        },
    ]


def load_child_harness(config=None):
    def child_harness(state):
        runtime = state.runtime()
        return vf.Harness(
            client=runtime.model_client(state),
            model=runtime.model(state),
            sampling_args=runtime.sampling_args(state),
            config=config,
        )

    return child_harness


def load_toolset(config=None):
    return vf.Toolset(
        tools=[ask_subagent],
        bindings={
            "ask_subagent.harness": load_child_harness(
                getattr(config, "child_harness", None)
            )
        },
        scope="rollout",
        config=config,
    )


def load_taskset(config=None):
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


def load_harness(config=None):
    return vf.Harness(
        toolsets=[load_toolset(getattr(config, "toolset", None))],
        metrics=[subagent_calls],
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
