from __future__ import annotations

import verifiers.v1 as vf


async def ask_subagent(name: str, harness, runtime, state) -> str:
    """Ask a child language-model harness to produce the greeting for one name."""
    task = vf.Task(
        {
            "name": name,
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a child subagent. Reply with exactly "
                        f"`hello {name}` and no extra text."
                    ),
                },
                {"role": "user", "content": f"Say hello to {name}."},
            ],
        }
    ).freeze()
    child_state = await runtime.run_harness(
        harness,
        task,
        parent_state=state,
    )
    return completion_text(child_state.get("completion")).strip()


@vf.metric
async def subagent_calls(task, state) -> float:
    return float(len(state.get("child_rollouts", [])))


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(completion_text(state.get("completion")).strip() == task["answer"])


def completion_text(completion) -> str:
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return str(completion or "")


def source():
    return [
        {
            "names": ["world"],
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a parent coordinator. You must call "
                        "ask_subagent once for each requested name. After all "
                        "tool results are available, join the child answers "
                        "with ', ' and output only that final joined text."
                    ),
                },
                {"role": "user", "content": "Names: world"},
            ],
            "answer": "hello world",
        },
        {
            "names": ["prime", "verifiers"],
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a parent coordinator. You must call "
                        "ask_subagent once for each requested name. After all "
                        "tool results are available, join the child answers "
                        "with ', ' and output only that final joined text."
                    ),
                },
                {"role": "user", "content": "Names: prime, verifiers"},
            ],
            "answer": "hello prime, hello verifiers",
        },
    ]


def load_child_harness(config=None):
    return vf.Harness(config=config)


def load_toolset(config=None):
    return vf.Toolset(
        tools=[ask_subagent],
        objects={"child_harness": load_child_harness},
        bindings={"ask_subagent.harness": "objects.child_harness"},
        config=config,
    )


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[exact_answer], config=config)


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
