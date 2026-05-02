from __future__ import annotations

import verifiers.v1 as vf


@vf.reward(weight=1.0)
async def exact_order(task, state) -> float:
    return float(state["prediction"] == task["answer"])


@vf.metric(stage="group")
async def group_sorted(tasks, states) -> list[float]:
    return [
        float(state["prediction"] == sorted(task["items"]))
        for task, state in zip(tasks, states)
    ]


def source():
    return [
        {
            "prompt": "Sort these letters alphabetically: delta",
            "items": list("delta"),
            "answer": sorted("delta"),
        },
        {
            "prompt": "Sort these names alphabetically: Ada, Grace, Linus",
            "items": ["Grace", "Linus", "Ada"],
            "answer": ["Ada", "Grace", "Linus"],
        },
    ]


async def sort_program(task, state):
    state["prediction"] = sorted(task["items"])
    state["completion"] = [
        {"role": "assistant", "content": ", ".join(state["prediction"])}
    ]
    return state


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[exact_order],
        metrics=[group_sorted],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program=sort_program,
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
