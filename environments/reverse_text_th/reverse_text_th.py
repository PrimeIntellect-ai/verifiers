from __future__ import annotations

from difflib import SequenceMatcher

import verifiers.v1 as vf


@vf.reward(weight=1.0)
async def lcs_reward(task, state) -> float:
    return SequenceMatcher(None, state["prediction"], task["answer"]).ratio()


def source():
    return [
        {
            "prompt": [{"role": "user", "content": "Reverse this text: stressed"}],
            "text": "stressed",
            "answer": "desserts",
        },
        {
            "prompt": [{"role": "user", "content": "Reverse this text: drawer"}],
            "text": "drawer",
            "answer": "reward",
        },
    ]


async def reverse_program(task, state):
    state["prediction"] = task["text"][::-1]
    state["completion"] = [{"role": "assistant", "content": state["prediction"]}]
    return state


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[lcs_reward],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program=reverse_program,
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
