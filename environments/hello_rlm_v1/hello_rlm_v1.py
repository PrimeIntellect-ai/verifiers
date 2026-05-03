from __future__ import annotations

import verifiers.v1 as vf


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    return float(state["completion"][0]["content"] == task["answer"])


@vf.metric
async def rlm_turns(task, state) -> float:
    return float(state["command"]["returncode"] == 0)


def source():
    return [
        {
            "prompt": "Reply with exactly hello rlm.",
            "answer": "hello rlm",
        }
    ]


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program={
            "command": [
                "python",
                "-c",
                "print('hello rlm')",
            ],
        },
        metrics=[rlm_turns],
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
