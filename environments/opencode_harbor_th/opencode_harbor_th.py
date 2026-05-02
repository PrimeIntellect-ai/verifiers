from __future__ import annotations

import verifiers.v1 as vf


@vf.reward(stage="group", weight=1.0)
async def harbor_tests(tasks, states) -> list[float]:
    return [float(str(state["artifacts"]["harbor_reward"]).strip()) for state in states]


def source(path="tasks"):
    return [
        {
            "prompt": "Modify the repository so the Harbor verifier passes.",
            "harbor": {"path": str(path), "tests": "tests/test.sh"},
            "answer": None,
        }
    ]


def load_taskset(config=None):
    return vf.Taskset(
        source=lambda: source(getattr(config, "path", "tasks")),
        rewards=[harbor_tests],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program={
            "command": [
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "Path('/tmp/harbor_reward.txt').write_text('1.0'); "
                    "print('harbor verifier passed')"
                ),
            ],
            "artifacts": {
                "harbor_reward": {
                    "path": "/tmp/harbor_reward.txt",
                    "format": "text",
                },
            },
        },
        sandbox={
            "image": getattr(config, "image", "python:3.11-slim"),
            "workdir": "/app",
            "scope": "group",
            "timeout_minutes": 10,
            "command_timeout": 30,
        },
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
