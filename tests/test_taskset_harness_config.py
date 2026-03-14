from datasets import Dataset

from verifiers.envs.experimental.harnesses.acp_agent import ACPHarness
from verifiers.envs.experimental.harnesses.opencode import OpenCodeHarness
from verifiers.envs.experimental.tasksets.base import StaticTaskSet, Task


def _build_static_taskset(*, task_config: dict, agent_workdir: str | None = None):
    return StaticTaskSet(
        dataset=Dataset.from_list([{"task": "example", "prompt": "", "answer": ""}]),
        task_factory=lambda state: Task(),
        task_config=task_config,
        agent_workdir=agent_workdir,
    )


def test_static_taskset_builds_acp_harness_from_shared_task_config():
    taskset = _build_static_taskset(
        task_config={
            "agent": {
                "timeout_sec": 600,
                "harness": {
                    "transport": "acp",
                    "agent": "claude-code",
                    "cwd": "/testbed",
                },
            }
        },
        agent_workdir="/workspace",
    )

    harness = taskset.build_harness()

    assert isinstance(harness, ACPHarness)
    assert harness.command == ("claude", "acp")
    assert harness.cwd == "/testbed"
    assert harness.timeout_seconds == 600.0


def test_static_taskset_builds_interceptor_harness_from_shared_task_config():
    taskset = _build_static_taskset(
        task_config={
            "agent": {
                "harness": {
                    "transport": "interceptor",
                    "agent": "opencode",
                    "disabled_tools": ["question"],
                }
            }
        },
        agent_workdir="/workspace",
    )

    harness = taskset.build_harness()

    assert isinstance(harness, OpenCodeHarness)
    assert harness.agent_workdir == "/workspace"
    assert harness.disabled_tools == ["question"]
