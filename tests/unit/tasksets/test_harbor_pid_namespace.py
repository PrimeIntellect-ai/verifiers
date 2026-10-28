"""Every Harbor box a task sets up runs its commands in a PID namespace of its own,
a Compose task's separate verifier box included."""

import pytest

from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor import taskset
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTask,
    HarborTaskConfig,
)


class Backend:
    def __init__(self) -> None:
        self.isolated = 0

    async def isolate_processes(self) -> None:
        self.isolated += 1


class Box:
    """A box that answers every command and records whether it was isolated."""

    def __init__(self) -> None:
        self.backend = Backend()

    async def run(self, argv, env):
        return ProgramResult(exit_code=0, stdout="", stderr="")


def compose_task(tmp_path) -> HarborTask:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    build: .\n"
    )
    data = HarborData(prompt="p", task_dir=str(tmp_path))
    task = HarborTask(data, HarborTaskConfig())
    assert task.deployment_spec() is not None
    return task


@pytest.mark.parametrize("step", ["setup", "restore"])
async def test_a_compose_tasks_verifier_box_gets_a_pid_namespace(
    tmp_path, monkeypatch, step
):
    """A Compose task's services are containers, but its separate verifier is a box of
    its own: a grader there that kills every process outside a few paths (/mnt among
    them) otherwise killed Prime's command daemon (/mnt/sandboxd), and the box."""
    monkeypatch.setattr(taskset, "guard_onednn", lambda runtime: _none())
    monkeypatch.setattr(HarborTask, "entrypoint_dockerfile", lambda self: None)
    box = Box()
    await getattr(compose_task(tmp_path), step)(box)
    assert box.backend.isolated == 1


async def _none():
    return None
