from pathlib import Path

import pytest

from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTaskset,
    parse_task,
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.commands: list[list[str]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append(argv)
        return ProgramResult(exit_code=0, stdout="", stderr="")


def write_task_toml(task_dir: Path, content: str) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(content.strip() + "\n")


@pytest.mark.asyncio
async def test_harbor_setup_extracts_workspace_overlays(tmp_path: Path):
    task_dir = tmp_path / "task"
    write_task_toml(
        task_dir,
        """
        [environment]
        workdir = "/workspace"
        """,
    )
    (task_dir / "instruction.md").write_text("Check restored artifacts.")
    overlay = tmp_path / "overlay"
    (overlay / "workspace" / "examples" / "demo").mkdir(parents=True)
    (overlay / "workspace" / "examples" / "demo" / "main.py").write_text("print(1)")
    config = HarborConfig(
        id="harbor",
        dataset=task_dir.as_posix(),
        workspace_overlays=[{"source": overlay.as_posix()}],
    )
    taskset = HarborTaskset(config)
    task = parse_task(task_dir, 0, config)
    runtime = RecordingRuntime()

    await taskset.setup(task, runtime)

    assert "/tmp/harbor_setup.tgz" in runtime.writes
    assert runtime.commands == [
        [
            "sh",
            "-c",
            "mkdir -p / && tar -xzf /tmp/harbor_setup.tgz -C /",
        ]
    ]


@pytest.mark.asyncio
async def test_harbor_setup_runs_solution_entrypoints_in_task_workdir(tmp_path: Path):
    task_dir = tmp_path / "task"
    write_task_toml(
        task_dir,
        """
        [environment]
        workdir = "/workspace"

        [agent]
        timeout_sec = 12
        """,
    )
    (task_dir / "instruction.md").write_text("Run the reference solution.")
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/usr/bin/env bash\n")
    config = HarborConfig(
        id="harbor",
        dataset=task_dir.as_posix(),
        run_solutions=True,
    )
    taskset = HarborTaskset(config)
    task = parse_task(task_dir, 0, config)
    runtime = RecordingRuntime()

    await taskset.setup(task, runtime)

    assert runtime.commands[-1] == [
        "sh",
        "-c",
        "cd /workspace && timeout 12.0s bash /oracle/solve.sh",
    ]


@pytest.mark.asyncio
async def test_harbor_setup_extracts_and_runs_step_workdir_setup(tmp_path: Path):
    task_dir = tmp_path / "task"
    write_task_toml(
        task_dir,
        """
        [environment]
        workdir = "/workspace/examples"

        [[steps]]
        name = "01_step"
        """,
    )
    step_dir = task_dir / "steps" / "01_step"
    (step_dir / "workdir").mkdir(parents=True)
    (step_dir / "workdir" / "setup.sh").write_text("#!/usr/bin/env bash\n")
    (step_dir / "instruction.md").write_text("Use the workdir setup.")
    config = HarborConfig(id="harbor", dataset=task_dir.as_posix())
    taskset = HarborTaskset(config)
    task = parse_task(task_dir, 0, config)
    runtime = RecordingRuntime()

    await taskset.setup(task, runtime)

    assert runtime.commands == [
        [
            "sh",
            "-c",
            "mkdir -p /workspace/examples && tar -xzf /tmp/harbor_setup.tgz -C /",
        ],
        ["sh", "-c", "cd /workspace/examples && bash setup.sh"],
    ]
