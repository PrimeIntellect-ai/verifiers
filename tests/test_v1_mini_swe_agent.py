from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import verifiers.v1 as vf


def write_harbor_task(root: Path) -> Path:
    task_dir = root / "task-a"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Fix the bug.\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[environment]
docker_image = "python:3.11-slim"
cpus = 2
memory = "4G"
storage = "8G"

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 300
""".strip()
    )
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    (task_dir / "solution" / "solve.sh").write_text("true")
    return task_dir


def test_mini_swe_agent_builds_sandbox_program():
    harness = vf.MiniSWEAgent(system_prompt="Use tests.", agent_workdir="/app")
    program = cast(dict[str, Any], harness.program)

    assert isinstance(harness, vf.CLIHarness)
    assert program["sandbox"] is not False
    assert "OPENAI_MODEL" in cast(dict[str, object], program["env"])
    assert "/mini-swe-agent/prompt.txt" in cast(dict[str, object], program["files"])
    assert "/mini-swe-agent/system.txt" in cast(dict[str, object], program["files"])
    assert "mini_swe_agent_log" in cast(dict[str, object], program["artifacts"])


def test_mini_swe_agent_composes_with_harbor_taskset(tmp_path: Path):
    write_harbor_task(tmp_path)

    env = vf.Env(taskset=vf.HarborTaskset(tasks=tmp_path), harness=vf.MiniSWEAgent())
    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)

    assert isinstance(env.harness, vf.MiniSWEAgent)
    assert task["taskset_id"] == "harbor"
    assert task["instruction"] == "Fix the bug."


def test_mini_swe_agent_is_reexported():
    from verifiers.v1.packages.harnesses import MiniSWEAgent

    assert vf.MiniSWEAgent is MiniSWEAgent
