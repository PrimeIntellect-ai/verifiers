from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import verifiers.v1 as vf
from environments.rlm_swe_v1.rlm_swe_v1 import load_environment


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


def test_rlm_harness_builds_sandbox_program_without_eager_checkout():
    harness = vf.RLM(local_checkout="/tmp/does-not-need-to-exist-yet")
    program = cast(dict[str, Any], harness.program)

    assert isinstance(harness, vf.CLIHarness)
    assert program["sandbox"] is not False
    assert "RLM_MODEL" in cast(dict[str, object], program["env"])
    assert "rlm_metrics" in cast(dict[str, object], program["artifacts"])


def test_rlm_harness_can_upload_skills(tmp_path: Path):
    skills = tmp_path / "skills"
    (skills / "edit").mkdir(parents=True)
    (skills / "edit" / "SKILL.md").write_text("---\nname: edit\n---\n")

    harness = vf.RLM(local_checkout="/tmp/checkout", skills=skills)
    program = cast(dict[str, Any], harness.program)

    assert cast(dict[str, object], program["dirs"])["/rlm/skills"] == skills


def test_rlm_swe_environment_uses_harbor_taskset(tmp_path: Path):
    write_harbor_task(tmp_path)

    env = load_environment(tasks=tmp_path, local_checkout="/tmp/checkout")
    task = next(iter(env.taskset))

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.HarborTaskset)
    assert isinstance(env.harness, vf.RLM)
    assert task["taskset_id"] == "harbor"
    assert task["instruction"] == "Fix the bug."
    assert task["sandbox"]["image"] == "python:3.11-slim"
    assert task["program"]["env"]["AGENT_WORKDIR"] == "/app"
