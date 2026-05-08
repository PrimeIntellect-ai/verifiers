from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from datasets import Dataset

import verifiers.v1 as vf
from environments.rlm_swe_v1 import rlm_swe_v1


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


def test_rlm_swe_environment_uses_v1_r2e_taskset(monkeypatch):
    calls: dict[str, object] = {}

    def fake_load_dataset(dataset_name: str, **kwargs: object) -> Dataset:
        calls["dataset_name"] = dataset_name
        calls["kwargs"] = kwargs
        return fake_r2e_dataset()

    monkeypatch.setattr(rlm_swe_v1, "load_dataset", fake_load_dataset)

    env = rlm_swe_v1.load_environment(
        dataset_name="fake-r2e",
        local_checkout="/tmp/checkout",
        timeout_minutes=30,
        env={"CUSTOM": "1", "PATH": "/task/bin"},
        rlm_env={"CALLER": "1", "PATH": "/caller/bin"},
    )
    task = next(iter(env.taskset))
    program = cast(dict[str, Any], env.harness.program)

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, rlm_swe_v1.R2ESWETaskset)
    assert isinstance(env.harness, vf.RLM)
    assert calls["dataset_name"] == "fake-r2e"
    assert task["taskset_id"] == "swe/r2e"
    assert task["instruction"] == "Fix repo-0."
    assert task["sandbox"]["image"] == (
        f"{rlm_swe_v1.REGISTRY_PREFIX}/r2e/image:latest"
    )
    assert task["sandbox"]["workdir"] == "/testbed"
    assert task["sandbox"]["timeout_minutes"] == 30
    assert task["program"]["env"] == {"AGENT_WORKDIR": "/testbed"}
    assert program["env"]["PATH"] == "/task/bin"
    assert program["env"]["CUSTOM"] == "1"
    assert program["env"]["CALLER"] == "1"


async def test_rlm_swe_taskset_setup_and_reward(monkeypatch):
    monkeypatch.setattr(
        rlm_swe_v1, "load_dataset", lambda *args, **kwargs: fake_r2e_dataset()
    )
    taskset = rlm_swe_v1.load_taskset(timeout_minutes=30)
    task = next(iter(taskset))
    state = vf.State.for_task(task)
    sandbox = FakeSandbox()
    calls: dict[str, object] = {}

    async def fake_setup_sandbox(
        sandbox_arg: object, state_arg: dict[str, Any]
    ) -> None:
        calls["setup_sandbox"] = sandbox_arg
        calls["setup_state"] = state_arg

    async def fake_run_tests(
        sandbox_arg: object,
        state_arg: dict[str, Any],
        test_timeout: int,
    ) -> str:
        calls["run_tests"] = (sandbox_arg, state_arg, test_timeout)
        return """
=========================== short test summary info ============================
PASSED tests/test_example.py::test_fix
"""

    monkeypatch.setattr(taskset, "setup_sandbox", fake_setup_sandbox)
    monkeypatch.setattr(taskset, "run_tests", fake_run_tests)

    await taskset.setup_r2e_sandbox(task, state, sandbox=sandbox)
    reward = await taskset.solved(task, state)
    await taskset.cleanup_r2e_state(task, state)

    assert calls["setup_sandbox"] is sandbox
    assert calls["setup_state"] is state
    assert calls["run_tests"] == (sandbox, state, 1800)
    assert state["sandbox_id"] == "sandbox-1"
    assert state["test_timeout"] == 1800
    assert reward == 1.0
    assert "sandbox_client" not in state
    assert "_rlm_swe_sandbox" not in state


def fake_r2e_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "commit_hash": f"commit-{index}",
                "repo_name": "example/repo",
                "problem_statement": f"Fix repo-{index}.",
                "docker_image": "r2e/image:latest",
                "expected_output_json": '{"test_fix": "PASSED"}',
                "parsed_commit_content": '{"file_diffs": []}',
            }
            for index in range(12)
        ]
    )


class FakeLease:
    client = object()


class FakeSandbox:
    id = "sandbox-1"
    lease = FakeLease()
