from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from pathlib import Path

import pytest
from datasets import Dataset

import verifiers.v1 as vf
from environments.rlm_swe_v1 import rlm_swe_v1


def as_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def test_rlm_harness_builds_sandbox_program_without_eager_checkout():
    harness = vf.RLM(local_checkout="/tmp/does-not-need-to-exist-yet")
    program = as_mapping(harness.program)
    program_env = as_mapping(program["env"])
    artifacts = as_mapping(program["artifacts"])
    setup = program["setup"]

    assert isinstance(harness, vf.CLIHarness)
    assert program["sandbox"] is not False
    assert isinstance(setup, list)
    assert "apt-get -o Acquire::Retries=3 update" in setup[0]
    assert "apt-get -o Acquire::Retries=3 install" in setup[0]
    assert "RLM_MODEL" in program_env
    assert "rlm_metrics" in artifacts


def test_rlm_harness_can_upload_skills(tmp_path: Path):
    skills = tmp_path / "skills"
    (skills / "edit").mkdir(parents=True)
    (skills / "edit" / "SKILL.md").write_text("---\nname: edit\n---\n")

    harness = vf.RLM(local_checkout="/tmp/checkout", skills=skills)
    program = as_mapping(harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == skills


def test_rlm_harness_uploads_taskset_skills_by_default(tmp_path: Path):
    skills = tmp_path / "taskset-skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: taskset\n---\n")

    class SkillTaskset(vf.Taskset):
        def get_upload_dirs(self):
            return {"skills": skills}

    env = vf.Env(
        taskset=SkillTaskset(source=[]),
        harness=vf.RLM(local_checkout="/tmp/checkout"),
    )
    program = as_mapping(env.harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == skills


def test_taskset_discovers_sibling_skills_dir_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "skill_taskset_module"
    module_file = tmp_path / f"{module_name}.py"
    skills = tmp_path / "skills"
    module_file.write_text("")
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: sibling\n---\n")
    module = types.ModuleType(module_name)
    module.__file__ = str(module_file)
    module.__package__ = ""
    monkeypatch.setitem(sys.modules, module_name, module)
    skill_taskset_type = type(
        "SkillTaskset", (vf.Taskset,), {"__module__": module_name}
    )

    taskset = skill_taskset_type(source=[])

    assert taskset.get_upload_dirs() == {"skills": skills}


def test_rlm_harness_explicit_skills_override_taskset_skills(tmp_path: Path):
    taskset_skills = tmp_path / "taskset-skills"
    explicit_skills = tmp_path / "explicit-skills"
    taskset_skills.mkdir()
    explicit_skills.mkdir()

    class SkillTaskset(vf.Taskset):
        def get_upload_dirs(self):
            return {"skills": taskset_skills}

    env = vf.Env(
        taskset=SkillTaskset(source=[]),
        harness=vf.RLM(local_checkout="/tmp/checkout", skills=explicit_skills),
    )
    program = as_mapping(env.harness.program)
    dirs = as_mapping(program["dirs"])

    assert dirs["/rlm/skills"] == explicit_skills


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
    program = as_mapping(env.harness.program)
    program_env = as_mapping(program["env"])

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
    assert program_env["PATH"] == "/caller/bin"
    assert program_env["CUSTOM"] == "1"
    assert program_env["CALLER"] == "1"
    assert program_env["PAGER"] == "cat"
    assert program_env["RLM_TOOLS"] == "bash,edit"


def test_rlm_swe_taskset_hooks_are_registered_with_runtime():
    taskset = rlm_swe_v1.load_taskset()
    env = vf.Env(taskset=taskset)

    setup_names = [handler.__name__ for handler in env.harness.runtime.rollout_setup]
    cleanup_names = [
        handler.__name__ for handler in env.harness.runtime.rollout_cleanup
    ]
    signal_names = {signal["name"] for signal in env.harness.runtime.rollout_signals}

    assert setup_names.count("setup_r2e_sandbox") == 1
    assert cleanup_names.count("cleanup_r2e_state") == 1
    assert "solved" in signal_names


@pytest.mark.asyncio
async def test_rlm_swe_taskset_setup_and_reward(monkeypatch):
    monkeypatch.setattr(
        rlm_swe_v1, "load_dataset", lambda *args, **kwargs: fake_r2e_dataset()
    )
    taskset = rlm_swe_v1.load_taskset(timeout_minutes=30)
    task = next(iter(taskset))
    state = vf.State.for_task(task)
    sandbox = FakeSandbox()
    calls: dict[str, object] = {}

    async def fake_setup_sandbox(sandbox_arg: object, state_arg: vf.State) -> None:
        calls["setup_sandbox"] = sandbox_arg
        calls["setup_state"] = state_arg

    async def fake_run_tests(
        sandbox_arg: object,
        state_arg: vf.State,
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


@pytest.mark.asyncio
async def test_rlm_swe_run_tests_quotes_env_values():
    taskset = rlm_swe_v1.load_taskset(
        hide_tests_from_agent=False,
        env={"SAFE": "two words; $(echo nope)", "QUOTE": "it's ok"},
    )
    sandbox = RecordingSandbox()

    output = await taskset.run_tests(sandbox, {}, 123)

    assert output == "test output"
    assert len(sandbox.background_jobs) == 1
    command = sandbox.background_jobs[0]["command"]
    assert "SAFE='two words; $(echo nope)'" in command
    assert "QUOTE='it'\"'\"'s ok'" in command
    assert command.endswith("/bin/bash run_tests.sh > test_output.txt 2>&1")


def test_rlm_swe_get_env_vars_uses_configured_repo_path():
    taskset = rlm_swe_v1.load_taskset(repo_path="/workspace/repo")

    path = taskset.get_env_vars()["PATH"]

    assert "/workspace/repo/.venv/bin" in path
    assert "/testbed/.venv/bin" not in path


def test_rlm_swe_reward_rejects_pytest_summary_without_nodeid():
    taskset = rlm_swe_v1.load_taskset()
    test_output = """
=========================== short test summary info ============================
PASSED tests/test_example.py
"""

    reward = taskset.calculate_reward(
        test_output,
        {"expected_output_json": '{"test_fix": "PASSED"}'},
    )
    parsed = rlm_swe_v1.parse_log_pytest(test_output)

    assert reward == 0.0
    assert "" not in parsed


def test_rlm_swe_parse_log_pytest_uses_leading_status_token():
    test_output = """
=========================== short test summary info ============================
FAILED tests/test_PASSED_handler.py::test_fix - AssertionError
PASSED tests/test_failed_handler.py::test_other
ERROR tests/test_failed_handler.py::test_error - setup failed
"""

    parsed = rlm_swe_v1.parse_log_pytest(test_output)

    assert parsed == {
        "test_fix": "FAILED",
        "test_other": "PASSED",
        "test_error": "ERROR",
    }


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


class FakeCommandResult:
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class RecordingSandbox:
    def __init__(self):
        self.background_jobs: list[dict[str, object]] = []
        self.commands: list[dict[str, object]] = []

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> FakeCommandResult:
        self.background_jobs.append(
            {
                "command": command,
                "timeout": timeout,
                "working_dir": working_dir,
            }
        )
        return FakeCommandResult()

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> FakeCommandResult:
        self.commands.append(
            {
                "command": command,
                "timeout": timeout,
                "working_dir": working_dir,
            }
        )
        return FakeCommandResult(stdout="test output")
