from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pytest

from verifiers.envs.experimental.composable.tasksets.swe import swe_bench_pro
from verifiers.envs.experimental.composable.tasksets.swe.swe_bench_pro import (
    SWEBenchProTaskSet,
    _download_file,
)
from verifiers.envs.experimental.composable.tasksets.swe.swe_tasksets import (
    make_swe_taskset,
)


def test_make_swe_taskset_reaches_swebench_pro() -> None:
    taskset = make_swe_taskset("swebench-pro", max_examples=0)

    assert isinstance(taskset, SWEBenchProTaskSet)
    assert len(taskset._dataset) == 0


def test_instruction_preserves_curly_braces() -> None:
    taskset = SWEBenchProTaskSet.__new__(SWEBenchProTaskSet)

    instruction = taskset.get_instruction(
        {
            "problem_statement": "Fix code using ${HOME} and {'key': 'value'}",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "instance_id": "repo__issue-1",
        }
    )

    assert "${HOME}" in instruction
    assert "{'key': 'value'}" in instruction


def test_calculate_reward_handles_malformed_parser_output() -> None:
    taskset = SWEBenchProTaskSet.__new__(SWEBenchProTaskSet)

    reward = taskset._calculate_reward(
        "SWEBENCH_PRO_OUTPUT_START\n{bad json\nSWEBENCH_PRO_OUTPUT_END",
        {"fail_to_pass": "['test_a']", "pass_to_pass": "[]"},
    )

    assert reward == 0.0


def test_calculate_reward_handles_malformed_expected_tests() -> None:
    taskset = SWEBenchProTaskSet.__new__(SWEBenchProTaskSet)

    reward = taskset._calculate_reward(
        (
            "SWEBENCH_PRO_OUTPUT_START\n"
            '{"tests":[{"name":"test_a","status":"PASSED"}]}\n'
            "SWEBENCH_PRO_OUTPUT_END"
        ),
        {"fail_to_pass": "not a literal", "pass_to_pass": "[]"},
    )

    assert reward == 0.0


def test_calculate_reward_requires_all_expected_tests() -> None:
    taskset = SWEBenchProTaskSet.__new__(SWEBenchProTaskSet)

    reward = taskset._calculate_reward(
        (
            "SWEBENCH_PRO_OUTPUT_START\n"
            '{"tests":[{"name":"test_a","status":"PASSED"}]}\n'
            "SWEBENCH_PRO_OUTPUT_END"
        ),
        {"fail_to_pass": "['test_a']", "pass_to_pass": "['test_b']"},
    )

    assert reward == 0.0


def test_download_file_retries_normal_fetch_errors(monkeypatch, tmp_path) -> None:
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"script"

    def urlopen(url: str, timeout: int):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary failure")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    out_path = tmp_path / "run_script.sh"
    _download_file("https://example.test/run_script.sh", str(out_path))

    assert attempts == 3
    assert out_path.read_bytes() == b"script"


@pytest.mark.asyncio
async def test_run_tests_filters_destructive_before_repo_set_cmd(
    monkeypatch,
) -> None:
    class Result:
        def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0):
            self.stdout = stdout
            self.stderr = stderr
            self.exit_code = exit_code

    class SandboxClient:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.background_jobs: list[str] = []

        async def execute_command(self, sandbox_id, command, **kwargs):
            self.commands.append(command)
            if command == "cat /logs/verifier/scoring.log":
                return Result("SWEBENCH_PRO_OUTPUT_START\n{}\nSWEBENCH_PRO_OUTPUT_END")
            return Result()

        async def upload_file(self, sandbox_id, remote_path, local_path):
            assert Path(local_path).exists()

        async def run_background_job(self, sandbox_id, command, timeout):
            self.background_jobs.append(command)
            return Result()

    def fake_download(url: str, path: str) -> None:
        Path(path).write_text("script")

    monkeypatch.setattr(swe_bench_pro, "_download_file", fake_download)

    taskset = SWEBenchProTaskSet.__new__(SWEBenchProTaskSet)
    taskset.agent_workdir = "/app"
    taskset.run_scripts_url = "https://example.test/run_scripts"
    sandbox_client = SandboxClient()

    await taskset._run_tests(
        sandbox_client,
        "sandbox-id",
        {
            "info": {
                "base_commit": "base123",
                "instance_id": "instance-1",
                "selected_test_files_to_run": "['tests/foo.py']",
                "test_patch": "",
                "before_repo_set_cmd": (
                    "git reset --hard base123\n"
                    "git clean -fd\n"
                    "git checkout base123\n"
                    "git checkout target456 -- tests/foo.py"
                ),
            }
        },
        test_timeout=60,
    )

    checkout_commands = [
        command for command in sandbox_client.commands if "target456" in command
    ]
    assert checkout_commands == ["set -e\ngit checkout target456 -- tests/foo.py"]
    assert "git reset --hard base123" not in sandbox_client.background_jobs[0]
    assert "git clean -fd" not in sandbox_client.background_jobs[0]
