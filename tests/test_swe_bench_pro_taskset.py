from __future__ import annotations

import time
import urllib.request

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
