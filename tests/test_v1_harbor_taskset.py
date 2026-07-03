"""Unit tests for the v1 harbor taskset's [verifier.env] handling."""

import pytest

from verifiers.v1.tasksets.harbor.taskset import HarborConfig, parse_task, verifier_env

TASK_TOML = """
version = "1.0"

[environment]
docker_image = "ubuntu:24.04"
cpus = 1

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 300

[verifier.env]
JUDGE_API_KEY = "${TEST_HARBOR_JUDGE_KEY:-}"
REPO_NAME = "demo"
"""


def write_task(root, task_toml: str = TASK_TOML):
    (root / "tests").mkdir()
    (root / "instruction.md").write_text("do the thing")
    (root / "task.toml").write_text(task_toml)
    return root


def test_parse_task_reads_verifier_env(tmp_path):
    task = parse_task(write_task(tmp_path), 0, HarborConfig())
    assert task.verifier_env == {
        "JUDGE_API_KEY": "${TEST_HARBOR_JUDGE_KEY:-}",
        "REPO_NAME": "demo",
    }


def test_parse_task_without_verifier_env(tmp_path):
    toml = TASK_TOML.split("[verifier.env]")[0]
    task = parse_task(write_task(tmp_path, toml), 0, HarborConfig())
    assert task.verifier_env == {}
    assert verifier_env(task) == {}  # no harbor import needed for the empty case


def test_verifier_env_resolves_from_host(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    monkeypatch.setenv("TEST_HARBOR_JUDGE_KEY", "sk-123")
    task = parse_task(write_task(tmp_path), 0, HarborConfig())
    assert verifier_env(task) == {"JUDGE_API_KEY": "sk-123", "REPO_NAME": "demo"}
    # resolution never mutates the serialized templates
    assert task.verifier_env["JUDGE_API_KEY"] == "${TEST_HARBOR_JUDGE_KEY:-}"


def test_verifier_env_falls_back_to_default(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    monkeypatch.delenv("TEST_HARBOR_JUDGE_KEY", raising=False)
    task = parse_task(write_task(tmp_path), 0, HarborConfig())
    assert verifier_env(task) == {"JUDGE_API_KEY": "", "REPO_NAME": "demo"}
