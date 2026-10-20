"""Which network a parsed Harbor task's shared-box verifier grades under."""

import pytest

pytest.importorskip("harbor")

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, parse_task


def task(tmp_path, *, environment=None, agent=None, verifier=None):
    def section(name, mode):
        return f"[{name}]\nnetwork_mode = {mode!r}\n" if mode else f"[{name}]\n"

    (tmp_path / "instruction.md").write_text("Do nothing.")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (tmp_path / "task.toml").write_text(
        'version = "1.0"\n'
        + section("environment", environment).replace(
            "[environment]\n", '[environment]\ndocker_image = "python:3.11-slim"\n'
        )
        + section("agent", agent)
        + section("verifier", verifier)
    )
    return parse_task(tmp_path, 0, HarborConfig())


def test_an_offline_task_grades_offline(tmp_path):
    data = task(
        tmp_path, environment="no-network", agent="no-network", verifier="no-network"
    )
    assert data.scoring.mode == "shared"
    assert data.scoring.network == vf.NetworkPolicyConfig(allow=[])


def test_the_environment_baseline_applies_without_a_verifier_policy(tmp_path):
    data = task(tmp_path, environment="no-network")
    assert data.scoring.network == vf.NetworkPolicyConfig(allow=[])


def test_a_public_verifier_opens_the_box(tmp_path):
    assert task(tmp_path, agent="no-network").scoring.network is None
    assert task(tmp_path, agent="no-network", verifier="public").scoring.network is None


def test_a_restricted_verifier_beside_a_public_agent_grades_open(tmp_path, caplog):
    assert task(tmp_path, verifier="no-network").scoring.network is None
    assert "cannot restrict a box that started open" in caplog.text
