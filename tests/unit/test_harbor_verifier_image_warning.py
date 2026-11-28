"""When `HarborTaskset` warns about an unbuilt separate-verifier Dockerfile."""

import logging
from collections.abc import Iterator

import pytest

pytest.importorskip("harbor")

from verifiers.v1.tasksets.harbor import taskset as harbor
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTask,
    HarborTaskset,
)


def make_task(tmp_path, *, tests_file: str | None = "Dockerfile"):
    task_dir = tmp_path / "sample-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    if tests_file is not None:
        (task_dir / "tests" / tests_file).write_text("FROM python:3.12-slim\n")
    (task_dir / "instruction.md").write_text("Solve it.")
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        "[verifier]\n"
        'environment_mode = "separate"\n'
        "[environment]\n"
        "cpus = 1\n"
    )
    return task_dir


def test_no_warning_when_subclass_sets_verifier_image(tmp_path, monkeypatch, caplog):
    make_task(tmp_path, tests_file="Dockerfile")
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)

    class PrebuiltTaskset(HarborTaskset):
        def load(self) -> Iterator[HarborTask]:
            for task in super().load():
                yield task.with_data(
                    image="registry.example.com/agent:latest",
                    verifier_image="registry.example.com/verifier:latest",
                )

    with caplog.at_level(logging.WARNING, logger="verifiers.v1.tasksets.harbor"):
        [task] = list(PrebuiltTaskset(HarborConfig(ignore_dockerfile=True)))

    assert task.data.verifier_image == "registry.example.com/verifier:latest"
    assert "names no docker_image" not in caplog.text


@pytest.mark.parametrize("tests_file", ["Dockerfile", "docker-compose.yaml"])
def test_warns_when_unbuilt_verifier_falls_back_to_agent_image(
    tmp_path, monkeypatch, caplog, tests_file
):
    make_task(tmp_path, tests_file=tests_file)
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)

    with caplog.at_level(logging.WARNING, logger="verifiers.v1.tasksets.harbor"):
        [task] = list(HarborTaskset(HarborConfig(ignore_dockerfile=True)))

    assert task.data.verifier_image is None
    assert f"building tests/{tests_file}" in caplog.text


def test_no_warning_for_fresh_copy_separate_verifier(tmp_path, monkeypatch, caplog):
    make_task(tmp_path, tests_file=None)
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)

    with caplog.at_level(logging.WARNING, logger="verifiers.v1.tasksets.harbor"):
        [task] = list(HarborTaskset(HarborConfig(ignore_dockerfile=True)))

    assert task.data.verifier is not None
    assert task.data.verifier_image is None
    assert "names no docker_image" not in caplog.text
