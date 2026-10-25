"""A task that starts from a checkpoint: its solver box and an isolated grader's fresh
box restore it instead of pulling an image."""

from types import SimpleNamespace

import pytest

from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.grading import verifier_runtime
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, restore_config
from verifiers.v1.tasksets.harbor import taskset as harbor
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.utils.compile import resolve_runtime_config

POLICY = ScoringConfig(mode="isolated")


def task(**data) -> HarborTask:
    return HarborTask(HarborData(idx=0, prompt="p", **data))


def test_the_solver_box_starts_from_the_tasks_checkpoint():
    config = resolve_runtime_config(
        PrimeConfig(cpu=4), task(checkpoint="ck-world", image="world:1")
    )
    assert (config.checkpoint, config.cpu) == ("ck-world", 4)
    with pytest.raises(ValueError, match="cannot start from a checkpoint"):
        resolve_runtime_config(DockerConfig(), task(checkpoint="ck-world"))


def test_a_resumed_run_starts_from_its_own_later_checkpoint():
    config = resolve_runtime_config(PrimeConfig(), task(checkpoint="ck-world"))
    assert restore_config(config, "ck-turn-9").checkpoint == "ck-turn-9"


def test_the_grader_starts_from_the_tasks_checkpoint_not_the_solvers():
    # The solver box was restored from a later checkpoint of its own run.
    solver = SimpleNamespace(config=PrimeConfig(checkpoint="ck-turn-9", workdir="/app"))
    base = PrimeConfig(checkpoint="ck-turn-9")
    config = verifier_runtime(task(checkpoint="ck-world"), solver, base, POLICY)
    assert config.checkpoint == "ck-world"
    plain = verifier_runtime(task(image="world:1"), solver, base, POLICY)
    assert plain.checkpoint is None


def _harbor_task(tmp_path, monkeypatch, environment: str, verifier: str = ""):
    task_dir = tmp_path / "from-checkpoint"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests/test.sh").write_text("exit 0\n")
    (task_dir / "instruction.md").write_text("Do it.")
    (task_dir / "task.toml").write_text(f"[environment]\n{environment}{verifier}")
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)
    return task_dir


def test_task_toml_names_the_checkpoint(tmp_path, monkeypatch):
    _harbor_task(
        tmp_path,
        monkeypatch,
        'checkpoint_id = "ck-world"\ncpus = 2\n',
        '[verifier]\nenvironment_mode = "separate"\n',
    )
    [loaded] = list(harbor.HarborTaskset(harbor.HarborConfig()))
    assert loaded.data.checkpoint == "ck-world"
    assert loaded.data.image is None
    grading = harbor.verifier_box_data(loaded.data)
    assert grading.checkpoint == "ck-world"
    own_image = harbor.verifier_box_data(
        loaded.data.model_copy(update={"verifier_image": "grader:1"})
    )
    assert own_image.checkpoint is None


def test_a_compose_task_cannot_name_a_checkpoint(tmp_path, monkeypatch):
    task_dir = _harbor_task(tmp_path, monkeypatch, 'checkpoint_id = "ck-world"\n')
    (task_dir / "environment").mkdir()
    (task_dir / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    image: python:3.11-slim\n"
    )
    with pytest.raises(ValueError, match="checkpoint_id"):
        list(harbor.HarborTaskset(harbor.HarborConfig()))


def test_a_checkpoint_task_needs_no_image_beside_its_dockerfile(tmp_path, monkeypatch):
    task_dir = _harbor_task(tmp_path, monkeypatch, 'checkpoint_id = "ck-world"\n')
    (task_dir / "environment").mkdir()
    (task_dir / "environment/Dockerfile").write_text("FROM python:3.11-slim\n")
    [loaded] = list(harbor.HarborTaskset(harbor.HarborConfig()))
    assert (loaded.data.checkpoint, loaded.data.image) == ("ck-world", None)
    assert not loaded.data.upload_environment
    # Without a checkpoint the same task still needs a pullable image.
    (task_dir / "task.toml").write_text("[environment]\ncpus = 2\n")
    with pytest.raises(ValueError, match="pullable docker_image"):
        list(harbor.HarborTaskset(harbor.HarborConfig()))
