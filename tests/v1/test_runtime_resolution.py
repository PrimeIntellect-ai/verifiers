from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.task import Task, TaskData
from verifiers.v1.utils.compile import resolve_runtime_config


def test_explicit_runtime_image_wins_over_task_image() -> None:
    task = Task(TaskData(image="canonical-prime-image"))
    resolved = resolve_runtime_config(
        DockerConfig(image="local-development-image", privileged=True), task
    )

    assert resolved.image == "local-development-image"
    assert resolved.privileged is True


def test_task_image_fills_default_runtime_image() -> None:
    task = Task(TaskData(image="canonical-task-image"))

    resolved = resolve_runtime_config(DockerConfig(), task)

    assert resolved.image == "canonical-task-image"
