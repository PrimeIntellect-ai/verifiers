from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tasksets.harbor_v1 import images
from tasksets.harbor_v1 import taskset as harbor


def write_task(
    root: Path,
    name: str,
    *,
    dockerfile: bool = True,
    docker_image: str | None = None,
) -> Path:
    task_dir = root / name
    task_dir.mkdir(parents=True)
    environment = task_dir / "environment"
    environment.mkdir()
    config = [
        "[task]",
        f'name = "{name}"',
        "",
        "[environment]",
        "build_timeout_sec = 30",
    ]
    if docker_image:
        config.append(f'docker_image = "{docker_image}"')
    (task_dir / "task.toml").write_text("\n".join(config))
    (task_dir / "instruction.md").write_text("Fix it.")
    if dockerfile:
        (environment / "Dockerfile").write_text("FROM alpine:latest\n")
    return task_dir


@pytest.mark.parametrize(
    ("dataset", "task_name", "expected"),
    [
        (
            "openthoughts/openthoughts-tblite",
            "broken-python",
            "openthoughts-openthoughts-tblite-broken-python:latest",
        ),
        ("hello-world@1.0", "hello-world", "hello-world-1.0-hello-world:latest"),
    ],
)
def test_prime_image_name(dataset: str, task_name: str, expected: str) -> None:
    assert images.prime_image_name(dataset, Path(task_name)) == expected


def test_find_prime_images_prefers_completed_row() -> None:
    target = "dataset-task:latest"
    calls = []

    class Client:
        config = SimpleNamespace(team_id="team-example")

        def request(self, method, endpoint, params=None):
            calls.append((method, endpoint, params))
            return {
                "data": [
                    {
                        "imageName": "dataset-task",
                        "imageTag": "latest",
                        "artifactType": "CONTAINER_IMAGE",
                        "status": "FAILED",
                    },
                    {
                        "imageName": "dataset-task",
                        "imageTag": "latest",
                        "artifactType": "CONTAINER_IMAGE",
                        "status": "COMPLETED",
                    },
                ]
            }

    client = cast(images.APIClient, Client())
    assert images._find_prime_images(client, {target})[target]["status"] == "COMPLETED"
    assert calls == [
        (
            "GET",
            "/images",
            {
                "limit": 10_000,
                "search": "dataset-task",
                "teamId": "team-example",
            },
        )
    ]


def test_ensure_prime_images_reuses_completed_image(tmp_path, monkeypatch) -> None:
    task_dir = write_task(tmp_path, "broken-python")
    image = "openthoughts-openthoughts-tblite-broken-python:latest"
    display_ref = f"team-example/{image}"
    monkeypatch.setattr(images, "APIClient", object)
    monkeypatch.setattr(
        images,
        "_find_prime_images",
        lambda client, targets: {
            image: {
                "id": "existing",
                "status": "COMPLETED",
                "displayRef": display_ref,
            }
        },
    )
    monkeypatch.setattr(
        images,
        "_create_prime_image",
        lambda image, task_dir: pytest.fail("completed images must not be rebuilt"),
    )

    resolved = images.ensure_prime_images(
        "openthoughts/openthoughts-tblite", [task_dir]
    )

    assert resolved == {task_dir: display_ref}


def test_ensure_prime_images_ignores_declared_images(tmp_path, monkeypatch) -> None:
    task_dir = write_task(
        tmp_path, "broken-python", docker_image="example/broken-python:v1"
    )
    monkeypatch.setattr(
        images,
        "APIClient",
        lambda: pytest.fail("declared images must not query Prime"),
    )

    assert images.ensure_prime_images("org/dataset", [task_dir]) == {}


def test_ensure_prime_images_pushes_missing_image(tmp_path, monkeypatch) -> None:
    task_dir = write_task(tmp_path, "broken-python")
    image = "openthoughts-openthoughts-tblite-broken-python:latest"
    display_ref = f"team-example/{image}"
    image_lists = iter(
        [
            {},
            {
                image: {
                    "id": "new-build",
                    "status": "COMPLETED",
                    "displayRef": display_ref,
                }
            },
        ]
    )
    pushed = []
    monkeypatch.setattr(images, "APIClient", object)
    monkeypatch.setattr(
        images, "_find_prime_images", lambda client, targets: next(image_lists)
    )
    monkeypatch.setattr(
        images,
        "_create_prime_image",
        lambda image, task_dir: pushed.append((image, task_dir)),
    )
    monkeypatch.setattr(images.time, "sleep", lambda seconds: None)

    resolved = images.ensure_prime_images(
        "openthoughts/openthoughts-tblite", [task_dir]
    )

    assert pushed == [(image, task_dir)]
    assert resolved == {task_dir: display_ref}


def test_ensure_prime_images_rebuilds_failed_image(tmp_path, monkeypatch) -> None:
    task_dir = write_task(tmp_path, "broken-python")
    image = "openthoughts-openthoughts-tblite-broken-python:latest"
    display_ref = f"team-example/{image}"
    image_lists = iter(
        [
            {image: {"id": "image-row", "status": "FAILED"}},
            {
                image: {
                    "id": "image-row",
                    "status": "COMPLETED",
                    "displayRef": display_ref,
                }
            },
        ]
    )
    pushed = []
    monkeypatch.setattr(images, "APIClient", object)
    monkeypatch.setattr(
        images, "_find_prime_images", lambda client, targets: next(image_lists)
    )
    monkeypatch.setattr(
        images,
        "_create_prime_image",
        lambda image, task_dir: pushed.append((image, task_dir)),
    )
    monkeypatch.setattr(images.time, "sleep", lambda seconds: None)

    resolved = images.ensure_prime_images(
        "openthoughts/openthoughts-tblite", [task_dir]
    )

    assert pushed == [(image, task_dir)]
    assert resolved == {task_dir: display_ref}


def test_create_prime_image_delegates_environment_context_to_cli(
    tmp_path, monkeypatch
) -> None:
    task_dir = write_task(tmp_path, "broken-python")
    calls = []

    monkeypatch.setattr(
        images.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    images._create_prime_image("dataset-broken-python:latest", task_dir)

    environment = task_dir / "environment"
    assert calls == [
        (
            [
                "prime",
                "images",
                "--plain",
                "push",
                "dataset-broken-python:latest",
                "--context",
                str(environment),
            ],
            {"check": True},
        )
    ]


def test_harbor_taskset_uses_resolved_prime_image(tmp_path, monkeypatch) -> None:
    task_dir = write_task(tmp_path, "broken-python")
    display_ref = "team-example/dataset-broken-python:latest"
    monkeypatch.setattr(harbor, "dataset_dir", lambda dataset: tmp_path)
    monkeypatch.setattr(
        harbor,
        "ensure_prime_images",
        lambda dataset, task_dirs: {task_dir: display_ref},
    )

    taskset = harbor.HarborTaskset(harbor.HarborConfig(dataset="org/dataset"))
    (task,) = taskset.load_tasks()

    assert task.image == display_ref
