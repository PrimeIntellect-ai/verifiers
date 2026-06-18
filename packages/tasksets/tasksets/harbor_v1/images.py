"""Resolve Harbor Dockerfiles to images in the Prime registry."""

import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

from prime_sandboxes import APIClient

IMAGE_TAG = "latest"
IMAGE_SEARCH_LIMIT = 10_000
IMAGE_BUILD_TIMEOUT = 600.0
IMAGE_POLL_INTERVAL = 5.0
ACTIVE_STATUSES = frozenset({"PENDING", "BUILDING", "UPLOADING"})

ImageRow = dict[str, object]


def prime_image_name(dataset: str, task_dir: Path) -> str:
    """Return the stable Prime image name for a Harbor dataset task."""
    source = f"{dataset}-{task_dir.name}".lower()
    image_name = re.sub(r"[^a-z0-9._-]+", "-", source).strip("._-")
    if not image_name:
        raise ValueError(f"cannot derive a Prime image name for {task_dir}")
    return f"{image_name}:{IMAGE_TAG}"


def _find_prime_images(client: APIClient, targets: set[str]) -> dict[str, ImageRow]:
    """Find the best available container-image row for each target."""
    names = sorted(image.rsplit(":", 1)[0] for image in targets)
    params: dict[str, str | int] = {
        "limit": IMAGE_SEARCH_LIMIT,
        "search": os.path.commonprefix(names),
    }
    if client.config.team_id:
        params["teamId"] = client.config.team_id

    found: dict[str, ImageRow] = {}
    for row in client.request("GET", "/images", params=params)["data"]:
        image = f"{row.get('imageName')}:{row.get('imageTag')}"
        if image not in targets or row.get("artifactType") != "CONTAINER_IMAGE":
            continue
        current = found.get(image)
        if current and (
            current.get("status") == "COMPLETED"
            or row.get("status") != "COMPLETED"
            and current.get("status") in ACTIVE_STATUSES
        ):
            continue
        found[image] = row

    return found


def _create_prime_image(image: str, task_dir: Path) -> None:
    """Have the Prime CLI build the task's unchanged environment context."""
    environment = task_dir / "environment"
    print(f"Creating Prime image {image} for Harbor task {task_dir.name}...")
    subprocess.run(
        [
            "prime",
            "images",
            "--plain",
            "push",
            image,
            "--context",
            str(environment),
        ],
        check=True,
    )


def ensure_prime_images(dataset: str, task_dirs: list[Path]) -> dict[Path, str]:
    """Reuse or build Prime images for tasks that only declare a Dockerfile."""
    targets: dict[Path, str] = {}
    build_dirs: dict[str, Path] = {}
    timeouts: dict[str, float] = {}
    for task_dir in task_dirs:
        config = tomllib.loads((task_dir / "task.toml").read_text())
        environment = config.get("environment", {})
        if environment.get("docker_image"):
            continue
        if not (task_dir / "environment" / "Dockerfile").is_file():
            continue
        image = prime_image_name(dataset, task_dir)
        targets[task_dir] = image
        build_dirs.setdefault(image, task_dir)
        timeouts[image] = float(
            environment.get("build_timeout_sec", IMAGE_BUILD_TIMEOUT)
        )

    if not targets:
        return {}

    client = APIClient()
    rows = _find_prime_images(client, set(targets.values()))
    resolved: dict[str, str] = {}
    pending: set[str] = set()
    deadlines: dict[str, float] = {}

    for image, task_dir in build_dirs.items():
        row = rows.get(image)
        status = row.get("status") if row else None
        if row and status == "COMPLETED":
            resolved[image] = str(row.get("displayRef") or row["fullImagePath"])
            continue
        if status not in ACTIVE_STATUSES:
            _create_prime_image(image, task_dir)
        pending.add(image)
        deadlines[image] = time.monotonic() + timeouts[image]

    while pending:
        time.sleep(IMAGE_POLL_INTERVAL)
        rows = _find_prime_images(client, pending)
        now = time.monotonic()
        for image in list(pending):
            row = rows.get(image)
            if not row:
                continue
            status = row.get("status")
            if status == "COMPLETED":
                resolved[image] = str(row.get("displayRef") or row["fullImagePath"])
                pending.remove(image)
                print(f"Prime image {image} is ready.")
                continue
            if status not in ACTIVE_STATUSES:
                detail = row.get("errorMessage") or status
                raise RuntimeError(f"Prime image {image} failed to build: {detail}")

        timed_out = [image for image in pending if now >= deadlines[image]]
        if timed_out:
            images = ", ".join(sorted(timed_out))
            raise RuntimeError(f"timed out waiting for Prime images: {images}")

    return {task_dir: resolved[image] for task_dir, image in targets.items()}
