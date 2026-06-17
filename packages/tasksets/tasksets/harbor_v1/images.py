"""Resolve Harbor Dockerfiles to images in the Prime registry."""

import os
import re
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path

import httpx
from prime_sandboxes import APIClient, APIError

IMAGE_TAG = "latest"
IMAGE_PAGE_SIZE = 250
IMAGE_BUILD_TIMEOUT = 600.0
IMAGE_POLL_INTERVAL = 5.0
ACTIVE_STATUSES = {"PENDING", "BUILDING", "UPLOADING"}

ImageRow = dict[str, object]


def prime_image_name(dataset: str, task_dir: Path) -> str:
    """Return the stable Prime image name for a Harbor dataset task."""
    dataset_name = dataset.split("@", 1)[0].rsplit("/", 1)[-1]
    source = (
        dataset_name
        if task_dir.name == dataset_name
        else f"{dataset_name}-{task_dir.name}"
    )
    image_name = re.sub(r"[^a-z0-9._-]+", "-", source.lower()).strip("._-")
    if not image_name:
        raise ValueError(f"cannot derive a Prime image name for {task_dir}")
    return f"{image_name}:{IMAGE_TAG}"


def _find_prime_images(client: APIClient, targets: set[str]) -> dict[str, ImageRow]:
    """Find the newest container-image row for each target."""
    names = sorted(image.rsplit(":", 1)[0] for image in targets)
    search = os.path.commonprefix(names)
    found: dict[str, ImageRow] = {}
    offset = 0

    while True:
        params: dict[str, str | int] = {
            "limit": IMAGE_PAGE_SIZE,
            "offset": offset,
            "search": search,
        }
        if client.config.team_id:
            params["teamId"] = client.config.team_id
        try:
            response = client.request("GET", "/images", params=params)
        except APIError as error:
            raise RuntimeError(f"failed to search Prime images: {error}") from error

        rows = response.get("data", [])
        for row in rows:
            name, tag = row.get("imageName"), row.get("imageTag")
            image = f"{name}:{tag}"
            if (
                image in targets
                and image not in found
                and row.get("artifactType") == "CONTAINER_IMAGE"
            ):
                found[image] = row
        offset += len(rows)
        total_count = int(response.get("total_count", offset))
        if targets <= found.keys() or not rows or offset >= total_count:
            break

    return found


def _create_prime_image(client: APIClient, image: str, task_dir: Path) -> None:
    """Upload a Harbor task's unchanged environment context to Prime."""
    environment = task_dir / "environment"
    image_name, image_tag = image.rsplit(":", 1)
    payload: dict[str, str] = {
        "image_name": image_name,
        "image_tag": image_tag,
        "dockerfile_path": "Dockerfile",
        "platform": "linux/amd64",
    }
    if client.config.team_id:
        payload["teamId"] = client.config.team_id

    print(f"Creating Prime image {image} for Harbor task {task_dir.name}...")
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            tar.add(environment, arcname=".")
        archive.seek(0)

        try:
            build = client.request("POST", "/images/build", json=payload)
            upload = httpx.put(
                build["upload_url"],
                content=archive,
                headers={"Content-Type": "application/octet-stream"},
                timeout=IMAGE_BUILD_TIMEOUT,
            )
            upload.raise_for_status()
            client.request(
                "POST",
                f"/images/build/{build['build_id']}/start",
                json={"context_uploaded": True},
            )
        except (APIError, httpx.HTTPError) as error:
            raise RuntimeError(
                f"failed to create Prime image {image}: {error}"
            ) from error


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
    previous_rows: dict[str, tuple[object, object]] = {}
    deadlines: dict[str, float] = {}

    for image, task_dir in build_dirs.items():
        row = rows.get(image)
        status = row.get("status") if row else None
        if row and status == "COMPLETED":
            resolved[image] = str(row.get("displayRef") or row["fullImagePath"])
            continue
        if status not in ACTIVE_STATUSES:
            if row:
                previous_rows[image] = (row.get("id"), status)
            _create_prime_image(client, image, task_dir)
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
            marker = (row.get("id"), status)
            if marker == previous_rows.get(image):
                continue
            previous_rows.pop(image, None)
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
