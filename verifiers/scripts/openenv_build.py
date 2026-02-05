from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

yaml: Any | None
try:
    import yaml as _yaml  # type: ignore
except ImportError:
    yaml = None
else:
    yaml = _yaml


def _find_dockerfile(project_path: Path, dockerfile_arg: str | None) -> Path:
    if dockerfile_arg:
        dockerfile = Path(dockerfile_arg)
        if not dockerfile.is_absolute():
            dockerfile = (project_path / dockerfile).resolve()
        if not dockerfile.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")
        return dockerfile

    candidates = [
        project_path / "server" / "Dockerfile",
        project_path / "Dockerfile",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No Dockerfile found. Expected server/Dockerfile or Dockerfile."
    )


def _read_openenv_name(project_path: Path) -> str | None:
    if yaml is None:
        return None
    manifest = project_path / "openenv.yaml"
    if not manifest.exists():
        return None
    try:
        data = yaml.safe_load(manifest.read_text())
    except Exception:
        return None
    if isinstance(data, dict):
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _ensure_tag(image: str) -> str:
    last_segment = image.rsplit("/", 1)[-1]
    if ":" not in last_segment:
        return f"{image}:latest"
    return image


def _derive_image_name(project_path: Path) -> str:
    name = _read_openenv_name(project_path) or project_path.name
    return _ensure_tag(name)


def _extract_image_ref(item: dict[str, Any]) -> str | None:
    for key in ("image", "image_reference", "image_ref", "name", "ref"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            name = value.get("name")
            tag = value.get("tag")
            if isinstance(name, str) and name:
                if isinstance(tag, str) and tag:
                    return f"{name}:{tag}"
                return name
    return None


def _get_images_list() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["prime", "images", "list", "--output", "json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        items = data.get("items") or data.get("data") or data.get("images") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _resolve_fully_qualified_image(image: str) -> tuple[str | None, str | None]:
    if "/" in image:
        return image, None

    items = _get_images_list()
    if not items:
        return None, None
    desired = _ensure_tag(image)
    desired_base, desired_tag = desired.rsplit(":", 1)
    for item in items:
        ref = _extract_image_ref(item)
        if not ref:
            continue
        if ref.endswith(f"/{desired_base}:{desired_tag}") or ref.endswith(
            f"{desired_base}:{desired_tag}"
        ):
            status = item.get("status") or item.get("state")
            return ref, str(status) if status is not None else None

    return None, None


def _parse_image_from_push_output(output: str) -> str | None:
    for line in output.splitlines():
        match = re.search(r"Image:\s*(\\S+)", line)
        if match:
            return match.group(1).strip()
    return None


def _get_image_status(image_ref: str) -> str | None:
    items = _get_images_list()
    if not items:
        return None
    for item in items:
        ref = _extract_image_ref(item)
        if ref == image_ref:
            status = item.get("status") or item.get("state")
            return str(status) if status is not None else None
    return None


def _wait_for_ready(image_ref: str, timeout_s: int, interval_s: float) -> str | None:
    start = time.time()
    last_status = None
    while (time.time() - start) < timeout_s:
        status = _get_image_status(image_ref)
        if status:
            last_status = status
            status_norm = status.lower()
            if status_norm in ("ready", "succeeded"):
                return status
            if status_norm in ("failed", "error"):
                return status
        time.sleep(interval_s)
    return last_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and register an OpenEnv Docker image for Prime sandboxes."
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Path to the OpenEnv project (default: current directory).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image name to publish (default: from openenv.yaml name).",
    )
    parser.add_argument(
        "--dockerfile",
        default=None,
        help="Path to Dockerfile (default: server/Dockerfile or Dockerfile).",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Docker build context (default: project path).",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for image to reach Ready status (default: true).",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1200,
        help="Max seconds to wait for image Ready status (default: 1200).",
    )
    parser.add_argument(
        "--wait-interval",
        type=float,
        default=5.0,
        help="Seconds between status checks (default: 5).",
    )
    args = parser.parse_args(argv)

    project_path = Path(args.path).expanduser().resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"Project path not found: {project_path}", file=sys.stderr)
        return 2

    dockerfile = _find_dockerfile(project_path, args.dockerfile)
    context = (
        Path(args.context).expanduser().resolve() if args.context else project_path
    )

    image = _ensure_tag(args.image) if args.image else _derive_image_name(project_path)

    if shutil.which("prime") is None:
        print(
            "prime CLI not found. Install with: uv tool install prime",
            file=sys.stderr,
        )
        return 2

    cmd = [
        "prime",
        "images",
        "push",
        image,
        "--dockerfile",
        str(dockerfile),
        "--context",
        str(context),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        return e.returncode or 1

    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    resolved_image = _parse_image_from_push_output(output)
    status = None
    if resolved_image is None:
        resolved_image, status = _resolve_fully_qualified_image(image)
    if resolved_image is None:
        raise RuntimeError(
            "Could not resolve a fully qualified image reference. "
            "Run `prime images list --output json` and ensure the image is listed."
        )

    if status is None:
        status = _get_image_status(resolved_image)

    if args.wait:
        status = _wait_for_ready(resolved_image, args.wait_timeout, args.wait_interval)
        if status is None:
            print(
                "Timed out waiting for image status. Run `prime images list` to check progress.",
                file=sys.stderr,
            )
            return 1
        if status.lower() not in ("ready", "succeeded"):
            print(
                f"Image build did not complete successfully (status={status}).",
                file=sys.stderr,
            )
            return 1
    else:
        if status and status.lower() not in ("ready", "succeeded"):
            print(
                f"Image status is {status}. Wait for status Ready before running eval.",
                file=sys.stderr,
            )

    marker = project_path / ".openenv_image"
    marker.write_text(f"{resolved_image}\n")
    print(f"Wrote {marker} with image '{resolved_image}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
