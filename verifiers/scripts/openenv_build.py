from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


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
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return e.returncode or 1

    marker = project_path / ".openenv_image"
    marker.write_text(f"{image}\n")
    print(f"Wrote {marker} with image '{image}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
