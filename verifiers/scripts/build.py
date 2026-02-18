from __future__ import annotations

import argparse
import ast
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
    import yaml as _yaml
except ImportError:
    yaml = None
else:
    yaml = _yaml


DEFAULT_WAIT_TIMEOUT_S = 1200
DEFAULT_WAIT_INTERVAL_S = 5.0


def _normalize_env_id(raw_env_id: str) -> tuple[str, str]:
    env_id_dash = raw_env_id.strip()
    if not env_id_dash:
        raise ValueError("Environment id cannot be empty.")
    if "_" in env_id_dash:
        raise ValueError(
            "Environment id must use hyphens (e.g. openenv-echo), not underscores."
        )
    env_id_underscore = env_id_dash.replace("-", "_")
    return env_id_dash, env_id_underscore


def _find_dockerfile(project_dir: Path) -> Path:
    dockerfile = project_dir / "server" / "Dockerfile"
    if dockerfile.exists():
        return dockerfile
    raise FileNotFoundError(
        f"No Dockerfile found at {dockerfile}. Expected enforced layout with proj/server/Dockerfile."
    )


def _resolve_project_dir(environments_root: Path, env_id_underscore: str) -> Path:
    env_path = environments_root / env_id_underscore
    if not env_path.exists() or not env_path.is_dir():
        raise FileNotFoundError(
            f"Environment not found: {env_path}. Expected directory '{env_id_underscore}' under {environments_root}."
        )

    project_dir = env_path / "proj"
    if not project_dir.exists() or not project_dir.is_dir():
        raise FileNotFoundError(
            f"Embedded project directory not found: {project_dir}. "
            "Required structure: environments/<env_id_underscore>/proj/"
        )

    required = [
        project_dir / "openenv.yaml",
        project_dir / "pyproject.toml",
    ]
    for req in required:
        if not req.exists():
            raise FileNotFoundError(
                f"Required file missing: {req}. Expected project files under proj/."
            )

    return project_dir


def _read_project_port(project_dir: Path) -> int:
    openenv_yaml = project_dir / "openenv.yaml"
    if not openenv_yaml.exists() or yaml is None:
        return 8000
    try:
        data = yaml.safe_load(openenv_yaml.read_text())
    except Exception:
        return 8000
    if isinstance(data, dict) and "port" in data:
        try:
            return int(data["port"])
        except Exception:
            return 8000
    return 8000


def _read_project_app(project_dir: Path) -> str:
    openenv_yaml = project_dir / "openenv.yaml"
    if not openenv_yaml.exists() or yaml is None:
        return "server.app:app"
    try:
        data = yaml.safe_load(openenv_yaml.read_text())
    except Exception:
        return "server.app:app"
    if isinstance(data, dict):
        app = data.get("app")
        if isinstance(app, str) and app.strip():
            return app.strip()
    return "server.app:app"


def _build_start_command(app: str, port: int) -> str:
    # Prime sandboxes default to `tail -f /dev/null` unless start_command is explicit.
    # Also avoid relying on image-level PATH, which may not be propagated by sandbox runtime.
    return (
        "sh -lc "
        f'"cd /app/env && /app/.venv/bin/uvicorn {app} --host 0.0.0.0 --port {int(port)}"'
    )


def _resolve_app_module(project_dir: Path, app: str) -> Path:
    module = app.split(":", 1)[0].strip()
    if not module:
        raise RuntimeError(f"Invalid app entrypoint in openenv.yaml: {app}")
    app_module = project_dir / Path(*module.split("."))
    app_py = app_module.with_suffix(".py")
    if app_py.exists():
        return app_py
    init_py = app_module / "__init__.py"
    if init_py.exists():
        return init_py
    raise RuntimeError(
        f"Could not resolve app module from openenv.yaml app='{app}'. "
        f"Expected {app_py} or {init_py}."
    )


def _name_from_ast(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _detect_contract(project_dir: Path, app: str) -> str:
    """
    Detect OpenEnv contract from create_app(..., action_cls, observation_cls, ...):
    - mcp: action_cls=CallToolAction and observation_cls=CallToolObservation
    - gym: all other supported create_app signatures
    """
    app_file = _resolve_app_module(project_dir, app)
    try:
        tree = ast.parse(app_file.read_text(), filename=str(app_file))
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse app module for contract detection: {app_file}"
        ) from e

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _name_from_ast(node.func)
        if func_name != "create_app":
            continue
        if len(node.args) < 3:
            continue
        action_name = _name_from_ast(node.args[1])
        observation_name = _name_from_ast(node.args[2])
        if (
            action_name == "CallToolAction"
            and observation_name == "CallToolObservation"
        ):
            return "mcp"
        return "gym"

    raise RuntimeError(
        f"Could not detect OpenEnv contract: no supported create_app(...) call found in {app_file}."
    )


def _write_build_manifest(
    project_dir: Path,
    image: str,
    port: int,
    env_id: str,
    status: str | None,
    start_command: str,
    app: str,
    contract: str,
) -> Path:
    manifest = {
        "schema_version": 1,
        "environment_id": env_id,
        "image": image,
        "port": int(port),
        "app": app,
        "contract": contract,
        "start_command": start_command,
        "image_status": status,
    }
    manifest_path = project_dir / ".build.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _extract_image_ref(item: dict[str, Any]) -> str | None:
    for key in ("displayRef", "fullImagePath"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    image_name = item.get("imageName")
    image_tag = item.get("imageTag")
    if isinstance(image_name, str) and image_name:
        if isinstance(image_tag, str) and image_tag:
            return f"{image_name}:{image_tag}"
        return image_name
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

    for item in items:
        ref = _extract_image_ref(item)
        if not ref:
            continue
        if ref.endswith(f"/{image}") or ref.endswith(image):
            status = item.get("status") or item.get("state")
            return ref, str(status) if status is not None else None

    return None, None


def _parse_image_from_push_output(output: str) -> str | None:
    for line in output.splitlines():
        match = re.search(r"Image:\s*(\S+)", line)
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


def _is_success_status(status: str) -> bool:
    return status.lower() in {"ready", "succeeded", "completed"}


def _is_failure_status(status: str) -> bool:
    return status.lower() in {"failed", "error"}


def _wait_for_ready(
    image_ref: str,
    timeout_s: int = DEFAULT_WAIT_TIMEOUT_S,
    interval_s: float = DEFAULT_WAIT_INTERVAL_S,
) -> str | None:
    start = time.time()
    last_status = None
    while (time.time() - start) < timeout_s:
        status = _get_image_status(image_ref)
        if status:
            last_status = status
            if _is_success_status(status):
                return status
            if _is_failure_status(status):
                return status
        time.sleep(interval_s)
    return last_status


def _resolve_env_push_target(raw_env_id: str | None, raw_path: str) -> tuple[str, Path]:
    base_path = Path(raw_path).expanduser().resolve()

    if raw_env_id is None:
        env_dir = base_path
        env_id_underscore = env_dir.name
        if not env_id_underscore:
            raise ValueError("Could not infer environment id from --path.")
    else:
        _, env_id_underscore = _normalize_env_id(raw_env_id)
        env_dir = base_path / env_id_underscore

    env_id_dash = env_id_underscore.replace("_", "-")
    return env_id_dash, env_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and register a Docker image for an environment using the enforced "
            "layout: environments/<env_id_underscore>/proj."
        )
    )
    parser.add_argument(
        "env",
        nargs="?",
        help=(
            "Optional environment id (hyphenated, e.g. openenv-echo). "
            "When provided, it is appended to --path as the final directory name "
            "after converting hyphens to underscores."
        ),
    )
    parser.add_argument(
        "-p",
        "--path",
        default="./environments",
        help=(
            "Base path for environments (default: ./environments). "
            "When env id is omitted, this should point directly to the target "
            "environment directory."
        ),
    )
    args = parser.parse_args(argv)

    try:
        env_id_dash, env_path = _resolve_env_push_target(args.env, args.path)
        env_id_underscore = env_path.name
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not env_path.exists() or not env_path.is_dir():
        print(f"Environment path not found: {env_path}", file=sys.stderr)
        return 2

    try:
        project_dir = _resolve_project_dir(env_path.parent, env_id_underscore)
        dockerfile = _find_dockerfile(project_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    if shutil.which("prime") is None:
        print(
            "prime CLI not found. Install with: uv tool install prime",
            file=sys.stderr,
        )
        return 2

    dockerfile_rel = dockerfile.relative_to(project_dir)
    image = f"{env_id_dash}:latest"
    port = _read_project_port(project_dir)
    app = _read_project_app(project_dir)
    contract = _detect_contract(project_dir, app)
    start_command = _build_start_command(app=app, port=port)

    cmd = [
        "prime",
        "images",
        "push",
        image,
        "--dockerfile",
        str(dockerfile_rel),
        "--context",
        ".",
    ]
    print(
        f"Building image for '{env_id_dash}' with context='{project_dir}' "
        f"dockerfile='{dockerfile_rel}' image='{image}'"
    )

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_dir),
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
        print(
            "Could not resolve a fully qualified image reference. "
            "Run `prime images list --output json` and ensure the image is listed.",
            file=sys.stderr,
        )
        return 1

    if status is None:
        status = _get_image_status(resolved_image)
    if status is None or (
        not _is_success_status(status) and not _is_failure_status(status)
    ):
        status = _wait_for_ready(resolved_image)
    if status is None:
        print(
            "Timed out waiting for image status. Run `prime images list` to check progress.",
            file=sys.stderr,
        )
        return 1
    if not _is_success_status(status):
        print(
            f"Image build did not complete successfully (status={status}).",
            file=sys.stderr,
        )
        return 1

    manifest_path = _write_build_manifest(
        project_dir=project_dir,
        image=resolved_image,
        port=port,
        env_id=env_id_dash,
        status=status,
        start_command=start_command,
        app=app,
        contract=contract,
    )
    print(
        f"Wrote {manifest_path} with image='{resolved_image}' port={port} app='{app}' contract='{contract}' "
        f"start_command='{start_command}' status={status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
