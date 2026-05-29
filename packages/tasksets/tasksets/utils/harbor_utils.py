import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import cast

from verifiers.v1.sandbox import SandboxConfig
from verifiers.v1.utils.sandbox_utils import SandboxClient

TASKS_SUBDIR = "tasks"


def bundle_tasks_root(module_name: str) -> Path:
    try:
        tasks = cast(os.PathLike[str], files(module_name) / TASKS_SUBDIR)
        return Path(os.fspath(tasks))
    except TypeError as exc:
        module = sys.modules.get(module_name)
        module_file = module.__dict__.get("__file__") if module is not None else None
        if not isinstance(module_file, str):
            raise exc
        return Path(module_file).resolve().parent / TASKS_SUBDIR


def harbor_sandbox(default: SandboxConfig, configured: SandboxConfig) -> SandboxConfig:
    return SandboxConfig.model_validate(
        {**default.data(fill_defaults=False), **configured.data(fill_defaults=False)}
    )


def harbor_task_dirs(root: Path, task_names: Iterable[str] | None = None) -> list[Path]:
    selected = set(task_names or [])
    if not root.exists():
        raise FileNotFoundError(f"Harbor tasks path not found: {root}")
    tasks: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            raise ValueError(
                f"Harbor tasks root {root} contains non-directory entry {task_dir}."
            )
        if not (
            (task_dir / "task.toml").exists() and (task_dir / "instruction.md").exists()
        ):
            raise ValueError(
                f"Malformed Harbor task {task_dir}: missing task.toml or "
                "instruction.md."
            )
        if not selected or task_dir.name in selected:
            tasks.append(task_dir)
    if selected:
        found = {path.name for path in tasks}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"Requested Harbor tasks not found: {missing}.")
    return tasks


def parse_number(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("Expected a numeric value.")
    return float(value)


def parse_gb(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().lower()
    if text.endswith("gb"):
        return float(text[:-2])
    if text.endswith("g"):
        return float(text[:-1])
    if text.endswith("mb"):
        return float(text[:-2]) / 1024
    if text.endswith("m"):
        return float(text[:-1]) / 1024
    return float(text)


def download_harbor_dataset(
    dataset_id: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> Path:
    harbor_bin = shutil.which("harbor")
    uvx_bin = shutil.which("uvx")
    if harbor_bin is None and uvx_bin is None:
        raise FileNotFoundError(
            f"Harbor dataset {dataset_id!r} requires the Harbor CLI or uvx. "
            "Install Harbor or uvx before using Harbor Hub datasets."
        )
    root = cache_dir or Path.home() / ".cache" / "verifiers" / "harbor"
    dataset_dir = root / (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("_") or "dataset"
    )
    task_root = dataset_dir / dataset_id.rsplit("/", 1)[-1]
    if dataset_dir.exists() and not refresh:
        return task_root
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if harbor_bin is not None:
        command = [
            harbor_bin,
            "datasets",
            "download",
            dataset_id,
            "--output-dir",
            str(dataset_dir),
        ]
    else:
        assert uvx_bin is not None
        command = [
            uvx_bin,
            "harbor",
            "datasets",
            "download",
            dataset_id,
            "--output-dir",
            str(dataset_dir),
        ]
    if refresh:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    return task_root


async def upload_harbor_tests(
    client: SandboxClient, sandbox_id: str, task_dir: Path
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for dirname, arc_root in (("solution", "oracle"), ("tests", "tests")):
                root = task_dir / dirname
                if not root.exists():
                    continue
                for item in root.iterdir():
                    tar.add(item, arcname=f"{arc_root}/{item.name}")
        remote_tar = "/tmp/harbor_tests.tar.gz"
        await client.upload_file(sandbox_id, remote_tar, str(tar_path))
        await client.execute_command(
            sandbox_id=sandbox_id,
            command=(
                f"mkdir -p /oracle /tests /logs/verifier && "
                f"tar -xzf {remote_tar} -C / && rm {remote_tar}"
            ),
            timeout=900,
        )
    finally:
        tar_path.unlink(missing_ok=True)


def parse_reward_text(reward_text: str) -> float:
    if not reward_text:
        return 0.0
    try:
        return float(reward_text)
    except ValueError:
        try:
            data = json.loads(reward_text)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(data, dict):
        return 0.0
    return float(data.get("reward", 0.0))
