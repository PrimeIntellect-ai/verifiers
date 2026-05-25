import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import cast

from verifiers.utils.import_utils import load_toml

from ...config import CallableEntry, TasksetConfig
from ...taskset import Taskset
from ...utils.sandbox_utils import SandboxClient
from verifiers.decorators import reward
from ...types import ConfigData

TASKS_SUBDIR = "tasks"


def _resolve_caller_package() -> str | None:
    for frame_info in inspect.stack()[1:]:
        package = frame_info.frame.f_globals.get("__package__")
        if not isinstance(package, str) or not package:
            package = frame_info.frame.f_globals.get("__name__")
        if not isinstance(package, str) or not package or package == "__main__":
            continue
        if package.startswith("verifiers"):
            continue
        return package
    return None


def _bundle_tasks_root(module_name: str) -> Path:
    try:
        tasks = cast(os.PathLike[str], files(module_name) / TASKS_SUBDIR)
        return Path(os.fspath(tasks))
    except TypeError as exc:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise exc
        return Path(module_file).resolve().parent / TASKS_SUBDIR


class HarborTasksetConfig(TasksetConfig):
    tasks: str | None = "load_tasks"
    rewards: list[CallableEntry] = ["harbor_reward"]
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    docker_image: str = "python:3.11-slim"
    cpu_cores: float = 2.0
    memory_gb: float = 4.0
    disk_size_gb: float = 10.0
    timeout_minutes: int = 120
    agent_timeout_seconds: float = 900.0
    verifier_timeout_seconds: float = 900.0
    workdir: str = "/app"
    task_dir: str = "/task"
    scope: str = "rollout"
    env: dict[str, str] = {}


class HarborTaskset(Taskset):
    config: HarborTasksetConfig

    def __init__(self, config: HarborTasksetConfig | None = None):
        config = HarborTasksetConfig() if config is None else config
        assert isinstance(config, HarborTasksetConfig)
        if config.dataset is not None and not isinstance(config.dataset, str):
            raise TypeError("HarborTaskset dataset must be a string.")
        if config.dataset is None and config.bundle_package is None:
            config = config.model_copy(
                update={"bundle_package": _resolve_caller_package()}
            )
        cache_dir_value = config.cache_dir
        self._cache_dir = (
            Path(str(cache_dir_value)).expanduser() if cache_dir_value else None
        )
        if config.scope not in {"rollout", "group", "global"}:
            raise ValueError("HarborTaskset scope must be rollout, group, or global.")
        super().__init__(config=config)
        self.taskset_id = self.config.taskset_id or "harbor"

    @property
    def task_names(self) -> list[str]:
        return list(self.config.task_names or [])

    @property
    def cpu_cores(self) -> float:
        return self.config.cpu_cores

    def resolve_tasks_root(self) -> Path:
        return resolve_tasks_root(self.config, cache_dir=self._cache_dir)


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_tasks(config: HarborTasksetConfig) -> list[ConfigData]:
    cache_dir = Path(str(config.cache_dir)).expanduser() if config.cache_dir else None
    root = resolve_tasks_root(config, cache_dir=cache_dir)
    task_dirs = harbor_task_dirs(root, list(config.task_names or []))
    rows = [
        harbor_task_row(config, task_dir, index)
        for index, task_dir in enumerate(task_dirs)
    ]
    if not rows:
        raise ValueError(f"No valid Harbor tasks found in {root}.")
    return rows


def resolve_tasks_root(
    config: HarborTasksetConfig, *, cache_dir: Path | None = None
) -> Path:
    if config.dataset is not None:
        return download_harbor_dataset(
            config.dataset,
            cache_dir=cache_dir,
            refresh=config.refresh,
        )
    if config.bundle_package is None:
        raise RuntimeError(
            "HarborTaskset() without a dataset must be constructed from inside "
            "an installed Python package. Pass dataset='...' to fetch from "
            "Harbor Hub, or construct it from a packaged environment."
        )
    root = _bundle_tasks_root(config.bundle_package)
    if not root.exists():
        raise FileNotFoundError(
            "HarborTaskset() without a dataset requires "
            f"{config.bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
            f"directories. Not found: {root}"
        )
    return root


def harbor_task_row(
    config: HarborTasksetConfig, task_dir: Path, index: int
) -> ConfigData:
    task_toml_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    with task_toml_path.open("rb") as f:
        task_config = load_toml(f)
    environment = task_config.get("environment", {}) or {}
    if not isinstance(environment, Mapping):
        raise TypeError(f"{task_toml_path} [environment] must be a mapping.")
    agent_config = task_config.get("agent", {}) or {}
    verifier_config = task_config.get("verifier", {}) or {}
    instruction = instruction_path.read_text().strip()
    task_remote_dir = config.task_dir.rstrip("/") or "/task"
    sandbox = {
        "image": environment.get("docker_image") or config.docker_image,
        "cpu_cores": parse_number(environment.get("cpus"), config.cpu_cores),
        "memory_gb": parse_gb(environment.get("memory"), config.memory_gb),
        "disk_size_gb": parse_gb(environment.get("storage"), config.disk_size_gb),
        "timeout_minutes": config.timeout_minutes,
        "command_timeout": int(
            parse_number(agent_config.get("timeout_sec"), config.agent_timeout_seconds)
        ),
        "workdir": config.workdir,
        "scope": config.scope,
    }
    if "allow_internet" in environment:
        sandbox["network_access"] = bool(environment["allow_internet"])
    return {
        "example_id": index,
        "task_name": task_dir.name,
        "instruction": instruction,
        "task_toml": task_toml_path.read_text(),
        "task_dir": str(task_dir),
        "prompt": [{"role": "user", "content": instruction}],
        "sandbox": sandbox,
        "program": {
            "files": {
                f"{task_remote_dir}/instruction.md": {"task": "instruction"},
                f"{task_remote_dir}/task.toml": {"task": "task_toml"},
            },
            "env": {
                "HARBOR_TASK_NAME": task_dir.name,
                "HARBOR_TASK_DIR": task_remote_dir,
                "HARBOR_INSTRUCTION_PATH": f"{task_remote_dir}/instruction.md",
                "AGENT_WORKDIR": config.workdir,
                **config.env,
            },
        },
        "harbor": {
            "task_dir": str(task_dir),
            "task_name": task_dir.name,
            "config": task_config,
            "docker_image": environment.get("docker_image"),
            "test_timeout": parse_number(
                verifier_config.get("timeout_sec"),
                config.verifier_timeout_seconds,
            ),
        },
        "info": {
            "harbor": {
                "task_name": task_dir.name,
                "docker_image": environment.get("docker_image"),
            }
        },
    }


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
        if not is_harbor_task_dir(task_dir):
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


def is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "task.toml").exists()
        and (path / "instruction.md").exists()
    )


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
    dataset_dir = root / safe_dataset_dir_name(dataset_id)
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


def safe_dataset_dir_name(dataset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("_") or "dataset"


@reward(weight=1.0)
async def harbor_reward(task, state) -> float:
    if state.get("error") is not None:
        return 0.0
    sandbox_id = state.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        return 0.0
    harbor = task.get("harbor")
    if not isinstance(harbor, Mapping):
        return 0.0
    task_dir = Path(str(harbor["task_dir"]))
    from prime_sandboxes import AsyncSandboxClient

    client = cast(SandboxClient, AsyncSandboxClient())
    try:
        await upload_harbor_tests(client, sandbox_id, task_dir)
        test_timeout = int(parse_number(harbor.get("test_timeout"), 900))
        result = await client.run_background_job(
            sandbox_id=sandbox_id,
            command="bash test.sh",
            working_dir="/tests",
            timeout=test_timeout,
        )
        state["harbor_tests"] = {
            "returncode": result.exit_code,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        reward_result = await client.execute_command(
            sandbox_id=sandbox_id,
            command=(
                "if [ -s /logs/verifier/reward.txt ]; then "
                "cat /logs/verifier/reward.txt; "
                "elif [ -s /logs/verifier/reward.json ]; then "
                "cat /logs/verifier/reward.json; fi"
            ),
        )
    except Exception as e:
        state["harbor_error"] = str(e)
        return 0.0
    finally:
        await client.aclose()
    return parse_reward_text(str(reward_result.stdout or "").strip())


async def upload_harbor_tests(
    client: SandboxClient, sandbox_id: str, task_dir: Path
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    try:
        await build_harbor_tests_archive(task_dir, tar_path)
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


async def build_harbor_tests_archive(task_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for dirname, arc_root in (("solution", "oracle"), ("tests", "tests")):
            root = task_dir / dirname
            if not root.exists():
                continue
            for item in root.iterdir():
                tar.add(item, arcname=f"{arc_root}/{item.name}")


def parse_reward_text(reward_text: str) -> float:
    if not reward_text:
        return 0.0
    try:
        return float(reward_text)
    except ValueError:
        pass
    try:
        data = json.loads(reward_text)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(data, Mapping):
        return 0.0
    return float(data.get("reward", 0.0))
