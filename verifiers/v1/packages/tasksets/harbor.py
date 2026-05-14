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

from pydantic import Field
from typing_extensions import Unpack

from verifiers.utils.import_utils import load_toml

from ...config import TasksetConfig, merge_config_value
from ...taskset import Taskset, TasksetKwargs
from ...utils.sandbox_utils import SandboxClient
from verifiers.decorators import reward
from ...types import ConfigData, Handler, ProgramOptionMap

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
    dataset: str | None = None
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
    env: ConfigData = Field(default_factory=dict)


class HarborTaskset(Taskset):
    config_type = HarborTasksetConfig

    def __init__(
        self,
        dataset: str | None = None,
        task_names: Iterable[str] | None = None,
        cache_dir: str | Path | None = None,
        refresh: bool | None = None,
        docker_image: str | None = None,
        cpu_cores: float | None = None,
        memory_gb: float | None = None,
        disk_size_gb: float | None = None,
        timeout_minutes: int | None = None,
        agent_timeout_seconds: float | None = None,
        verifier_timeout_seconds: float | None = None,
        workdir: str | None = None,
        task_dir: str | None = None,
        scope: str | None = None,
        env: ProgramOptionMap | None = None,
        rewards: Iterable[Handler] = (),
        config: HarborTasksetConfig | None = None,
        **kwargs: Unpack[TasksetKwargs],
    ):
        self.config = type(self).config_type.from_config(config)
        dataset_value = merge_config_value(dataset, self.config.dataset)
        if dataset_value is not None and not isinstance(dataset_value, str):
            raise TypeError("HarborTaskset dataset must be a string.")
        self.dataset = dataset_value
        self._bundle_package = (
            _resolve_caller_package() if self.dataset is None else None
        )
        self.task_names = list(
            task_names
            if task_names is not None
            else self.config.task_names
            if self.config.task_names is not None
            else []
        )
        cache_dir_value = merge_config_value(cache_dir, self.config.cache_dir)
        self.cache_dir = (
            Path(str(cache_dir_value)).expanduser() if cache_dir_value else None
        )
        self.refresh = bool(self.config.refresh if refresh is None else refresh)
        self.docker_image = str(
            docker_image if docker_image is not None else self.config.docker_image
        )
        self.cpu_cores = float(
            cpu_cores if cpu_cores is not None else self.config.cpu_cores
        )
        self.memory_gb = float(
            memory_gb if memory_gb is not None else self.config.memory_gb
        )
        self.disk_size_gb = float(
            disk_size_gb if disk_size_gb is not None else self.config.disk_size_gb
        )
        self.timeout_minutes = int(
            timeout_minutes
            if timeout_minutes is not None
            else self.config.timeout_minutes
        )
        self.agent_timeout_seconds = float(
            agent_timeout_seconds
            if agent_timeout_seconds is not None
            else self.config.agent_timeout_seconds
        )
        self.verifier_timeout_seconds = float(
            verifier_timeout_seconds
            if verifier_timeout_seconds is not None
            else self.config.verifier_timeout_seconds
        )
        self.workdir = workdir if workdir is not None else self.config.workdir
        self.task_dir = task_dir if task_dir is not None else self.config.task_dir
        self.scope = scope if scope is not None else self.config.scope
        if self.scope not in {"rollout", "group", "global"}:
            raise ValueError("HarborTaskset scope must be rollout, group, or global.")
        self.env = {**self.config.env, **dict(env or {})}
        extra_rewards = list(rewards)
        super().__init__(
            source=self.load_rows,
            taskset_id="harbor",
            rewards=[harbor_reward, *extra_rewards],
            config=self.config,
            **kwargs,
        )

    def load_rows(self) -> list[ConfigData]:
        root = self.resolve_tasks_root()
        task_dirs = harbor_task_dirs(root, self.task_names)
        rows = [
            self.task_row(task_dir, index) for index, task_dir in enumerate(task_dirs)
        ]
        if not rows:
            raise ValueError(f"No valid Harbor tasks found in {root}.")
        return rows

    def resolve_tasks_root(self) -> Path:
        if self.dataset is not None:
            return download_harbor_dataset(
                self.dataset,
                cache_dir=self.cache_dir,
                refresh=self.refresh,
            )
        if self._bundle_package is None:
            raise RuntimeError(
                "HarborTaskset() without a dataset must be constructed from inside "
                "an installed Python package. Pass dataset='...' to fetch from "
                "Harbor Hub, or construct it from a packaged environment."
            )
        root = _bundle_tasks_root(self._bundle_package)
        if not root.exists():
            raise FileNotFoundError(
                "HarborTaskset() without a dataset requires "
                f"{self._bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
                f"directories. Not found: {root}"
            )
        return root

    def task_row(self, task_dir: Path, index: int) -> ConfigData:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        with task_toml_path.open("rb") as f:
            config = load_toml(f)
        environment = config.get("environment", {}) or {}
        if not isinstance(environment, Mapping):
            raise TypeError(f"{task_toml_path} [environment] must be a mapping.")
        agent_config = config.get("agent", {}) or {}
        verifier_config = config.get("verifier", {}) or {}
        instruction = instruction_path.read_text().strip()
        task_remote_dir = self.task_dir.rstrip("/") or "/task"
        sandbox = {
            "image": environment.get("docker_image") or self.docker_image,
            "cpu_cores": parse_number(environment.get("cpus"), self.cpu_cores),
            "memory_gb": parse_gb(environment.get("memory"), self.memory_gb),
            "disk_size_gb": parse_gb(environment.get("storage"), self.disk_size_gb),
            "timeout_minutes": self.timeout_minutes,
            "command_timeout": int(
                parse_number(
                    agent_config.get("timeout_sec"), self.agent_timeout_seconds
                )
            ),
            "workdir": self.workdir,
            "scope": self.scope,
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
                    "AGENT_WORKDIR": self.workdir,
                    **self.env,
                },
            },
            "harbor": {
                "task_dir": str(task_dir),
                "task_name": task_dir.name,
                "config": config,
                "docker_image": environment.get("docker_image"),
                "test_timeout": parse_number(
                    verifier_config.get("timeout_sec"),
                    self.verifier_timeout_seconds,
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
