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

import verifiers as vf
from verifiers.utils.import_utils import load_toml
from verifiers.v1.config import CallableEntry
from verifiers.v1.sandbox import SandboxConfig, sandbox_config_mapping
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import ConfigData
from verifiers.v1.utils.sandbox_utils import SandboxClient

TASKS_SUBDIR = "tasks"
HARBOR_DEFAULT_SANDBOX = SandboxConfig(
    image="python:3.11-slim",
    cpu_cores=2.0,
    memory_gb=4.0,
    disk_size_gb=10.0,
    timeout_minutes=120,
    workdir="/app",
    command_timeout=900,
)


class HarborTasksetConfig(TasksetConfig):
    rewards: list[CallableEntry] = ["harbor_reward"]
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    sandbox: SandboxConfig = HARBOR_DEFAULT_SANDBOX
    verifier_timeout_seconds: float = 900.0
    task_dir: str = "/task"
    env: dict[str, str] = {}


class HarborTaskset(Taskset[HarborTasksetConfig]):
    def __init__(self, config: HarborTasksetConfig | None = None):
        config = HarborTasksetConfig() if config is None else config
        assert isinstance(config, HarborTasksetConfig)
        if config.dataset is not None and not isinstance(config.dataset, str):
            raise TypeError("HarborTaskset dataset must be a string.")
        if config.dataset is None and config.bundle_package is None:
            config = config.model_copy(
                update={"bundle_package": self._caller_package()}
            )
        cache_dir_value = config.cache_dir
        self.cache_dir_path = (
            Path(str(cache_dir_value)).expanduser() if cache_dir_value else None
        )
        super().__init__(config=config)
        self.taskset_id = self.config.taskset_id or "harbor"

    def load_tasks(self) -> list[ConfigData]:
        root = self._tasks_root()
        task_dirs = self._task_dirs(root, list(self.config.task_names or []))
        tasks = [
            self._task(task_dir, index) for index, task_dir in enumerate(task_dirs)
        ]
        assert tasks, f"No valid Harbor tasks found in {root}."
        return tasks

    def _tasks_root(self) -> Path:
        if self.config.dataset is not None:
            return self._download_dataset(
                self.config.dataset,
                cache_dir=self.cache_dir_path,
                refresh=self.config.refresh,
            )
        if self.config.bundle_package is None:
            raise RuntimeError(
                "HarborTaskset() without a dataset must be constructed from inside "
                "an installed Python package. Pass dataset='...' to fetch from "
                "Harbor Hub, or construct it from a packaged environment."
            )
        root = self._bundle_tasks_root(self.config.bundle_package)
        if not root.exists():
            raise FileNotFoundError(
                "HarborTaskset() without a dataset requires "
                f"{self.config.bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
                f"directories. Not found: {root}"
            )
        return root

    def _task(self, task_dir: Path, index: int) -> ConfigData:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        with task_toml_path.open("rb") as f:
            task_config = load_toml(f)
        environment = task_config.get("environment", {}) or {}
        assert isinstance(environment, Mapping)
        agent_config = task_config.get("agent", {}) or {}
        verifier_config = task_config.get("verifier", {}) or {}
        if not isinstance(agent_config, Mapping):
            raise TypeError(f"{task_toml_path} [agent] must be a mapping.")
        if not isinstance(verifier_config, Mapping):
            raise TypeError(f"{task_toml_path} [verifier] must be a mapping.")
        instruction = instruction_path.read_text().strip()
        task_remote_dir = self.config.task_dir.rstrip("/") or "/task"
        sandbox = self._sandbox(self.config)
        cpu_cores = sandbox["cpu_cores"]
        memory_gb = sandbox["memory_gb"]
        disk_size_gb = sandbox["disk_size_gb"]
        command_timeout = sandbox["command_timeout"]
        assert isinstance(cpu_cores, int | float)
        assert isinstance(memory_gb, int | float)
        assert isinstance(disk_size_gb, int | float)
        assert isinstance(command_timeout, int | float)
        sandbox = {
            **sandbox,
            "image": environment.get("docker_image") or sandbox["image"],
            "cpu_cores": self._parse_number(environment.get("cpus"), cpu_cores),
            "memory_gb": self._parse_gb(environment.get("memory"), memory_gb),
            "disk_size_gb": self._parse_gb(environment.get("storage"), disk_size_gb),
            "command_timeout": int(
                self._parse_number(agent_config.get("timeout_sec"), command_timeout)
            ),
        }
        if "allow_internet" in environment:
            sandbox["network_access"] = bool(environment["allow_internet"])
        workdir = str(sandbox.get("workdir") or "/app")
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
                    "AGENT_WORKDIR": workdir,
                    **self.config.env,
                },
            },
            "harbor": {
                "task_dir": str(task_dir),
                "task_name": task_dir.name,
                "config": task_config,
                "docker_image": environment.get("docker_image"),
                "test_timeout": self._parse_number(
                    verifier_config.get("timeout_sec"),
                    self.config.verifier_timeout_seconds,
                ),
            },
            "info": {
                "harbor": {
                    "task_name": task_dir.name,
                    "docker_image": environment.get("docker_image"),
                }
            },
        }

    @classmethod
    def _caller_package(cls) -> str | None:
        for frame_info in inspect.stack()[1:]:
            package = frame_info.frame.f_globals.get("__package__")
            if not isinstance(package, str) or not package:
                package = frame_info.frame.f_globals.get("__name__")
            if not isinstance(package, str) or not package or package == "__main__":
                continue
            if package.startswith(("verifiers", "tasksets", "harnesses")):
                continue
            return package
        return None

    @classmethod
    def _bundle_tasks_root(cls, module_name: str) -> Path:
        try:
            tasks = cast(os.PathLike[str], files(module_name) / TASKS_SUBDIR)
            return Path(os.fspath(tasks))
        except TypeError as exc:
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise exc
            return Path(module_file).resolve().parent / TASKS_SUBDIR

    @classmethod
    def _sandbox(cls, config: HarborTasksetConfig) -> ConfigData:
        defaults = (
            sandbox_config_mapping(HARBOR_DEFAULT_SANDBOX, fill_defaults=False) or {}
        )
        configured = sandbox_config_mapping(config.sandbox, fill_defaults=False) or {}
        return {**defaults, **configured}

    @classmethod
    def _task_dirs(
        cls, root: Path, task_names: Iterable[str] | None = None
    ) -> list[Path]:
        selected = set(task_names or [])
        if not root.exists():
            raise FileNotFoundError(f"Harbor tasks path not found: {root}")
        tasks: list[Path] = []
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                raise ValueError(
                    f"Harbor tasks root {root} contains non-directory entry {task_dir}."
                )
            if not cls._is_task_dir(task_dir):
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

    @classmethod
    def _is_task_dir(cls, path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "task.toml").exists()
            and (path / "instruction.md").exists()
        )

    @classmethod
    def _parse_number(cls, value: object, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise TypeError("Expected a numeric value.")
        return float(value)

    @classmethod
    def _parse_gb(cls, value: object, default: float) -> float:
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

    @classmethod
    def _download_dataset(
        cls, dataset_id: str, *, cache_dir: Path | None = None, refresh: bool = False
    ) -> Path:
        harbor_bin = shutil.which("harbor")
        uvx_bin = shutil.which("uvx")
        if harbor_bin is None and uvx_bin is None:
            raise FileNotFoundError(
                f"Harbor dataset {dataset_id!r} requires the Harbor CLI or uvx. "
                "Install Harbor or uvx before using Harbor Hub datasets."
            )
        root = cache_dir or Path.home() / ".cache" / "verifiers" / "harbor"
        dataset_dir = root / cls._dataset_dir_name(dataset_id)
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

    @classmethod
    def _dataset_dir_name(cls, dataset_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("_") or "dataset"

    @classmethod
    async def _upload_tests(
        cls, client: SandboxClient, sandbox_id: str, task_dir: Path
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
        try:
            await cls._build_tests_archive(task_dir, tar_path)
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

    @classmethod
    async def _build_tests_archive(cls, task_dir: Path, tar_path: Path) -> None:
        with tarfile.open(tar_path, "w:gz") as tar:
            for dirname, arc_root in (("solution", "oracle"), ("tests", "tests")):
                root = task_dir / dirname
                if not root.exists():
                    continue
                for item in root.iterdir():
                    tar.add(item, arcname=f"{arc_root}/{item.name}")

    @classmethod
    def _parse_reward_text(cls, reward_text: str) -> float:
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


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


@vf.reward(weight=1.0)
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
        await HarborTaskset._upload_tests(client, sandbox_id, task_dir)
        test_timeout = int(HarborTaskset._parse_number(harbor.get("test_timeout"), 900))
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
    return HarborTaskset._parse_reward_text(str(reward_result.stdout or "").strip())
