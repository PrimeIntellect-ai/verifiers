from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from verifiers.decorators import reward
from verifiers.utils.import_utils import load_toml

from ...config import TasksetConfig
from ...taskset import Taskset

logger = logging.getLogger(__name__)
GB_SCALES = {"gb": 1.0, "g": 1.0, "mb": 1 / 1024, "m": 1 / 1024}


class HarborTasksetConfig(TasksetConfig):
    tasks: str | Path | None = None
    task_names: list[str] | None = None
    cache_dir: str | Path | None = None
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
    scope: Literal["rollout", "group", "global"] = "rollout"
    env: dict[str, object] = Field(default_factory=dict)


HARBOR_CONFIG_FIELDS = frozenset(HarborTasksetConfig.model_fields) - frozenset(
    TasksetConfig.model_fields
)


class HarborTaskset(Taskset):
    config_type = HarborTasksetConfig

    def __init__(
        self,
        tasks: str | Path | None = None,
        *,
        rewards: Iterable[Callable[..., object]] = (),
        config: HarborTasksetConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        config_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs) if key in HARBOR_CONFIG_FIELDS
        }
        if tasks is not None:
            config_kwargs["tasks"] = tasks
        env = config_kwargs.pop("env", None)
        self.config = type(self).config_type.from_config(config, **config_kwargs)
        if env is not None:
            if not isinstance(env, Mapping):
                raise TypeError("HarborTaskset env must be a mapping.")
            self.config.env = {**self.config.env, **dict(env)}
        for field in HARBOR_CONFIG_FIELDS:
            setattr(self, field, getattr(self.config, field))
        self.task_names = list(self.task_names or [])
        self.cache_dir = (
            Path(str(self.cache_dir)).expanduser() if self.cache_dir else None
        )
        self.env = dict(self.env)
        extra_rewards = list(rewards)
        super().__init__(
            source=self.load_rows,
            taskset_id="harbor",
            rewards=[harbor_reward, *extra_rewards],
            config=self.config,
            **kwargs,
        )

    def load_rows(self) -> list[dict[str, Any]]:
        if self.tasks is None:
            raise ValueError("HarborTaskset requires tasks=path_or_registry_id.")
        root = Path(self.tasks).expanduser()
        if not root.exists():
            if not isinstance(self.tasks, str):
                raise FileNotFoundError(f"Harbor tasks path not found: {root}")
            root = download_harbor_dataset(
                self.tasks,
                cache_dir=self.cache_dir,
                refresh=self.refresh,
            )
        task_dirs = harbor_task_dirs(root, self.task_names)
        rows = [
            self.task_row(task_dir, index) for index, task_dir in enumerate(task_dirs)
        ]
        if not rows:
            raise ValueError(f"No valid Harbor tasks found in {root}.")
        return rows

    def task_row(self, task_dir: Path, index: int) -> dict[str, Any]:
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
        docker_image = environment.get("docker_image")
        cpus = environment.get("cpus")
        agent_timeout = agent_config.get("timeout_sec")
        verifier_timeout = verifier_config.get("timeout_sec")
        return {
            "example_id": index,
            "task_name": task_dir.name,
            "instruction": instruction,
            "task_toml": task_toml_path.read_text(),
            "task_dir": str(task_dir),
            "prompt": [{"role": "user", "content": instruction}],
            "sandbox": {
                "image": docker_image or self.docker_image,
                "cpu_cores": float(cpus if cpus is not None else self.cpu_cores),
                "memory_gb": parse_gb(environment.get("memory"), self.memory_gb),
                "disk_size_gb": parse_gb(environment.get("storage"), self.disk_size_gb),
                "timeout_minutes": self.timeout_minutes,
                "command_timeout": int(
                    float(
                        agent_timeout
                        if agent_timeout is not None
                        else self.agent_timeout_seconds
                    )
                ),
                "workdir": self.workdir,
                "scope": self.scope,
            },
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
                "docker_image": docker_image,
                "test_timeout": float(
                    verifier_timeout
                    if verifier_timeout is not None
                    else self.verifier_timeout_seconds
                ),
            },
            "info": {
                "harbor": {
                    "task_name": task_dir.name,
                    "docker_image": docker_image,
                }
            },
        }


def harbor_task_dirs(root: Path, task_names: Iterable[str] | None = None) -> list[Path]:
    selected = set(task_names or [])
    if is_harbor_task_dir(root):
        if selected and root.name not in selected:
            return []
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Harbor tasks path not found: {root}")
    candidates = [
        task_dir
        for task_dir in sorted(root.iterdir())
        if task_dir.is_dir() and (not selected or task_dir.name in selected)
    ]
    tasks = [task_dir for task_dir in candidates if is_harbor_task_dir(task_dir)]
    for task_dir in [task_dir for task_dir in candidates if task_dir not in tasks]:
        logger.warning("Skipping %s: missing task.toml or instruction.md", task_dir)
    missing = sorted(selected - {path.name for path in tasks})
    if missing:
        raise ValueError(f"Requested Harbor tasks not found: {missing}.")
    return tasks


def is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "task.toml").exists()
        and (path / "instruction.md").exists()
    )


def parse_gb(value: object, default: float) -> float:
    if value is None:
        return default
    text = str(value).strip().lower()
    for suffix, scale in GB_SCALES.items():
        if text.endswith(suffix):
            return float(text.removesuffix(suffix)) * scale
    return float(text)


def download_harbor_dataset(
    dataset_id: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> Path:
    harbor_bin = shutil.which("harbor")
    uvx_bin = shutil.which("uvx")
    if harbor_bin is None and uvx_bin is None:
        raise FileNotFoundError(
            f"{dataset_id!r} is not a local path and the Harbor CLI is not installed. "
            "Install Harbor, install uvx, or pass a local Harbor task directory."
        )
    root = cache_dir or Path.home() / ".cache" / "verifiers" / "harbor"
    dataset_dir = root / safe_dataset_dir_name(dataset_id)
    if dataset_dir.exists() and not refresh:
        return dataset_dir
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
    return dataset_dir


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

    client = AsyncSandboxClient()
    try:
        await upload_harbor_tests(client, sandbox_id, task_dir)
        test_timeout = harbor.get("test_timeout")
        result = await client.execute_command(
            sandbox_id=sandbox_id,
            command="bash test.sh",
            working_dir="/tests",
            timeout=int(test_timeout if test_timeout is not None else 900),
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


async def upload_harbor_tests(client: object, sandbox_id: str, task_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    try:
        await build_harbor_tests_archive(task_dir, tar_path)
        remote_tar = "/tmp/harbor_tests.tar.gz"
        sandbox_client = cast(Any, client)
        await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
        await sandbox_client.execute_command(
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
