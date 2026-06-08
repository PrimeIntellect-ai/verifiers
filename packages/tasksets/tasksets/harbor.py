import io
import tarfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import verifiers.v1 as vf
from verifiers.utils.import_utils import load_toml
from tasksets.utils.harbor_utils import (
    TASKS_SUBDIR,
    bundle_tasks_root,
    download_harbor_dataset,
    harbor_task_dirs,
    parse_gb,
    parse_number,
    parse_reward_text,
)

HarborSource = Literal["hub", "local", "package"]


class HarborTasksetConfig(vf.TasksetConfig):
    id: str | None = "harbor"
    source: HarborSource = "hub"
    dataset: str = "hello-world"
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    require_image: bool = False
    verifier_timeout_seconds: float = 900.0


class HarborSpec(BaseModel, extra="forbid"):
    task_name: str
    docker_image: str | None = None
    test_timeout: float


class HarborTask(vf.Task, frozen=True):
    task_name: str
    instruction: str
    task_dir: str = Field(default="", exclude=True)
    harbor: HarborSpec


class HarborTaskset(vf.Taskset[HarborTasksetConfig]):
    task_type = HarborTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        root = self.task_root()
        task_dirs = harbor_task_dirs(root, self.config.task_names)
        tasks = [self.task_from_dir(task_dir) for task_dir in task_dirs]
        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {root}.")
        return tasks

    def to_task(self, task: vf.Task | vf.JsonData) -> vf.Task:
        harbor_task = super().to_task(task)
        if not isinstance(harbor_task, HarborTask):
            raise TypeError("HarborTaskset expected a HarborTask.")
        if harbor_task.task_dir:
            return harbor_task
        return self.task_from_dir(self.task_root() / harbor_task.task_name)

    def task_root(self) -> Path:
        config = self.config
        if config.source == "hub":
            cache_dir = (
                Path(config.cache_dir).expanduser() if config.cache_dir else None
            )
            return download_harbor_dataset(
                config.dataset,
                cache_dir=cache_dir,
                refresh=config.refresh,
            )
        if config.source == "local":
            return Path(config.dataset).expanduser().resolve()
        root = bundle_tasks_root(config.dataset)
        if not root.exists():
            raise FileNotFoundError(
                "HarborTaskset package source must contain "
                f"{TASKS_SUBDIR}/. Not found: {root}"
            )
        return root

    def task_from_dir(self, task_dir: Path) -> HarborTask:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        with task_toml_path.open("rb") as f:
            task_config = load_toml(f)
        environment = mapping(task_config.get("environment"), "environment")
        verifier_config = mapping(task_config.get("verifier"), "verifier")
        instruction = instruction_path.read_text().strip()
        raw_docker_image = environment.get("docker_image")
        docker_image = raw_docker_image if isinstance(raw_docker_image, str) else None
        if docker_image is None and (task_dir / "environment" / "Dockerfile").exists():
            raise ValueError(
                f"Harbor task {task_dir.name!r} declares environment/Dockerfile "
                "but no pullable [environment].docker_image. Building Harbor "
                "Dockerfiles is not supported."
            )
        if docker_image is None and self.config.require_image:
            raise ValueError(
                f"Harbor task {task_dir.name!r} has no pullable "
                "[environment].docker_image."
            )
        cpu_cores = (
            None
            if environment.get("cpus") is None
            else parse_number(environment.get("cpus"), 0.0)
        )
        memory_value = environment.get("memory_gb")
        if memory_value is None and environment.get("memory_mb") is not None:
            memory_gb = parse_number(environment.get("memory_mb"), 0.0) / 1024
        elif memory_value is None and environment.get("memory") is not None:
            memory_gb = parse_gb(environment.get("memory"), 0.0)
        else:
            memory_gb = None if memory_value is None else parse_gb(memory_value, 0.0)
        disk_value = environment.get("storage_gb")
        if disk_value is None and environment.get("storage_mb") is not None:
            disk_gb = parse_number(environment.get("storage_mb"), 0.0) / 1024
        elif disk_value is None and environment.get("storage") is not None:
            disk_gb = parse_gb(environment.get("storage"), 0.0)
        else:
            disk_gb = None if disk_value is None else parse_gb(disk_value, 0.0)
        gpu_count = (
            None
            if environment.get("gpus") is None
            else int(parse_number(environment.get("gpus"), 0.0))
        )
        harbor = HarborSpec(
            task_name=task_dir.name,
            docker_image=docker_image,
            test_timeout=parse_number(
                verifier_config.get("timeout_sec"),
                self.config.verifier_timeout_seconds,
            ),
        )
        return HarborTask(
            task_name=task_dir.name,
            instruction=instruction,
            task_dir=str(task_dir),
            prompt=[vf.UserMessage(content=instruction)],
            image=docker_image,
            resources=vf.Resources(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                gpu_count=gpu_count,
                disk_gb=disk_gb,
            ),
            harbor=harbor,
        )

    @vf.reward(weight=1.0)
    async def harbor_reward(self, task: HarborTask, runtime: vf.Runtime) -> float:
        tests_dir = Path(task.task_dir) / "tests"
        await runtime.write("/tmp/tests.tgz", make_tar(tests_dir))
        extract = await runtime.run(
            [
                "sh",
                "-c",
                "mkdir -p /logs/verifier /tests && tar -xzf /tmp/tests.tgz -C /tests",
            ]
        )
        if extract.returncode != 0:
            return 0.0
        await runtime.run(
            ["sh", "-c", "cd /tests && bash test.sh"],
            timeout=task.harbor.test_timeout,
        )
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
        except (OSError, RuntimeError, ValueError):
            return 0.0
        return parse_reward_text(reward)


def make_tar(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


def mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Harbor task [{name}] must be a mapping.")
    return {str(key): item for key, item in value.items()}


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
