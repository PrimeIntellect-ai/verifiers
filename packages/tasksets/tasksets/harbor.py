from pathlib import Path

from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.utils.import_utils import load_toml
from verifiers.v1.utils.json_utils import json_data

from tasksets.utils.harbor_utils import (
    TASKS_SUBDIR,
    bundle_tasks_root,
    download_harbor_dataset,
    harbor_task_dirs,
    parse_gb,
    parse_number,
    parse_reward_text,
)

HARBOR_DEFAULT_RUNTIME: vf.JsonData = {
    "image": "python:3.11-slim",
    "cpu_cores": 2.0,
    "memory_gb": 4.0,
    "disk_size_gb": 10.0,
    "workdir": "/app",
    "timeout_seconds": 900.0,
}


class HarborTasksetConfig(vf.TasksetConfig):
    id: str | None = "harbor"
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    task_runtime: vf.JsonData = dict(HARBOR_DEFAULT_RUNTIME)
    verifier_timeout_seconds: float = 900.0
    task_dir: str = "/task"
    env: dict[str, str] = {}


class HarborRuntimeSpec(BaseModel, extra="forbid"):
    image: str
    cpu_cores: float = 2.0
    memory_gb: float = 4.0
    disk_size_gb: float = 10.0
    workdir: str = "/app"
    timeout_seconds: float = 900.0
    network_access: bool | None = None


class HarborProgramSpec(BaseModel, extra="forbid"):
    files: dict[str, dict[str, str]]
    env: dict[str, str]


class HarborSpec(BaseModel, extra="forbid"):
    task_dir: str
    task_name: str
    config: vf.JsonData
    docker_image: str | None = None
    test_timeout: float


class HarborTask(vf.Task, frozen=True):
    task_name: str
    instruction: str
    task_toml: str
    task_dir: str
    runtime_config: HarborRuntimeSpec
    program: HarborProgramSpec
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

    def task_root(self) -> Path:
        config = self.config
        if config.dataset is not None:
            cache_dir = (
                Path(config.cache_dir).expanduser() if config.cache_dir else None
            )
            return download_harbor_dataset(
                config.dataset,
                cache_dir=cache_dir,
                refresh=config.refresh,
            )
        if config.bundle_package is None:
            raise RuntimeError("HarborTaskset requires dataset or bundle_package.")
        root = bundle_tasks_root(config.bundle_package)
        if not root.exists():
            raise FileNotFoundError(
                "HarborTaskset bundle_package must contain "
                f"{TASKS_SUBDIR}/. Not found: {root}"
            )
        return root

    def task_from_dir(self, task_dir: Path) -> HarborTask:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        with task_toml_path.open("rb") as f:
            task_config = load_toml(f)
        environment = mapping(task_config.get("environment"), "environment")
        agent_config = mapping(task_config.get("agent"), "agent")
        verifier_config = mapping(task_config.get("verifier"), "verifier")
        instruction = instruction_path.read_text().strip()
        task_remote_dir = self.config.task_dir.rstrip("/") or "/task"
        runtime = harbor_runtime(
            self.config.task_runtime,
            environment=environment,
            agent_config=agent_config,
        )
        workdir = str(runtime.get("workdir") or "/app")
        program = HarborProgramSpec(
            files={
                f"{task_remote_dir}/instruction.md": {"task": "instruction"},
                f"{task_remote_dir}/task.toml": {"task": "task_toml"},
            },
            env={
                "HARBOR_TASK_NAME": task_dir.name,
                "HARBOR_TASK_DIR": task_remote_dir,
                "HARBOR_INSTRUCTION_PATH": f"{task_remote_dir}/instruction.md",
                "AGENT_WORKDIR": workdir,
                **self.config.env,
            },
        )
        raw_docker_image = environment.get("docker_image")
        docker_image = raw_docker_image if isinstance(raw_docker_image, str) else None
        harbor = HarborSpec(
            task_dir=str(task_dir),
            task_name=task_dir.name,
            config=json_data(task_config, context="Harbor task config"),
            docker_image=docker_image,
            test_timeout=parse_number(
                verifier_config.get("timeout_sec"),
                self.config.verifier_timeout_seconds,
            ),
        )
        return HarborTask(
            task_name=task_dir.name,
            instruction=instruction,
            task_toml=task_toml_path.read_text(),
            task_dir=str(task_dir),
            prompt=[vf.UserMessage(content=instruction)],
            image=str(runtime["image"]),
            runtime_config=HarborRuntimeSpec.model_validate(runtime),
            program=program,
            harbor=harbor,
        )

    @vf.reward(weight=1.0)
    async def harbor_reward(self, state: vf.State) -> float:
        reward = state.artifacts.get("harbor_reward")
        if isinstance(reward, int | float):
            return float(reward)
        tests = state.artifacts.get("harbor_tests")
        if isinstance(tests, dict):
            reward_text = tests.get("reward")
            if isinstance(reward_text, str):
                return parse_reward_text(reward_text)
            returncode = tests.get("returncode")
            if isinstance(returncode, int | float) and int(returncode) == 0:
                return 1.0
        return 0.0


def mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Harbor task [{name}] must be a mapping.")
    return {str(key): item for key, item in value.items()}


def harbor_runtime(
    configured: vf.JsonData,
    *,
    environment: dict[str, object],
    agent_config: dict[str, object],
) -> dict[str, object]:
    runtime: dict[str, object] = dict(configured)
    runtime["image"] = environment.get("docker_image") or runtime.get("image")
    runtime["cpu_cores"] = parse_number(environment.get("cpus"), 2.0)
    runtime["memory_gb"] = parse_gb(environment.get("memory"), 4.0)
    runtime["disk_size_gb"] = parse_gb(environment.get("storage"), 10.0)
    timeout_value = runtime.get("timeout_seconds")
    timeout_default = (
        float(timeout_value) if isinstance(timeout_value, str | int | float) else 900.0
    )
    runtime["timeout_seconds"] = parse_number(
        agent_config.get("timeout_sec"),
        timeout_default,
    )
    if "allow_internet" in environment:
        runtime["network_access"] = bool(environment["allow_internet"])
    return runtime


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
