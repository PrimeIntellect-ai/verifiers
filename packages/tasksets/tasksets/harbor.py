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

HarborSource = Literal["harbor", "package"]


class HarborTasksetConfig(vf.TasksetConfig):
    id: str | None = "harbor"
    source: HarborSource = "harbor"
    dataset: str = "hello-world"
    tasks: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    require_image: bool = False


class Author(BaseModel, extra="forbid", frozen=True):
    name: str | None = None
    email: str | None = None


class HarborTask(vf.Task, frozen=True):
    task_name: str
    instruction: str
    agent_timeout: float | None = None
    scoring_timeout: float | None = None
    keywords: list[str] = Field(default_factory=list)
    authors: list[Author] = Field(default_factory=list)
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    task_dir: str = Field(default="", exclude=True)

    @classmethod
    def from_dir(cls, task_dir: Path, *, require_image: bool) -> "HarborTask":
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        with task_toml_path.open("rb") as f:
            task_config = load_toml(f)
        sections: dict[str, dict[str, object]] = {}
        for name in ("environment", "task", "metadata", "agent", "verifier"):
            section = task_config.get(name) or {}
            if not isinstance(section, dict):
                raise TypeError(f"Harbor task [{name}] must be a mapping.")
            sections[name] = {str(key): value for key, value in section.items()}

        environment = sections["environment"]
        task_meta = sections["task"]
        metadata = sections["metadata"]
        agent_config = sections["agent"]
        verifier_config = sections["verifier"]

        raw_authors = task_meta.get("authors", [])
        if raw_authors is None:
            raw_authors = []
        if not isinstance(raw_authors, list):
            raise TypeError("Harbor task [task].authors must be a list.")
        authors_data: list[object] = list(raw_authors)
        if not authors_data and isinstance(metadata.get("author_name"), str):
            authors_data = [
                {
                    "name": metadata["author_name"],
                    "email": metadata.get("author_email"),
                }
            ]
        authors = [Author.model_validate(author) for author in authors_data]

        raw_docker_image = environment.get("docker_image")
        if raw_docker_image is not None and not isinstance(raw_docker_image, str):
            raise TypeError("Harbor task [environment].docker_image must be a string.")
        if (
            raw_docker_image is None
            and (task_dir / "environment" / "Dockerfile").exists()
        ):
            raise ValueError(
                f"Harbor task {task_dir.name!r} declares environment/Dockerfile "
                "but no pullable [environment].docker_image. Building Harbor "
                "Dockerfiles is not supported."
            )
        if raw_docker_image is None and require_image:
            raise ValueError(
                f"Harbor task {task_dir.name!r} has no pullable "
                "[environment].docker_image."
            )

        raw_agent_timeout = agent_config.get("timeout_sec")
        raw_scoring_timeout = verifier_config.get("timeout_sec")
        instruction = instruction_path.read_text().strip()
        raw_name = task_meta.get("name")
        if raw_name is not None and not isinstance(raw_name, str):
            raise TypeError("Harbor task [task].name must be a string.")
        raw_description = task_meta.get("description")
        if raw_description is not None and not isinstance(raw_description, str):
            raise TypeError("Harbor task [task].description must be a string.")
        raw_difficulty = metadata.get("difficulty")
        if raw_difficulty is not None and not isinstance(raw_difficulty, str):
            raise TypeError("Harbor task [metadata].difficulty must be a string.")
        raw_category = metadata.get("category")
        if raw_category is not None and not isinstance(raw_category, str):
            raise TypeError("Harbor task [metadata].category must be a string.")
        raw_keywords = task_meta.get("keywords", [])
        if not isinstance(raw_keywords, list):
            raise TypeError("Harbor task [task].keywords must be a list.")
        keywords: list[str] = []
        for keyword in raw_keywords:
            if not isinstance(keyword, str):
                raise TypeError("Harbor task [task].keywords must contain strings.")
            keywords.append(keyword)
        raw_tags = metadata.get("tags", [])
        if not isinstance(raw_tags, list):
            raise TypeError("Harbor task [metadata].tags must be a list.")
        tags: list[str] = []
        for tag in raw_tags:
            if not isinstance(tag, str):
                raise TypeError("Harbor task [metadata].tags must contain strings.")
            tags.append(tag)
        return cls(
            task_name=task_dir.name,
            instruction=instruction,
            task_dir=str(task_dir),
            prompt=[vf.UserMessage(content=instruction)],
            name=raw_name or task_dir.name,
            description=raw_description,
            image=raw_docker_image,
            resources=cls.resources_from_environment(environment),
            agent_timeout=(
                None
                if raw_agent_timeout is None
                else parse_number(raw_agent_timeout, 0.0)
            ),
            scoring_timeout=(
                None
                if raw_scoring_timeout is None
                else parse_number(raw_scoring_timeout, 0.0)
            ),
            keywords=keywords,
            authors=authors,
            difficulty=raw_difficulty,
            category=raw_category,
            tags=tags,
        )

    @staticmethod
    def resources_from_environment(environment: dict[str, object]) -> vf.Resources:
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
        return vf.Resources(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            disk_gb=disk_gb,
        )


class HarborTaskset(vf.Taskset[HarborTasksetConfig]):
    task_type = HarborTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        root = self.task_root()
        task_dirs = harbor_task_dirs(root, self.config.tasks)
        tasks = [
            HarborTask.from_dir(task_dir, require_image=self.config.require_image)
            for task_dir in task_dirs
        ]
        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {root}.")
        return tasks

    def to_task(self, task: vf.Task | vf.JsonData) -> vf.Task:
        harbor_task = super().to_task(task)
        if not isinstance(harbor_task, HarborTask):
            raise TypeError("HarborTaskset expected a HarborTask.")
        if harbor_task.task_dir:
            return harbor_task
        return HarborTask.from_dir(
            self.task_root() / harbor_task.task_name,
            require_image=self.config.require_image,
        )

    def task_root(self) -> Path:
        config = self.config
        if config.source == "harbor":
            cache_dir = (
                Path(config.cache_dir).expanduser() if config.cache_dir else None
            )
            return download_harbor_dataset(
                config.dataset,
                cache_dir=cache_dir,
                refresh=config.refresh,
            )
        if config.source == "package":
            root = bundle_tasks_root(config.dataset)
            if not root.exists():
                raise FileNotFoundError(
                    "HarborTaskset package source must contain "
                    f"{TASKS_SUBDIR}/. Not found: {root}"
                )
            return root
        raise ValueError(f"Unknown Harbor source: {config.source!r}")

    @vf.reward(weight=1.0)
    async def harbor_reward(self, task: HarborTask, runtime: vf.Runtime) -> float:
        tests_dir = Path(task.task_dir) / "tests"
        await runtime.write("/tmp/tests.tgz", self.make_tar(tests_dir))
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
            timeout=task.scoring_timeout,
        )
        try:
            reward = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
        except (OSError, RuntimeError, ValueError):
            return 0.0
        return parse_reward_text(reward)

    @staticmethod
    def make_tar(directory: Path) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for item in sorted(directory.iterdir()):
                tar.add(item, arcname=item.name)
        return buffer.getvalue()


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
