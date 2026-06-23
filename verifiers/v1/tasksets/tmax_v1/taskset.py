"""tmax-v1: TMax Harbor tasks with explicit prebuilt Prime images."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor_v1 import HarborConfig, HarborTask, HarborTaskset


class TMaxConfig(HarborConfig):
    dataset: Literal["tmax/TMax-15K-Harbor@latest"] = "tmax/TMax-15K-Harbor@latest"
    ignore_dockerfile: bool = True
    """Keep Harbor from rejecting Dockerfile-only tasks; `image_map` supplies the image."""
    image_map: dict[str, str] = Field(default_factory=dict)
    """Prime image by Harbor task directory id, e.g. `task_000004_aeddda76`."""

    @field_validator("tasks", mode="before")
    @classmethod
    def coerce_single_task(cls, tasks: object) -> object:
        if isinstance(tasks, str):
            return [tasks]
        return tasks


class TMaxTaskset(HarborTaskset, vf.Taskset[HarborTask, TMaxConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[HarborTask]:
        tasks: list[HarborTask] = []
        for task in super().load_tasks():
            task_id = Path(task.task_dir).name
            image = self.config.image_map.get(task_id)
            if image is None:
                raise ValueError(
                    f"tmax-v1 task {task_id!r} has no image mapping. Add "
                    f'`{task_id} = "..."` under `[taskset.image_map]`.'
                )
            tasks.append(task.model_copy(update={"image": image}))
        return tasks
