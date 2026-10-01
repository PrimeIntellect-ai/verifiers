"""The taskset plugin's config: which rows load, under `--env.taskset.*`."""

from pathlib import Path

from pydantic import SerializeAsAny
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.types import ID


class SharedTasksetConfig(BaseConfig):
    """The knobs every taskset has, whatever its id — what several tasksets of one run
    can share."""

    task: SerializeAsAny[TaskConfig] = TaskConfig()
    """Config passed to each task, under `--env.taskset.task.*`."""
    system_prompt: Path | None = None
    """File whose text overrides each task's `TaskData.system_prompt` on
    iteration (e.g. a GEPA `best_system_prompt.txt`)."""
    network: NetworkPolicyConfig | None = None
    """Execution-time egress policy for every task of this taskset — the same
    `allow`/`block` object the runtimes carry. Set from TOML/CLI (`[env.taskset.network]`,
    `--env.taskset.network.allow`) it replaces each task's own policy and the taskset's
    default (`Taskset.network`); None leaves those in place. The runtime's own rules
    still intersect."""


class TasksetConfig(SharedTasksetConfig):
    id: ID = ""
    """Installed taskset package, set with `--env.taskset.id` (or the
    positional `eval <taskset-id>`)."""

    @property
    def name(self) -> str:
        return self.id
