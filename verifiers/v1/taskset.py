"""The taskset: a factory that derives tasks — point it at a dataset, get typed `Task`s.

Pure preprocessing: config in, tasks out. All episode behavior — rewards, metrics, stops,
lifecycle hooks, tools, user simulators — lives on the `Task` class the factory produces
(see `verifiers.v1.task`); the taskset never reaches execution. It exists so a dataset is
one `id` away (`--taskset.id gsm8k-v1`, hub-installable, typed knobs like `--taskset.split`),
but it is one task constructor among equals: a topology's `load_tasks`, a task built mid-`go`
from an upstream trace, or a replay buffer mint the same first-class tasks.

The generic declares what it produces — `Taskset[GSM8KTask, GSM8KConfig]` — which is how the
CLI narrows `--taskset.*` flags, how the wire types a trace's task (`task_type`), and how the
Environment validates the task class against the harness before any data is loaded.
"""

from typing import Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.task import TaskT
from verifiers.v1.types import ID
from verifiers.v1.utils.install import env_name


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-derivation knobs (split, size, paths)."""

    id: ID = ""
    """The taskset id, which selects this taskset: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `ID`). Set via `--taskset.id`."""

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over the task type it produces and its config, so `self.config` and
    `load_tasks` are fully typed. Subclass: implement `load_tasks`."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load_tasks(self) -> list[TaskT]:
        raise NotImplementedError
