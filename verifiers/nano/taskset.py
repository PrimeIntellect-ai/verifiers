"""The taskset: produces typed tasks and owns scoring.

A `Taskset` is the data + judgement half of an environment. It yields typed
`Task`s, optionally provides a `User` simulator and a `Toolset`, and defines
rewards/metrics as decorated methods. All task framing lives in each task's user
prompt (baked in by `load_tasks`); the harness owns only control flow.

For a heterogeneous taskset (different verification per task), override `score`
to dispatch on the typed task, or have a single `@reward` branch on a task field.
"""

from typing import Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.nano import scoring
from verifiers.nano.task import TaskT
from verifiers.nano.tools import Toolset
from verifiers.nano.transcript import Transcript
from verifiers.nano.user import User


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-generation knobs."""


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over its task and config types, so `self.config` and `load_tasks`
    are fully typed. Subclass: implement `load_tasks`, add @reward/@metric."""

    transcript_type: type[Transcript] = Transcript

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self.user: User | None = None
        self.toolset: Toolset | None = None

    def load_tasks(self) -> list[TaskT]:
        raise NotImplementedError

    async def score(self, transcript: Transcript) -> None:
        """Run all decorated rewards/metrics over the finished transcript."""
        await scoring.score(self, transcript)
