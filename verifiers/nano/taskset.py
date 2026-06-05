"""The taskset: produces typed tasks and owns scoring.

A `Taskset` is the data + judgement half of an environment. It yields typed
`Task`s, optionally provides a `User` simulator and a `Toolset`, and defines
rewards/metrics as decorated methods. All task framing lives in each task's user
prompt (baked in by `load_tasks`); the agent drives control flow.

For a heterogeneous taskset (different verification per task), override `score`
to dispatch on the typed task, or have a single `@reward` branch on a task field.
"""

from typing import Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.nano.decorators import discover_decorated
from verifiers.nano.runtime import Runtime, RuntimeConfig
from verifiers.nano.task import TaskT
from verifiers.nano.toolset import Toolset
from verifiers.nano.transcript import Transcript
from verifiers.nano.user import User


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-generation knobs."""

    allowed_runtimes: list[str] | None = None
    """Runtime kinds this taskset supports (e.g. ['docker', 'prime']);
    None means any. The Environment hard-blocks a mismatched runtime."""


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

    def runtime_config(self, task: TaskT, base: RuntimeConfig) -> RuntimeConfig:
        """The runtime config for a task; override to refine it per task (e.g. the
        container image a harbor task declares). Defaults to the env's runtime."""
        return base

    async def verify(self, transcript: Transcript, runtime: Runtime) -> None:
        """In-runtime verification before scoring (e.g. run a task's test script in
        the same runtime the agent used). No-op by default."""

    async def score(self, transcript: Transcript) -> None:
        """Run all `@metric`/`@reward` methods over the finished transcript: each
        metric is recorded in `transcript.metrics[name]`, each reward (weighted)
        in `transcript.rewards[name]`, and `transcript.reward` sums them."""
        task = transcript.task

        async def value(fn) -> float:
            return float(await fn(task, transcript))

        for fn in discover_decorated(self, "metric"):
            transcript.metrics[fn.__name__] = await value(fn)
        for fn in discover_decorated(self, "reward"):
            transcript.rewards[fn.__name__] = await value(fn) * float(
                getattr(fn, "_vf_weight", 1.0)
            )
