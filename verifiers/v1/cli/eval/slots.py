"""The eval runner's observable unit: one planned env-rollout."""

from dataclasses import dataclass, field

from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace, episode_ok


@dataclass
class RunSlot:
    """One planned env-rollout of a task, observable while it happens: `traces`
    collects the live traces as they mint, `done` lands when the rollout is
    final. The dashboard renders slots; `--resume` preloads kept episodes as
    `finished` slots."""

    task: Task
    traces: list[Trace] = field(default_factory=list)
    done: bool = False

    @classmethod
    def finished(cls, traces: list[Trace]) -> "RunSlot":
        return cls(task=Task(traces[0].task.data), traces=list(traces), done=True)

    @property
    def errors(self) -> list[Error]:
        """Episode-level errors, read off the traces' shared episode stamp."""
        for trace in self.traces:
            if trace.episode is not None:
                return trace.episode.errors
        return []

    @property
    def ok(self) -> bool:
        """Whether the finished rollout is good: it produced traces and none of
        them — nor the episode — captured an error."""
        return self.done and bool(self.traces) and episode_ok(self.traces)
