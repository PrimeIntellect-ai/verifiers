"""The episode — one run's traces plus their shared standing, whole."""

import uuid
from typing import Any, Generic

from pydantic import BaseModel, Field

from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.trace import AgentConfigT, Error, RunInfo, Trace
from verifiers.v1.types import Usage


class EnvInfo(BaseModel):
    """The env that ran the episode, self-describing without the run's config."""

    id: str = ""
    """`EnvConfig.env_id`, e.g. `agentic-judge+gsm8k-v1`."""

    name: str | None = None
    """What the caller knows this env by, when that differs from `id`."""


class GroupInfo(BaseModel):
    """The episodes planned together from one task, compared against each other. The `id` is
    per group, not per task: planning the same task again is a new group."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    size: int = 1
    """How many episodes were planned in it."""


class Episode(BaseModel, Generic[DataT, StateT, AgentConfigT]):
    """The artifact Env.run produces. Contains multiple agents' traces."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    env: EnvInfo = Field(default_factory=EnvInfo)
    """The env that produced this episode."""
    group: GroupInfo | None = None
    """The group it was planned in, when the producer planned one."""
    run: RunInfo | None = None
    """The run this episode belongs to (eval or train), consumer-stamped. It lives here rather than
    on each trace because the episode is what a consumer dispatches, and an episode that produced
    no traces would otherwise have nowhere to say which run it was."""
    ok: bool = False
    """Whether the episode completed successfully."""
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest to newest."""
    traces: list[Trace[DataT, StateT, AgentConfigT]] = Field(default_factory=list)
    """Every agent's trace, in completion order."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Scratch space for episode-level metadata, the counterpart to `Trace.info`. What describes
    the whole episode belongs here rather than repeated on each of its traces."""

    @property
    def last_error(self) -> Error | None:
        """The last episode-level error captured across attempts."""
        return self.errors[-1] if self.errors else None

    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed across every trace's model calls;
        judge/off-graph usage stays on the traces (`Trace.extra_usage`)."""
        return Usage.aggregate(u for t in self.traces if (u := t.usage) is not None)

    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), summed across traces."""
        return sum(t.num_input_tokens for t in self.traces)

    @property
    def num_output_tokens(self) -> int:
        """Model-generated tokens across all turns, summed across traces."""
        return sum(t.num_output_tokens for t in self.traces)

    @property
    def num_total_tokens(self) -> int:
        """Final sequence lengths per branch, summed across traces."""
        return sum(t.num_total_tokens for t in self.traces)

    @property
    def num_turns(self) -> int:
        """Sampled turns, summed across traces."""
        return sum(t.num_turns for t in self.traces)

    @property
    def by_agent(self) -> dict[str, list[Trace[DataT, StateT, AgentConfigT]]]:
        """Traces grouped by agent name (e.g. n solvers), in completion order."""
        grouped: dict[str, list[Trace[DataT, StateT, AgentConfigT]]] = {}
        for trace in self.traces:
            grouped.setdefault(trace.agent.name, []).append(trace)
        return grouped

    def record_run(self, run: RunInfo | None = None, **info: Any) -> None:
        """Record the run identity and any extra metadata about this episode. Both describe the
        episode as a whole, so they are recorded once here rather than repeated on every trace."""
        if run is not None:
            self.run = run
        self.info.update(info)

    @classmethod
    def of(cls, trace: Trace, env: str = "") -> "Episode":
        """The single-agent record: one trace as its own episode."""
        return cls(env=EnvInfo(id=env), traces=[trace], ok=trace.ok)


WireEpisode = Episode[WireTaskData, State, WireAgentConfig]
"""Record loader for consumers without the run's packages: unknown task fields
survive in `task.model_extra`, agent configs parse loose (`WireAgentConfig`)."""
