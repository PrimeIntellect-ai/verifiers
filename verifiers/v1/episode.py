"""The episode — one run's traces plus their shared standing, whole."""

import functools
import uuid
from typing import Any, Generic

from pydantic import BaseModel, Field, computed_field

from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.graph import RECORD_FLOAT_DECIMALS
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.trace import (
    RECORD_DUMP_EXCLUDE,
    AgentConfigT,
    Error,
    RunInfo,
    Trace,
    TraceTask,
    add_record_streams,
    item_adapter,
    item_exclude,
    merge_exclude,
    splice_json,
    trace_json_parts,
)
from verifiers.v1.types import Usage


class EnvInfo(BaseModel):
    """The env that ran the episode, self-describing without the run's config."""

    id: str = ""
    """`EnvConfig.env_id`, e.g. `agentic-judge+gsm8k`."""
    name: str | None = None
    """The name the consumer runs the env under (e.g. an orchestrator env key)."""


class GroupInfo(BaseModel):
    """The consumer-defined rollout group this episode belongs to."""

    id: str


class Episode(BaseModel, Generic[DataT, StateT, AgentConfigT]):
    """The artifact Env.run produces. Contains multiple agents' traces."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique ID for this episode, auto-generated. Stable across retries."""
    attempt: int = 1
    """Which attempt of the episode this is, from 1."""

    env: EnvInfo = Field(default_factory=EnvInfo)
    """The env that produced this episode."""
    task: TraceTask[DataT]
    """The task dispatched to the env, including when no trace was produced."""
    group: GroupInfo | None = None
    """Consumer-assigned rollout group, if this episode was grouped."""
    run: RunInfo | None = None
    """The run this episode belongs to, consumer-stamped."""
    ok: bool = False
    """Whether the episode completed successfully."""
    is_timeout: bool = False
    """Whether the env's own `run()`/`finalize()` deadline expired; a per-agent
    stage deadline flags its trace (`Trace.is_timeout`)."""
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest to newest."""
    traces: list[Trace[DataT, StateT, AgentConfigT]] = Field(default_factory=list)
    """Every agent's trace, in completion order."""

    def record_run(self, run: RunInfo) -> None:
        """Record the run identity on the dispatched episode and each of its traces."""
        self.run = run
        for trace in self.traces:
            trace.run = run

    @property
    def last_error(self) -> Error | None:
        """The last episode-level error captured across attempts."""
        return self.errors[-1] if self.errors else None

    @computed_field
    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed across every trace's model calls;
        judge/off-graph usage stays on the traces (`Trace.extra_usage`)."""
        return Usage.aggregate(u for t in self.traces if (u := t.usage) is not None)

    @computed_field
    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), summed across traces."""
        return sum(t.num_input_tokens for t in self.traces)

    @computed_field
    @property
    def num_output_tokens(self) -> int:
        """Model-generated tokens across all turns, summed across traces."""
        return sum(t.num_output_tokens for t in self.traces)

    @computed_field
    @property
    def num_total_tokens(self) -> int:
        """Final sequence lengths per branch, summed across traces."""
        return sum(t.num_total_tokens for t in self.traces)

    @computed_field
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

    def to_record(
        self,
        float_decimals: int | None = RECORD_FLOAT_DECIMALS,
        exclude: dict | set | None = None,
    ) -> dict[str, Any]:
        """JSON record without raw trace tensors, which a python-mode `model_dump` keeps.
        Per-token node streams are numpy arrays (`graph.RECORD_STREAMS`): encode it with
        `to_json`, or with orjson and `OPT_SERIALIZE_NUMPY`. Per-token float streams are
        rounded to `float_decimals` (`None` keeps every digit). `exclude` is a pydantic
        exclude spec for the other fields; the streams are always included."""
        record = self.model_dump(
            mode="json",
            exclude=merge_exclude(
                {"traces": {"__all__": RECORD_DUMP_EXCLUDE}}, exclude
            ),
            exclude_none=True,
        )
        for trace_record, trace in zip(record["traces"], self.traces, strict=True):
            add_record_streams(trace_record, trace, float_decimals, exclude_none=True)
        return record

    def to_json(
        self,
        float_decimals: int | None = RECORD_FLOAT_DECIMALS,
        exclude: dict | set | None = None,
    ) -> bytes:
        """`to_record` as compact UTF-8 JSON, byte for byte. NaN and infinity become
        `null`.

        Pydantic's dump and orjson each hold the GIL for a whole call, so encoding a
        large episode in one call stalls every other thread of the process, such as an
        asyncio event loop, for as long as it takes. This encodes the episode one node at
        a time instead, and releases the GIL about every millisecond, so other threads
        run in between and the stall no longer grows with the episode. The total work
        stays the same."""
        exclude = merge_exclude({"traces": {"__all__": RECORD_DUMP_EXCLUDE}}, exclude)
        head = self.model_dump(
            mode="json",
            exclude=merge_exclude(exclude, {"traces": {"__all__": True}}),
            exclude_none=True,
        )
        adapter = item_adapter(type(self), "traces")
        traces = (
            trace_json_parts(
                trace,
                functools.partial(adapter.dump_python, trace),
                item_exclude(exclude, "traces", index) or {},
                float_decimals,
                exclude_none=True,
            )
            for index, trace in enumerate(self.traces)
        )
        # One join at the end: joining per level would copy the whole record each time.
        return b"".join(splice_json(head, "traces", traces))


WireEpisode = Episode[WireTaskData, State, WireAgentConfig]
"""Record loader for consumers without the run's packages: unknown task fields
survive in `task.model_extra`, agent configs parse loose (`WireAgentConfig`)."""
