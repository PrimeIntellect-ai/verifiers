"""Read-only elapsed time and trace token totals, attributed to producing executions."""

from collections.abc import Iterable, Mapping
from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from verifiers.v1.flow.events import CallEvent, Event, RunEvent, StageEvent


class Stats(BaseModel):
    started_at: str | None = None
    finished_at: str | None = None
    tokens: int | None = None

    @computed_field
    @property
    def duration(self) -> float | None:
        """Elapsed seconds including intervening waits; an open span has no final duration."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (
            datetime.fromisoformat(self.finished_at)
            - datetime.fromisoformat(self.started_at)
        ).total_seconds()


class FlowStats(BaseModel):
    run: Stats = Field(default_factory=Stats)
    units: dict[str, Stats] = Field(default_factory=dict)
    executions: dict[str, Stats] = Field(default_factory=dict)


def summarize(events: Iterable[Event], tokens: Mapping[str, int]) -> FlowStats:
    """Sum each saved trace's num_total_tokens once, never attachments or extra usage.

    Unit elapsed time spans its first execution start through its latest finish.
    After TraceStore.index(), pass TraceStore.tokens as the trace-ID mapping.
    """
    result = FlowStats()
    seen: set[str] = set()
    for event in events:
        if isinstance(event, RunEvent):
            if event.type == "run_started":
                result.run.started_at = result.run.started_at or event.at
                result.run.finished_at = None
            elif event.type == "run_finished":
                result.run.finished_at = event.at
        elif isinstance(event, StageEvent):
            unit = result.units.setdefault(event.unit, Stats())
            if event.type == "started":
                unit.started_at = unit.started_at or event.at
                unit.finished_at = None
                result.executions[event.execution] = Stats(started_at=event.at)
            else:
                unit.finished_at = event.at
                result.executions[event.execution].finished_at = event.at
        elif (
            isinstance(event, CallEvent)
            and event.type == "rollout"
            and event.status == "started"
            and event.trace_id is not None
            and event.trace_id not in seen
        ):
            seen.add(event.trace_id)
            recorded = tokens.get(event.trace_id)
            if recorded is not None:
                for target in (
                    result.run,
                    result.units[event.invocation.unit],
                    result.executions[event.invocation.execution],
                ):
                    target.tokens = (target.tokens or 0) + recorded
    return result
