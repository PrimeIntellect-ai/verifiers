"""Read-only elapsed time and trace token totals, attributed to producing executions."""

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field

from verifiers.v1.flow.events import CallEvent, Event, RunEvent, StageEvent


class Stats(BaseModel):
    started_at: str | None = None
    finished_at: str | None = None
    tokens: int | None = None


class FlowStats(BaseModel):
    run: Stats = Field(default_factory=Stats)
    jobs: dict[str, Stats] = Field(default_factory=dict)
    executions: dict[str, Stats] = Field(default_factory=dict)


def summarize(events: Iterable[Event], tokens: Mapping[str, int]) -> FlowStats:
    """Sum final call traces once, never attachments, earlier retries or extra usage.

    Job elapsed time spans its first execution start through its latest finish.
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
            job = result.jobs.setdefault(event.job, Stats())
            if event.type == "started":
                job.started_at = job.started_at or event.at
                job.finished_at = None
                result.executions[event.execution] = Stats(started_at=event.at)
            else:
                job.finished_at = event.at
                result.executions[event.execution].finished_at = event.at
        elif (
            isinstance(event, CallEvent)
            and event.status != "attached"
            and event.trace_id is not None
            and event.trace_id not in seen
        ):
            seen.add(event.trace_id)
            recorded = tokens.get(event.trace_id)
            if recorded is not None:
                for target in (
                    result.run,
                    result.jobs[event.invocation.job],
                    result.executions[event.invocation.execution],
                ):
                    target.tokens = (target.tokens or 0) + recorded
    return result
