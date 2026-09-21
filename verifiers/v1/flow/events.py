"""Execution facts shared by the writer and read-only consumers."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from verifiers.v1.trace import Error

Status = Literal["ready", "held", "waiting", "terminal"]
RunReason = Literal["quiescent", "draining"]
CallStatus = Literal[
    "started", "succeeded", "failed", "attached", "stopped", "cancelled"
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Event(BaseModel):
    at: str = Field(default_factory=now)


class RunEvent(Event):
    type: Literal["run_started", "run_finished", "drain"]
    label: str | None = None
    reason: RunReason | None = None
    counts: dict[Status, int] = Field(default_factory=dict)


class Link(BaseModel):
    """A unit affected by a stage; does not identify a target execution."""

    unit: str
    label: str


class LinkEvent(Event):
    """The target execution selected work originating from the source execution."""

    type: Literal["link"] = "link"
    source_execution: str
    target_execution: str
    label: str


class StageEvent(Event):
    type: Literal["started", "stopped", "cancelled", "transition"]
    unit: str
    stage: str
    execution: str
    error: Error | None
    outcome: str | None = None
    to: str | None = None
    status: Status | None = None
    reason: str | None = None
    report: str | None = None
    """Filename under the run's reports/; contents are owned by the pipeline."""
    sha: str | None = None
    links: list[Link] = Field(default_factory=list)


class Invocation(BaseModel):
    model_config = ConfigDict(frozen=True)
    unit: str
    stage: str
    execution: str
    call: str
    key: str | None
    kind: str
    cache: str | None


class CallEvent(Event):
    type: Literal["call", "rollout"] = "call"
    invocation: Invocation
    status: CallStatus
    trace_id: str | None = None
    error: Error | None = None
    rollout: int | None = None
    source_call: str | None = None
    source_execution: str | None = None


class Steering(BaseModel):
    stage: str | None = None
    status: Status | None = None
    reason: str | None = None
    note: str | None = None
    data: dict[str, JsonValue] | None = None


class SteerEvent(Event):
    type: Literal["steer"] = "steer"
    unit: str
    sha: str
    action: Steering


EventRecord = RunEvent | StageEvent | CallEvent | SteerEvent | LinkEvent
event_adapter = TypeAdapter(Annotated[EventRecord, Field(discriminator="type")])


def append_event(path: Path, event: EventRecord) -> str:
    line = event.model_dump_json()
    with path.open("a") as file:
        file.write(line + "\n")
    return line
