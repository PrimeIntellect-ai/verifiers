"""Execution facts shared by the writer and read-only consumers."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from verifiers.v1.trace import Error
from verifiers.v1.utils.time import now

TRANSITIONS = "transitions.jsonl"

Status = Literal["ready", "held", "waiting", "terminal"]
RunReason = Literal["idle", "draining"]
CallStatus = Literal[
    "started", "succeeded", "failed", "attached", "stopped", "cancelled"
]


class BaseEvent(BaseModel):
    at: str = Field(default_factory=now)


class RunEvent(BaseEvent):
    type: Literal["run_started", "run_finished", "drain"]
    label: str | None = None
    reason: RunReason | None = None
    counts: dict[Status, int] = Field(default_factory=dict)


class Link(BaseModel):
    """A unit affected by a stage; does not identify a target execution."""

    unit: str
    label: str


class LinkEvent(BaseEvent):
    """The target execution selected work originating from the source execution."""

    type: Literal["link"] = "link"
    source_execution: str
    target_execution: str
    label: str


class StageEvent(BaseEvent):
    type: Literal["started", "stopped", "cancelled", "transition"]
    unit: str
    stage: str
    execution: str
    error: Error | None = None
    outcome: str | None = None
    to: str | None = None
    status: Status | None = None
    reason: str | None = None
    report: str | None = None
    """Filename under the run's reports/; contents are owned by the pipeline."""
    revision: int | None = None
    links: list[Link] = Field(default_factory=list)


class CallIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    unit: str
    stage: str
    execution: str
    call: str
    key: str | None
    kind: str
    cache: str | None


class CallEvent(BaseEvent):
    type: Literal["call"] = "call"
    invocation: CallIdentity
    status: CallStatus
    trace_id: str | None = None
    error: Error | None = None
    source_call: str | None = None
    source_execution: str | None = None


class Steering(BaseModel):
    stage: str | None = None
    status: Status | None = None
    reason: str | None = None
    note: str | None = None
    data: dict[str, JsonValue] | None = None


class SteerEvent(BaseEvent):
    type: Literal["steer"] = "steer"
    unit: str
    revision: int
    action: Steering


Event = RunEvent | StageEvent | CallEvent | SteerEvent | LinkEvent
event_adapter = TypeAdapter(Annotated[Event, Field(discriminator="type")])


def append_event(path: Path, event: Event) -> str:
    line = event.model_dump_json()
    with path.open("a") as file:
        file.write(line + "\n")
    return line
