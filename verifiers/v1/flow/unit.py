"""Atomic workflow state and boundary controls for one independently scheduled unit."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeInt
from typing_extensions import TypeVar

from verifiers.v1.flow.events import (
    TRANSITIONS,
    Status,
    SteerEvent,
    Steering,
    append_event,
)
from verifiers.v1.utils.time import now

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Flow

STATE = "state.json"


class UnitData(BaseModel):
    """Pipeline-owned durable data. Subclass for each kind of unit."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


D = TypeVar("D", bound=UnitData)
F = TypeVar("F", bound="Flow[Any]", default="Flow[Any]")


class UnitState(BaseModel, Generic[D]):
    model_config = ConfigDict(extra="forbid")

    data_type: str
    stages: list[str]
    revision: NonNegativeInt = 0
    stage: str
    status: Status = "ready"
    reason: str = ""
    data: D
    notes: list[str] = Field(default_factory=list)
    controls: dict[str, int] = Field(default_factory=dict)


class Execution(BaseModel):
    """An immutable reservation; the published next stage may change independently."""

    model_config = ConfigDict(frozen=True)
    id: str
    stage: str
    revision: int
    started_at: str


@dataclass(frozen=True)
class Transition(Generic[D]):
    """Publish a unit's next cursor and optional complete data together."""

    outcome: str
    summary: str = ""
    stage: str | None = None
    status: Status = "ready"
    data: D | None = None
    report: str | None = None
    """A pipeline-written filename under the run's reports/, published by this transition."""


class UnitInspection(BaseModel, Generic[D]):
    unit: str
    state: UnitState[D]
    active: Execution | None


def _write_state(path: Path, state: UnitState[Any]) -> None:
    """The caller holds the write lock; readers see the old or new complete state."""
    tmp = path / f"{STATE}.tmp"
    tmp.write_text(state.model_dump_json(indent=2) + "\n")
    os.replace(tmp, path / STATE)


class Unit(Generic[D, F]):
    """One current state file. A separate execution lock distinguishes held from settled."""

    # Bound by Flow; snapshot/data/execution are available while a stage runs.
    flow: F
    before: UnitState[D]
    data: D
    execution: Execution
    notes: str

    def __init__(self, path: Path, data_type: type[D] | None = None) -> None:
        self.path = Path(path)
        self.id = self.path.name
        definition = json.loads((self.path / STATE).read_text())
        if (
            data_type is not None
            and definition["data_type"]
            != f"{data_type.__module__}:{data_type.__name__}"
        ):
            raise ValueError(
                f"{path}: unit data model does not match {definition['data_type']}"
            )
        if data_type is None:
            module, name = definition["data_type"].split(":")
            data_type = getattr(importlib.import_module(module), name)
        self.data_type: type[D] = data_type
        self.state_type = cast(
            type[UnitState[D]], UnitState.__class_getitem__(data_type)
        )
        self.stages = frozenset(definition["stages"])

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        stage: str,
        data: D,
        stages: Iterable[str],
    ) -> Self:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (path / STATE).exists():
                allowed = sorted(stages)
                if stage not in allowed:
                    raise ValueError(f"unknown stage: {stage!r}")
                _write_state(
                    path,
                    UnitState(
                        data_type=f"{type(data).__module__}:{type(data).__name__}",
                        stages=allowed,
                        stage=stage,
                        data=data,
                    ),
                )
        return cls(path, type(data))

    def state(self) -> UnitState[D]:
        state = self.state_type.model_validate_json((self.path / STATE).read_text())
        self._validate_stage(state.stage)
        return state

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(
                f"unknown stage: {stage!r}; expected {sorted(self.stages)}"
            )

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        with (self.path / "write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def _execution_lock(self) -> Iterator[None]:
        with (self.path / "stage.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _active(self) -> Execution | None:
        try:
            with self._execution_lock():
                return None
        except BlockingIOError:
            return Execution.model_validate_json(
                (self.path / "active.json").read_text()
            )

    @contextmanager
    def executing(self) -> Iterator[None]:
        with ExitStack() as stack:
            with self._write_lock():
                stack.enter_context(self._execution_lock())
                state = self.state()
                execution = Execution(
                    id=uuid4().hex,
                    stage=state.stage,
                    revision=state.revision,
                    started_at=now(),
                )
                (self.path / "active.json").write_text(execution.model_dump_json())
            self.before, self.execution = state, execution
            self.data = state.data.model_copy(deep=True)
            self.notes = "\n\n".join(state.notes)
            yield

    def _publish(self, state: UnitState[D]) -> int:
        self._validate_stage(state.stage)
        # Revalidate even model_copy/update or mutated nested collections before touching disk.
        state = self.state_type.model_validate(state.model_dump(mode="json"))
        state.revision += 1
        _write_state(self.path, state)
        return state.revision

    def steer(
        self,
        *,
        stage: str | None = None,
        status: Status | None = None,
        reason: str | None = None,
        note: str | None = None,
        data: dict[str, JsonValue] | None = None,
        expected: int | None = None,
    ) -> int:
        """Boundary controls; data patches require a settled unit and its inspected revision."""
        with self._write_lock():
            state = self.state()
            if expected is not None and expected != state.revision:
                raise ValueError(f"{self.id}: stale workflow revision; inspect again")
            if data is not None:
                if expected is None:
                    raise ValueError("data updates require expected workflow revision")
                if self._active() is not None:
                    raise RuntimeError(f"{self.id}: stage is still active")
                state.data = self.data_type.model_validate(
                    {**state.data.model_dump(mode="json"), **data}
                )
            for key, value in (
                ("stage", stage),
                ("status", status),
                ("reason", reason),
            ):
                if value is not None:
                    setattr(state, key, value)
                    state.controls[key] = state.controls.get(key, 0) + 1
            if note is not None:
                state.notes.append(note)
            revision = self._publish(state)
            append_event(
                self.path.parent.parent / TRANSITIONS,
                SteerEvent(
                    unit=self.id,
                    revision=revision,
                    action=Steering(
                        stage=stage,
                        status=status,
                        reason=reason,
                        note=note,
                        data=data,
                    ),
                ),
            )
            return revision

    def apply(self, transition: Transition[D], *, before: UnitState[D]) -> int:
        with self._write_lock():
            state = self.state()
            for key, value in (
                (
                    "stage",
                    before.stage if transition.stage is None else transition.stage,
                ),
                ("status", transition.status),
                ("reason", transition.summary),
            ):
                if state.controls.get(key, 0) == before.controls.get(key, 0):
                    setattr(state, key, value)
            if transition.data is not None:
                state.data = transition.data
            if transition.status != "held":
                state.notes = state.notes[len(before.notes) :]
            return self._publish(state)

    def inspect(self) -> UnitInspection[D]:
        with self._write_lock():
            return UnitInspection(
                unit=self.id,
                state=self.state(),
                active=self._active(),
            )
