"""Atomic workflow state and boundary controls for one independently scheduled job."""

from __future__ import annotations

import fcntl
import importlib
import json
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)
from typing_extensions import TypeVar

from verifiers.v1.flow.events import (
    TRANSITIONS,
    Status,
    SteerEvent,
    Transition,
    append_event,
    now,
)

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import Flow

STATE = "state.json"


class JobData(BaseModel):
    """Pipeline-owned durable data. Subclass for each kind of job."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


D = TypeVar("D", bound=JobData)
F = TypeVar("F", bound="Flow[Any]", default="Flow[Any]")


class JobState(BaseModel, Generic[D]):
    model_config = ConfigDict(extra="forbid")

    data_type: str
    stages: list[str]
    revision: NonNegativeInt = 0
    stage: str
    status: Status = "ready"
    reason: str = ""
    data: D
    notes: list[str] = Field(default_factory=list)
    controls: dict[Literal["stage", "status", "reason"], NonNegativeInt] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_stage(self) -> Self:
        if self.stage not in self.stages:
            raise ValueError(
                f"unknown stage: {self.stage!r}; expected {sorted(self.stages)}"
            )
        return self


class Execution(BaseModel):
    """An immutable reservation; the published next stage may change independently."""

    model_config = ConfigDict(frozen=True)
    id: str
    stage: str
    started_at: str


class JobInspection(BaseModel, Generic[D]):
    job: str
    state: JobState[D]
    active: bool


def _write_state(path: Path, state: JobState[Any]) -> None:
    """The caller holds the write lock; readers see the old or new complete state."""
    tmp = path / f"{STATE}.tmp"
    tmp.write_text(state.model_dump_json(indent=2) + "\n")
    tmp.replace(path / STATE)


class Job(Generic[D, F]):
    """One current state file. A separate execution lock distinguishes held from settled."""

    # Bound by Flow; snapshot/data/execution are available while a stage runs.
    flow: F
    before: JobState[D]
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
            != f"{data_type.__module__}:{data_type.__qualname__}"
        ):
            raise ValueError(
                f"{path}: job data model does not match {definition['data_type']}"
            )
        if data_type is None:
            module, name = definition["data_type"].split(":")
            obj: Any = importlib.import_module(module)
            for attr in name.split("."):
                obj = getattr(obj, attr)
            data_type = obj
        self.data_type: type[D] = data_type
        self.state_type = cast(type[JobState[D]], JobState.__class_getitem__(data_type))

    @classmethod
    def _create(
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
                _write_state(
                    path,
                    JobState.__class_getitem__(type(data))(
                        data_type=f"{type(data).__module__}:{type(data).__qualname__}",
                        stages=sorted(stages),
                        stage=stage,
                        data=data.model_dump(mode="json"),
                    ),
                )
        return cls(path, type(data))

    def state(self) -> JobState[D]:
        return self.state_type.model_validate_json((self.path / STATE).read_text())

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

    def _active(self) -> bool:
        try:
            with self._execution_lock():
                return False
        except BlockingIOError:
            return True

    @contextmanager
    def executing(self) -> Iterator[None]:
        with ExitStack() as stack:
            with self._write_lock():
                stack.enter_context(self._execution_lock())
                state = self.state()
                execution = Execution(
                    id=uuid4().hex,
                    stage=state.stage,
                    started_at=now(),
                )
            self.before, self.execution = state, execution
            self.data = state.data.model_copy(deep=True)
            self.notes = "\n\n".join(state.notes)
            yield

    def _apply(
        self,
        transition: Transition[D],
        *,
        before: JobState[D] | None = None,
        expected: int | None = None,
    ) -> JobState[D]:
        """Publish a control, or reconcile a completed stage against its starting state."""
        with self._write_lock():
            state = self.state()
            if expected is not None and expected != state.revision:
                raise ValueError(f"{self.id}: stale workflow revision; inspect again")
            if transition.data is not None:
                if before is None:
                    if expected is None:
                        raise ValueError(
                            "data updates require expected workflow revision"
                        )
                    if self._active():
                        raise RuntimeError(f"{self.id}: stage is still active")
                state.data = transition.data
            for key in ("stage", "status", "reason"):
                value = getattr(transition, key)
                if value is None:
                    continue
                if before is None:
                    state.controls[key] = state.controls.get(key, 0) + 1
                elif state.controls.get(key, 0) != before.controls.get(key, 0):
                    continue
                setattr(state, key, value)
            if before is not None and transition.status != "held":
                state.notes = state.notes[len(before.notes) :]
            if transition.note is not None:
                state.notes.append(transition.note)
            # Revalidate nested mutations and model_copy updates before publication.
            state = self.state_type.model_validate(state.model_dump(mode="json"))
            state.revision += 1
            _write_state(self.path, state)
            if before is None:
                append_event(
                    self.path.parent.parent / TRANSITIONS,
                    SteerEvent(
                        job=self.id,
                        revision=state.revision,
                        action=transition.model_dump(mode="json", exclude_none=True),
                    ),
                )
            return state

    def inspect(self) -> JobInspection[D]:
        with self._write_lock():
            return JobInspection(
                job=self.id,
                state=self.state(),
                active=self._active(),
            )
