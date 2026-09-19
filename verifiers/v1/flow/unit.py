"""Committed workflow state and boundary controls for one independently scheduled unit."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from verifiers.v1.flow.events import Status, SteerEvent, Steering, append_event, now

STATE = "state.json"
DEFINITION = "unit.json"
_IDENTITY = ("-c", "user.name=flow", "-c", "user.email=flow@local")


class UnitData(BaseModel):
    """Pipeline-owned durable data. Subclass for each kind of unit."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


D = TypeVar("D", bound=UnitData)


class Note(BaseModel):
    id: int
    text: str


class Controls(BaseModel):
    version: int = 0
    fields: dict[str, int] = Field(default_factory=dict)


class UnitState(BaseModel, Generic[D]):
    model_config = ConfigDict(extra="forbid")

    stage: str
    status: Status = "ready"
    reason: str = ""
    outcome: str = ""
    data: D
    notes: list[Note] = Field(default_factory=list)
    controls: Controls = Field(default_factory=Controls)


class Execution(BaseModel):
    """An immutable reservation; the committed next stage may change independently."""

    model_config = ConfigDict(frozen=True)
    id: str
    stage: str
    revision: str
    started_at: str


@dataclass(frozen=True)
class Transition(Generic[D]):
    """Publish a unit's next cursor, optional complete data, and workflow files together."""

    outcome: str
    summary: str = ""
    stage: str | None = None
    status: Status = "ready"
    data: D | None = None
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    report: str | None = None


class UnitInspection(BaseModel, Generic[D]):
    unit: str
    revision: str
    state: UnitState[D]
    active: Execution | None
    dirty: bool


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *_IDENTITY, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class Unit(Generic[D]):
    """A Git checkpoint. A separate execution lock distinguishes held from settled."""

    def __init__(self, path: Path, data_type: type[D] | None = None) -> None:
        self.path = Path(path)
        self.id = self.path.name
        definition = json.loads(self.read(DEFINITION, "HEAD") or "null")
        if definition is None:
            raise ValueError(f"{path}: no committed {DEFINITION}")
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
        self.events = self.path / definition["events"]

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        stage: str,
        data: D,
        stages: Iterable[str],
        events: Path,
        files: Mapping[str, str | bytes] | None = None,
    ) -> Unit[D]:
        path = Path(path)
        if (path / ".git").exists():
            unit = cls(path, type(data))
            unit.check_clean()
            unit.state()
            return unit
        allowed = sorted(stages)
        if stage not in allowed:
            raise ValueError(f"unknown stage: {stage!r}")
        if {str(Path(p)) for p in files or {}} & {STATE, DEFINITION}:
            raise ValueError("reserved unit file")
        path.mkdir(parents=True, exist_ok=True)
        git(path, "init", "-q")
        definition = {
            "data_type": f"{type(data).__module__}:{type(data).__name__}",
            "stages": allowed,
            "events": os.path.relpath(events, path),
        }
        (path / DEFINITION).write_text(json.dumps(definition, indent=2) + "\n")
        (path / STATE).write_text(
            UnitState(stage=stage, data=data).model_dump_json(indent=2) + "\n"
        )
        for rel, value in (files or {}).items():
            target = cls._file_at(path, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value.encode() if isinstance(value, str) else value)
        git(path, "add", "-A")
        git(path, "commit", "-q", "-m", "init")
        return cls(path, type(data))

    @staticmethod
    def _file_at(root: Path, rel: str) -> Path:
        path = Path(rel)
        if (
            not rel
            or path.is_absolute()
            or any(p in ("..", ".git") for p in path.parts)
        ):
            raise ValueError(f"unsafe unit path: {rel!r}")
        file = root / path
        if not file.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unit path escapes repository: {rel!r}")
        return file

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")

    def read(self, rel: str, sha: str = "HEAD") -> str | None:
        self._file_at(self.path, rel)
        out = subprocess.run(
            ["git", "-C", str(self.path), "show", f"{sha}:{rel}"],
            capture_output=True,
            check=False,
        )
        return out.stdout.decode() if out.returncode == 0 else None

    def state(self) -> UnitState[D]:
        state = self.state_type.model_validate_json(self.read(STATE) or "")
        self._validate_stage(state.stage)
        return state

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(
                f"unknown stage: {stage!r}; expected {sorted(self.stages)}"
            )

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        with (self.path / ".git" / "flow-write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def _execution_lock(self) -> Iterator[None]:
        with (self.path / ".git" / "flow-stage.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _active(self) -> Execution | None:
        try:
            with self._execution_lock():
                return None
        except BlockingIOError:
            return Execution.model_validate_json(
                (self.path / ".git" / "active.json").read_text()
            )

    @contextmanager
    def executing(self) -> Iterator[tuple[UnitState[D], Execution]]:
        with ExitStack() as stack:
            with self._write_lock():
                stack.enter_context(self._execution_lock())
                self._check_clean()
                state = self.state()
                execution = Execution(
                    id=uuid4().hex,
                    stage=state.stage,
                    revision=self.head(),
                    started_at=now(),
                )
                (self.path / ".git" / "active.json").write_text(
                    execution.model_dump_json()
                )
            yield state, execution

    def check_clean(self) -> None:
        with self._write_lock():
            self._check_clean()

    def _check_clean(self) -> None:
        if git(self.path, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(
                f"{self.path}: dirty or incomplete publication; repair before continuing"
            )

    def _commit(
        self, message: str, state: UnitState[D], files: Mapping[str, str | bytes]
    ) -> str:
        self._check_clean()
        self._validate_stage(state.stage)
        # Revalidate even model_copy/update or mutated nested collections before touching disk.
        state = self.state_type.model_validate(state.model_dump(mode="json"))
        paths = {rel: self._file_at(self.path, rel) for rel in files}
        if {str(Path(p)) for p in paths} & {STATE, DEFINITION}:
            raise ValueError("reserved unit file")
        for rel, file in paths.items():
            file.parent.mkdir(parents=True, exist_ok=True)
            value = files[rel]
            file.write_bytes(value.encode() if isinstance(value, str) else value)
        (self.path / STATE).write_text(state.model_dump_json(indent=2) + "\n")
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def commit(
        self,
        message: str,
        *,
        data: D | None = None,
        stage: str | None = None,
        status: Status | None = None,
        files: Mapping[str, str | bytes] | None = None,
    ) -> str:
        """Publish pipeline files; cursor/data changes require a settled unit."""
        with self._write_lock():
            if (
                any(value is not None for value in (data, stage, status))
                and self._active() is not None
            ):
                raise RuntimeError(f"{self.id}: stage is still active")
            state = self.state()
            if data is not None:
                state.data = data
            if stage is not None:
                state.stage = stage
            if status is not None:
                state.status = status
            return self._commit(message, state, files or {})

    def steer(
        self,
        *,
        stage: str | None = None,
        status: Status | None = None,
        reason: str | None = None,
        note: str | None = None,
        data: dict[str, JsonValue] | None = None,
        expected: str | None = None,
    ) -> str:
        """Boundary controls; data patches require a settled unit and the inspected HEAD."""
        with self._write_lock():
            if expected is not None and expected != self.head():
                raise ValueError(f"{self.id}: stale workflow revision; inspect again")
            state = self.state()
            if data is not None:
                if expected is None:
                    raise ValueError("data updates require expected workflow revision")
                if self._active() is not None:
                    raise RuntimeError(f"{self.id}: stage is still active")
                state.data = self.data_type.model_validate(
                    {**state.data.model_dump(mode="json"), **data}
                )
            state.controls.version += 1
            for key, value in (
                ("stage", stage),
                ("status", status),
                ("reason", reason),
            ):
                if value is not None:
                    setattr(state, key, value)
                    state.controls.fields[key] = state.controls.version
            if note is not None:
                state.notes.append(Note(id=state.controls.version, text=note))
            sha = self._commit("steer" + (f": {note}" if note else ""), state, {})
            append_event(
                self.events,
                SteerEvent(
                    unit=self.id,
                    sha=sha,
                    action=Steering.model_validate(
                        {
                            k: v
                            for k, v in {
                                "stage": stage,
                                "status": status,
                                "reason": reason,
                                "note": note,
                                "data": data,
                            }.items()
                            if v is not None
                        }
                    ),
                ),
            )
            return sha

    def apply(self, transition: Transition[D], *, before: UnitState[D]) -> str:
        with self._write_lock():
            state = self.state()
            for key, value in (
                ("stage", transition.stage or before.stage),
                ("status", transition.status),
                ("reason", transition.summary),
            ):
                if state.controls.fields.get(key, 0) <= before.controls.version:
                    setattr(state, key, value)
            state.outcome = transition.outcome
            if transition.data is not None:
                state.data = transition.data
            if transition.status != "held":
                consumed = {note.id for note in before.notes}
                state.notes = [note for note in state.notes if note.id not in consumed]
            return self._commit(
                f"{before.stage}: {transition.outcome}", state, transition.files
            )

    def inspect(self) -> UnitInspection[D]:
        with self._write_lock():
            return UnitInspection(
                unit=self.id,
                revision=self.head(),
                state=self.state(),
                active=self._active(),
                dirty=bool(git(self.path, "status", "--porcelain")),
            )
