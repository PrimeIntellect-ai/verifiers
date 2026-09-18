"""A unit: one git repository whose head is where the unit stands.

`state.json` at the head carries the three fields the loop reads -- `stage`, `status`,
`reason` -- and whatever else the pipeline's stages keep there (credits, a version, a
pin). Every stage ends in one commit: its `Transition`, applied to the state, with the files
the stage wants kept beside it. `Unit.steer` commits operator controls with versions so
controls accepted during active work still win at the next boundary.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

STATE = "state.json"
Status = Literal["ready", "held", "waiting", "terminal"]
"""`ready`: run `stage` when a slot opens. `held`: wait for an operator. `waiting`: wait
for the pipeline (a campaign stage releases it). `terminal`: never touched again."""

_IDENTITY = ("-c", "user.name=flow", "-c", "user.email=flow@local")


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *_IDENTITY, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class Transition:
    """How a stage came out and where the unit goes next. `stage` None keeps the current
    stage (a hold, a wait); `status` is what the loop does with the unit afterwards."""

    outcome: str
    summary: str = ""
    stage: str | None = None
    status: Status = "ready"
    files: dict[str, str] = field(default_factory=dict)
    """Files to commit beside `state.json`, by path in the repository."""
    state: dict[str, Any] = field(default_factory=dict)
    """Fields to merge into `state.json`: the pipeline's own (credits, a pin, a version)."""

    @classmethod
    def to(cls, stage: str, outcome: str, summary: str = "", **kw: Any) -> Self:
        """On to `stage`."""
        return cls(outcome, summary, stage=stage, status="ready", **kw)

    @classmethod
    def end(cls, outcome: str, summary: str = "", **kw: Any) -> Self:
        """The unit's end: `outcome` names it (sealed, retired, done...)."""
        return cls(outcome, summary, status="terminal", **kw)

    @classmethod
    def hold(cls, reason: str, **kw: Any) -> Self:
        """Stop at this stage until an operator releases the unit."""
        return cls("held", reason, status="held", **kw)

    @classmethod
    def wait(cls, reason: str, **kw: Any) -> Self:
        """Stop at this stage until the pipeline releases the unit."""
        return cls("waiting", reason, status="waiting", **kw)


class Unit:
    """One repository. Scheduling state comes only from committed HEAD.

    Writes take a short per-repository lock. Dirty or failed publications require
    operator repair; they are never adopted as scheduling state.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.id = self.path.name

    @classmethod
    def create(
        cls, path: Path, state: dict[str, Any], files: dict[str, str] | None = None
    ) -> Self:
        """A new unit at `path`, or the clean unit already there."""
        unit = cls(path)
        if not (path / ".git").exists():
            path.mkdir(parents=True, exist_ok=True)
            git(path, "init", "-q")
            unit.commit("init", files=files, state={"status": "ready", **state})
        else:
            unit.check_clean()
            unit.state()  # refuse incomplete initialization
        return unit

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")

    def _file(self, rel: str, *, write: bool = False) -> Path:
        path = Path(rel)
        if (
            not rel
            or path.is_absolute()
            or ".." in path.parts
            or ".git" in path.parts
            or (write and path == Path(STATE))
        ):
            raise ValueError(f"reserved or unsafe unit path: {rel!r}")
        file = self.path / path
        if not file.resolve().is_relative_to(self.path.resolve()):
            raise ValueError(f"unit path escapes repository: {rel!r}")
        return file

    def read(self, rel: str, sha: str | None = None) -> str | None:
        """Read text without trimming whitespace, from the worktree or a Git ref."""
        file = self._file(rel)
        if sha is None:
            return file.read_bytes().decode() if file.exists() else None
        try:
            # Decode bytes directly: text-mode subprocess output normalizes CRLF.
            return subprocess.run(
                ["git", "-C", str(self.path), "show", f"{sha}:{rel}"],
                check=True,
                capture_output=True,
            ).stdout.decode()
        except subprocess.CalledProcessError:
            return None

    def read_json(self, rel: str, sha: str | None = None) -> dict[str, Any]:
        raw = self.read(rel, sha)
        return json.loads(raw) if raw else {}

    def state(self) -> dict[str, Any]:
        raw = self.read(STATE, "HEAD")
        if raw is None:
            raise RuntimeError(f"{self.path}: no committed {STATE}; repair the unit")
        return json.loads(raw)

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        lock_path = Path(git(self.path, "rev-parse", "--absolute-git-dir"))
        with (lock_path / "flow-write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def check_clean(self) -> None:
        with self._write_lock():
            self._check_clean()

    def _check_clean(self) -> None:
        if git(self.path, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(
                f"{self.path}: dirty or incomplete publication; repair and commit "
                "or discard changes before continuing"
            )

    def commit(
        self,
        message: str,
        *,
        files: dict[str, str] | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Write files and merge state in one commit. Refuse preexisting dirt."""
        if state and "_control" in state:
            raise ValueError("_control is reserved for Unit.steer")
        with self._write_lock():
            return self._commit(message, files=files, state=state)

    def _commit(
        self,
        message: str,
        *,
        files: dict[str, str] | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        self._check_clean()
        paths = {rel: self._file(rel, write=True) for rel in (files or {})}
        current: dict[str, Any] = {}
        if state is not None:
            try:
                self.head()
            except subprocess.CalledProcessError:
                pass  # the initial commit of a newly initialized repository
            else:
                current = self.state()
            state_text = json.dumps({**current, **state}, indent=1) + "\n"
        for rel, file in paths.items():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes((files or {})[rel].encode())
        if state is not None:
            self._file(STATE).write_bytes(state_text.encode())
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def steer(
        self,
        *,
        stage: str | None = None,
        status: Status | None = None,
        reason: str | None = None,
        note: str | None = None,
    ) -> str:
        """Set boundary controls and append a note, without stopping active work.

        A live hold saves the stage output then parks at its next cursor. A live
        route overrides that cursor. Only controls newer than the stage's start
        take precedence; normal application commits do not invalidate a stage.
        """
        if status is not None and status not in (
            "ready",
            "held",
            "waiting",
            "terminal",
        ):
            raise ValueError(f"unknown status: {status!r}")
        with self._write_lock():
            current = self.state()
            control = dict(current.get("_control", {}))
            version = control.get("version", 0) + 1
            control["version"] = version
            state: dict[str, Any] = {"_control": control}
            for field, value in (
                ("stage", stage),
                ("status", status),
                ("reason", reason),
            ):
                if value is not None:
                    state[field] = value
                    control[field] = {"version": version, "value": value}
            if note is not None:
                state["notes"] = [*current.get("notes", []), note]
            return self._commit("steer" + (f": {note}" if note else ""), state=state)

    def apply(
        self, transition: Transition, *, control_version: int | None = None
    ) -> str:
        """Publish a stage's output, preserving newer operator controls and notes."""
        if "_control" in transition.state:
            raise ValueError("_control is reserved for Unit.steer")
        with self._write_lock():
            current = self.state()
            state = {
                **transition.state,
                "stage": transition.stage or current.get("stage"),
                "status": transition.status,
                "outcome": transition.outcome,
                "reason": transition.summary,
            }
            if "notes" in state:
                state["notes"] = [
                    *current.get("notes", []),
                    *(
                        note
                        for note in state["notes"]
                        if note not in current.get("notes", [])
                    ),
                ]
            if control_version is not None:
                control = current.get("_control", {})
                for field in ("stage", "status", "reason"):
                    intent = control.get(field, {})
                    if intent.get("version", 0) > control_version:
                        state[field] = intent["value"]
            return self._commit(
                f"{current.get('stage')}: {transition.outcome}",
                files=transition.files,
                state=state,
            )

    def log(self, limit: int | None = None) -> list[dict[str, str]]:
        """The commits, newest first: `sha`, `at`, `message`."""
        args = ["log", "--format=%H%x1f%aI%x1f%s"]
        if limit:
            args.append(f"-{limit}")
        return [
            dict(zip(("sha", "at", "message"), line.split("\x1f")))
            for line in git(self.path, *args).splitlines()
            if line
        ]
