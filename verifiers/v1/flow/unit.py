"""A unit: one git repository whose head is where the unit stands.

`state.json` at the head carries the three fields the loop reads -- `stage`, `status`,
`reason` -- and whatever else the pipeline's stages keep there (credits, a version, a
pin). Every stage ends in one commit: its `Transition`, applied to the state, with the files
the stage wants kept beside it. An operator steers the same way, by committing an edit.
"""

from __future__ import annotations

import json
import subprocess
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
    """One repository. Reads come from the working tree, which is always the head:
    only `commit` writes, and it commits everything it writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.id = path.name

    @classmethod
    def create(
        cls, path: Path, state: dict[str, Any], files: dict[str, str] | None = None
    ) -> Self:
        """A new unit at `path`, or the one already there."""
        unit = cls(path)
        if not (path / ".git").exists():
            path.mkdir(parents=True, exist_ok=True)
            git(path, "init", "-q")
            unit.commit("init", files=files, state={"status": "ready", **state})
        return unit

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")

    def read(self, rel: str, sha: str | None = None) -> str | None:
        if sha is None:
            file = self.path / rel
            return file.read_text() if file.exists() else None
        try:
            return git(self.path, "show", f"{sha}:{rel}")
        except subprocess.CalledProcessError:
            return None

    def read_json(self, rel: str, sha: str | None = None) -> dict[str, Any]:
        raw = self.read(rel, sha)
        return json.loads(raw) if raw else {}

    def state(self) -> dict[str, Any]:
        return self.read_json(STATE)

    def commit(
        self,
        message: str,
        *,
        files: dict[str, str] | None = None,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Write `files` and merge `state` into `state.json`, as one commit; the sha."""
        for rel, text in (files or {}).items():
            (self.path / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.path / rel).write_text(text)
        if state is not None:
            merged = {**self.state(), **state}
            (self.path / STATE).write_text(json.dumps(merged, indent=1) + "\n")
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def apply(self, transition: Transition) -> str:
        """A stage's transition as the next commit: the stage the unit moves to (or stays
        at), its status, the outcome and summary as `reason`."""
        current = self.state()
        state = {
            **transition.state,
            "stage": transition.stage or current.get("stage"),
            "status": transition.status,
            "outcome": transition.outcome,
            "reason": transition.summary,
        }
        return self.commit(
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
