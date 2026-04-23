"""Shared skill contract for composable tasksets and harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.abc import Traversable
from pathlib import Path

__all__ = ["TaskSkills"]


@dataclass
class TaskSkills:
    """Task-provided skills after sandbox-side preparation.

    ``source_dir`` is a local package/resource directory to upload. ``skills_dir``
    is the resolved sandbox path the harness should register with the agent.
    """

    source_dir: Traversable | Path | None = None
    skills_dir: str | None = None

    @property
    def has_skills(self) -> bool:
        return bool(self.source_dir or self.skills_dir)
