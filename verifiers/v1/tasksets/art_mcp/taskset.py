"""Taskset adapter for OpenPipe ART MCP-RL scenario files.

ART MCP-RL stores generated scenarios as JSON/JSONL rows with a natural-language
``task`` and optional metadata such as ``difficulty``. This module maps those
rows into native verifiers v1 tasks while preserving enough structured data to
export them back to ART-style rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

import verifiers.v1 as vf


class ArtMCPTaskData(vf.TaskData):
    source_task: str
    difficulty: int = 1
    art_metadata: dict[str, Any] = Field(default_factory=dict)


class ArtMCPTask(vf.Task[ArtMCPTaskData]):
    pass


class ArtMCPTasksetConfig(vf.TasksetConfig):
    path: str
    system_prompt: str | None = None


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)
        return

    data = json.loads(path.read_text())
    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        rows = data.get("scenarios")
        if not isinstance(rows, list):
            rows = data.get("tasks")
        if isinstance(rows, list):
            yield from rows
            return
    raise ValueError(f"unsupported ART scenario file shape: {path}")


def load_art_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = list(_read_rows(Path(path)))
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {idx} must be a JSON object")
        task = row.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"row {idx} missing non-empty 'task'")
        difficulty = row.get("difficulty", 1)
        if not isinstance(difficulty, int) or difficulty < 1:
            raise ValueError(f"row {idx} has invalid difficulty: {difficulty!r}")
    return rows


def art_rows_from_tasks(tasks: Iterable[ArtMCPTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = dict(task.data.art_metadata)
        row["task"] = task.data.source_task
        row["difficulty"] = task.data.difficulty
        rows.append(row)
    return rows


class ArtMCPTaskset(vf.Taskset[ArtMCPTask, ArtMCPTasksetConfig]):
    def load(self) -> list[ArtMCPTask]:
        return [
            ArtMCPTask(
                ArtMCPTaskData(
                    idx=i,
                    prompt=row["task"],
                    system_prompt=self.config.system_prompt,
                    source_task=row["task"],
                    difficulty=row.get("difficulty", 1),
                    art_metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"task", "difficulty"}
                    },
                ),
                self.config.task,
            )
            for i, row in enumerate(load_art_rows(self.config.path))
        ]
