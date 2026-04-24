from __future__ import annotations

from collections.abc import Iterable

from verifiers.envs.experimental.taskset import Source, Taskset
from verifiers.rubrics.rubric import Rubric


class DatasetTaskset(Taskset):
    """Thin named wrapper around a lazy task source."""

    def __init__(
        self,
        source: Source,
        eval_source: Source = None,
        rubric: Rubric | None = None,
        tools: Iterable[object] | None = None,
        name: str | None = None,
    ):
        super().__init__(
            source=source,
            eval_source=eval_source,
            rubric=rubric,
            tools=tools,
            name=name,
        )
