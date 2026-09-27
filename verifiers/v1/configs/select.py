"""Which of a taskset's tasks a run uses: `Taskset.select(SelectConfig)`, under
`--select.*` on the eval, debug, validate and GEPA CLIs (`SelectCLIConfig`)."""

import re

from pydantic import AliasChoices, Field, field_validator
from pydantic_config import BaseConfig

IDX_RANGE = re.compile(r"(\d*):(\d*)")


class TaskMatchConfig(BaseConfig):
    """Tasks named by position or identity; a task matches when any list names it."""

    idx: list[int | str] = Field(default_factory=list)
    """Positions in the taskset's `load()` stream (`TaskData.idx`): ints and half-open
    `start:stop` ranges with optional ends (`"100:"`, `":50"`). A comma-separated
    string also works (`"0:10,17"`)."""
    ids: list[str] = Field(default_factory=list)
    """`TaskData.id` values."""
    keys: list[str] = Field(default_factory=list)
    """`Task.key` values, as recorded on each trace (`task.key`)."""
    names: list[str] = Field(default_factory=list)
    """`TaskData.name` values."""

    @field_validator("idx", mode="before")
    @classmethod
    def _parse_idx(cls, value):
        if isinstance(value, (int, str)):
            value = [value]
        items: list[int | str] = []
        for item in value:
            parts = item.split(",") if isinstance(item, str) else [item]
            items.extend(_parse_idx_item(part) for part in parts)
        return items

    @property
    def empty(self) -> bool:
        return not (self.idx or self.ids or self.keys or self.names)

    def idx_ranges(self) -> list[tuple[int, int | None]]:
        """`idx` as half-open `(start, stop)` ranges; `stop` is None for an open end."""
        ranges: list[tuple[int, int | None]] = []
        for item in self.idx:
            if isinstance(item, int):
                ranges.append((item, item + 1))
            else:
                start, stop = item.split(":")
                ranges.append((int(start or 0), int(stop) if stop else None))
        return ranges

    @property
    def idx_stop(self) -> int | None:
        """One past the last position this can match, when it names only closed `idx`
        ranges; None when a match could lie anywhere in the stream."""
        ranges = self.idx_ranges()
        if not ranges or self.ids or self.keys or self.names:
            return None
        if any(stop is None for _, stop in ranges):
            return None
        return max(stop for _, stop in ranges if stop is not None)


def _parse_idx_item(item: int | str) -> int | str:
    if isinstance(item, int):
        if item < 0:
            raise ValueError(f"idx {item} is negative")
        return item
    text = item.strip()
    if text.isdigit():
        return int(text)
    match = IDX_RANGE.fullmatch(text)
    if match is None:
        raise ValueError(f"idx {item!r} is not an int or a 'start:stop' range")
    start, stop = match.groups()
    if start and stop and int(start) >= int(stop):
        raise ValueError(f"idx range {item!r} is empty")
    return text


class SelectConfig(BaseConfig):
    """The steps apply in a fixed order: `include` keeps tasks, `exclude` drops them,
    then `shuffle`, `skip` and `limit` pick from what is left."""

    include: TaskMatchConfig = TaskMatchConfig()
    """Keep only the tasks this names. Empty keeps all."""
    exclude: TaskMatchConfig = TaskMatchConfig()
    """Drop the tasks this names."""
    shuffle: bool = False
    """Shuffle the kept tasks under `seed`. Needs a finite stream: an infinite taskset
    must be bounded by closed `include.idx` ranges first."""
    seed: int = 0
    """Seed for `shuffle`, fixed so runs select the same tasks."""
    skip: int = Field(0, ge=0)
    """Drop this many tasks after the shuffle."""
    limit: int | None = Field(None, ge=1)
    """Take at most this many tasks after `skip` (None = all)."""


class SelectCLIConfig(SelectConfig):
    """`SelectConfig` with CLI short flags, for the one `select` block of an entrypoint:
    `-n` sets `limit` and `-s` sets `shuffle`."""

    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle the kept tasks under `seed` (`-s`). Needs a finite stream: an infinite
    taskset must be bounded by closed `include.idx` ranges first."""
    limit: int | None = Field(None, ge=1, validation_alias=AliasChoices("limit", "n"))
    """Take at most this many tasks after `skip` (`-n`; None = all)."""
