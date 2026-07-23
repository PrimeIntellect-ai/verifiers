"""Live task-validation dashboard."""

import contextlib
import time
from collections import Counter
from dataclasses import dataclass

from rich.console import Group
from rich.markup import escape
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.configs.validate import ValidateConfig
from verifiers.v1.utils.format import format_time

_STYLE = {
    "pending": "dim",
    "running": "cyan",
    "valid": "green",
    "invalid": "yellow",
    "error": "red",
    "timeout": "red",
}
_MARK_WIDTH = max(len(state) for state in _STYLE)
# Each state name padded to a common width and bracketed, so the `[ ]` line up in a column with
# the name left-aligned inside — the outcome reads at a glance. `escape` keeps the brackets
# literal: Rich parses `[name]` in a cell as markup and would otherwise drop it.
_MARK = {state: escape(f"[{state:<{_MARK_WIDTH}}]") for state in _STYLE}
_DONE = ("valid", "invalid", "error", "timeout")
# State -> (visible label, color); insertion order is the summary order.
_OUTCOMES = {state: (state, _STYLE[state]) for state in _DONE}


@dataclass
class TaskProgress:
    idx: int
    name: str | None
    state: str = "pending"
    start: float | None = None
    end: float | None = None


def Overview(config: ValidateConfig) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("taskset", f"{config.taskset.name}  ·  {config.runtime.type} runtime")
    return grid


def Progress(
    states: list[TaskProgress],
    start: float,
    outcomes: dict[str, tuple[str, str]] = _OUTCOMES,
) -> Table:
    done = [s for s in states if s.state in outcomes]
    counts = Counter(s.state for s in done)
    stats = Text(f" {len(done)}/{len(states)} · {format_time(time.time() - start)}")
    for state, (label, style) in outcomes.items():
        stats.append(" · ")
        stats.append(f"{label} {counts[state]}", style=style)
    row = Table.grid()
    row.add_column()
    row.add_column()
    row.add_row(
        ProgressBar(total=len(states) or 1, completed=len(done), width=32), stats
    )
    return row


def Rows(states: list[TaskProgress], now: float) -> Table:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1, no_wrap=True)  # mark + task label
    grid.add_column(justify="right", no_wrap=True)  # outcome
    grid.add_column(justify="right", no_wrap=True)  # time
    for s in states:
        if (
            s.state == "pending"
        ):  # not started — like eval, show only in-flight/done rows
            continue
        label = f"name={s.name[:40]}" if s.name else f"idx={s.idx}"
        result = s.state if s.state in _DONE else ""
        elapsed = format_time((s.end or now) - s.start) if s.start else ""
        grid.add_row(
            f"{_MARK[s.state]} task {label}",
            f"{result} ·" if result else "",
            elapsed,
            style=_STYLE[s.state],
        )
    return grid


def _render(states: list[TaskProgress], config: ValidateConfig, start: float) -> Group:
    return Group(
        Overview(config),
        Progress(states, start),
        Rule(style="dim"),
        Rows(states, time.time()),
    )


@contextlib.asynccontextmanager
async def validate_dashboard(
    states: list[TaskProgress], config: ValidateConfig, start: float
):
    """Refresh the live validate view until the `with` block exits, then a final frame."""
    async with live_view(lambda: _render(states, config, start)):
        yield
