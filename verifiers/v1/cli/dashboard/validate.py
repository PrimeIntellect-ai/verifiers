"""The validate `--rich` dashboard: a taskset overview, a progress bar, and one row per task.

The model-free counterpart of the eval dashboard — no rollout phases, tokens, turns, or
reward, just each task's validation outcome: pending ○ / running ● / valid ✓ / invalid ✗ /
error ✗ / timeout ⏱. The runner advances a `TaskProgress` per task; this reads them each tick.
"""

import contextlib
import time
from dataclasses import dataclass

from rich.console import Group
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.configs.validate import ValidateConfig
from verifiers.v1.utils import format_time

_STYLE = {
    "pending": "dim",
    "running": "cyan",
    "valid": "green",
    "invalid": "yellow",
    "error": "red",
    "timeout": "red",
}
_MARK = {
    "pending": "○",
    "running": "●",
    "valid": "✓",
    "invalid": "✗",
    "error": "✗",
    "timeout": "⏱",
}
_DONE = ("valid", "invalid", "error", "timeout")


@dataclass
class TaskProgress:
    """Live state of one task's validation, read by the dashboard each tick and advanced by the
    runner (pending → running → its outcome)."""

    idx: int
    name: str | None
    state: str = "pending"
    start: float | None = None
    end: float | None = None


def _overview(config: ValidateConfig) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("taskset", f"{config.taskset.name}  ·  {config.runtime.type} runtime")
    return grid


def _progress(states: list[TaskProgress], start: float) -> Table:
    done = [s for s in states if s.state in _DONE]
    valid = sum(1 for s in done if s.state == "valid")
    stats = (
        f" {len(done)}/{len(states)} · {format_time(time.time() - start)} · "
        f"valid {valid} · invalid {len(done) - valid}"
    )
    row = Table.grid()
    row.add_column()
    row.add_column()
    row.add_row(
        ProgressBar(total=len(states) or 1, completed=len(done), width=32), Text(stats)
    )
    return row


def _rows(states: list[TaskProgress], now: float) -> Table:
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
        _overview(config),
        _progress(states, start),
        Rule(style="dim"),
        _rows(states, time.time()),
    )


@contextlib.asynccontextmanager
async def validate_dashboard(
    states: list[TaskProgress], config: ValidateConfig, start: float
):
    """Refresh the live validate view until the `with` block exits, then a final frame."""
    async with live_view(lambda: _render(states, config, start)):
        yield
