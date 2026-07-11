"""Live replay dashboard."""

import contextlib
import time
from dataclasses import dataclass

from rich.console import Group
from rich.markup import escape
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.cli.dashboard.validate import TaskProgress
from verifiers.v1.utils.format import format_time


@dataclass
class ReplayProgress(TaskProgress):
    detail: str = ""


# Colors mirror the eval dashboard for shared states: pending=dim, running=cyan, scored=green
# (eval's "success"), error=red. `skipped` is replay-only (a rollout that errored → copied through).
_STYLE = {
    "pending": "dim",
    "running": "cyan",
    "scored": "green",
    "skipped": "blue",
    "error": "red",
}
_MARK_WIDTH = max(len(state) for state in _STYLE)
_MARK = {state: escape(f"[{state:<{_MARK_WIDTH}}]") for state in _STYLE}
_DONE = ("scored", "error", "skipped")


def _render(
    states: list[TaskProgress], taskset_name: str, source: str, out: str, start: float
) -> Group:
    done = [s for s in states if s.state in _DONE]
    scored = sum(1 for s in done if s.state == "scored")
    skipped = sum(1 for s in done if s.state == "skipped")
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="dim")
    overview.add_column()
    overview.add_row("replay", f"{taskset_name}  ·  {source}")
    overview.add_row("output", out)

    stats = (
        f" {len(done)}/{len(states)} · {format_time(time.time() - start)} · "
        f"scored {scored} · failed {len(done) - scored - skipped} · skipped {skipped}"
    )
    progress = Table.grid()
    progress.add_column()
    progress.add_column()
    progress.add_row(
        ProgressBar(total=len(states) or 1, completed=len(done), width=32), Text(stats)
    )

    now = time.time()
    rows = Table.grid(expand=True, padding=(0, 1))
    rows.add_column(ratio=1, no_wrap=True)
    rows.add_column(justify="right", no_wrap=True)
    rows.add_column(justify="right", no_wrap=True)
    for s in states:
        if s.state == "pending":  # show only in-flight/done rows (like eval/validate)
            continue
        label = f"name={s.name[:40]}" if s.name else f"idx={s.idx}"
        parts = [
            p
            for p in (s.state if s.state in _DONE else "", getattr(s, "detail", ""))
            if p
        ]
        result = " ".join(parts) + " ·" if parts else ""
        elapsed = format_time((s.end or now) - s.start) if s.start else ""
        rows.add_row(
            f"{_MARK[s.state]} {label}",
            result,
            elapsed,
            style=_STYLE[s.state],
        )
    return Group(overview, progress, Rule(style="dim"), rows)


@contextlib.asynccontextmanager
async def replay_dashboard(
    states: list[TaskProgress], taskset_name: str, source: str, out: str, start: float
):
    """Refresh the live replay view until the `with` block exits, then a final frame."""
    async with live_view(lambda: _render(states, taskset_name, source, out, start)):
        yield
