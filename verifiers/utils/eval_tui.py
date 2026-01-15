"""
Rich-based TUI for live multi-environment evaluation display.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

# Check for Unix-specific terminal control modules
try:
    import select  # noqa: F401
    import termios  # noqa: F401
    import tty  # noqa: F401

    HAS_TERMINAL_CONTROL = True
except ImportError:
    HAS_TERMINAL_CONTROL = False

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from verifiers.types import EvalConfig


@dataclass
class EnvEvalState:
    """Dynamic state for a single environment evaluation (no static config)."""

    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: int = 0  # completed groups (not rollouts)
    total: int = 0  # total groups
    metrics: dict[str, float] = field(default_factory=dict)  # running avg metrics
    last_log: str = ""
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class EvalTUIState:
    """Global state for the evaluation TUI."""

    envs: dict[str, EnvEvalState] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def all_completed(self) -> bool:
        return all(env.status in ("completed", "failed") for env in self.envs.values())


class EvalTUI:
    """Rich Live TUI for multi-environment evaluation."""

    def __init__(self, configs: list[EvalConfig]):
        self.configs = configs
        self.configs_by_id: dict[str, EvalConfig] = {c.env_id: c for c in configs}
        self.state = EvalTUIState()
        self.console = Console()
        self._live: Live | None = None

        # Initialize env states (dynamic only, no static config)
        for config in configs:
            self.state.envs[config.env_id] = EnvEvalState(
                total=config.num_examples,  # Groups, not rollouts
            )

    def update_env_state(
        self,
        env_id: str,
        status: Literal["pending", "running", "completed", "failed"] | None = None,
        progress: int | None = None,
        total: int | None = None,
        metrics: dict[str, float] | None = None,
        last_log: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update the state of a specific environment evaluation."""
        if env_id not in self.state.envs:
            return

        env_state = self.state.envs[env_id]

        if status is not None:
            env_state.status = status
            if status == "running" and env_state.start_time is None:
                env_state.start_time = time.time()
            elif status in ("completed", "failed"):
                env_state.end_time = time.time()

        if progress is not None:
            env_state.progress = progress

        if total is not None:
            env_state.total = total

        if metrics is not None:
            env_state.metrics = metrics

        if last_log is not None:
            env_state.last_log = last_log

        if error is not None:
            env_state.error = error

        # Trigger a refresh of the live display
        self.refresh()

    def _make_header(self) -> Panel:
        """Create the header panel with config info."""
        header_text = Text()
        header_text.append("vf-eval", style="bold magenta")
        header_text.append(" | ")
        header_text.append(f"{len(self.configs)} environment(s)", style="green")

        # Show status
        if self.state.all_completed:
            header_text.append(" | ")
            failed = sum(1 for e in self.state.envs.values() if e.status == "failed")
            if failed > 0:
                header_text.append(f"COMPLETE ({failed} failed)", style="bold red")
            else:
                header_text.append("COMPLETE", style="bold green")

        return Panel(header_text, border_style="blue")

    def _get_total_rollouts(self) -> int:
        """Get total rollouts across all environments."""
        return sum(c.num_examples * c.rollouts_per_example for c in self.configs)

    def _get_completed_rollouts(self) -> int:
        """Get completed rollouts across all environments."""
        total = 0
        for env_id, env_state in self.state.envs.items():
            config = self.configs_by_id.get(env_id)
            if config:
                total += env_state.progress * config.rollouts_per_example
        return total

    def _make_global_progress(self) -> Panel:
        """Create the global progress bar panel."""
        total = self._get_total_rollouts()
        completed = self._get_completed_rollouts()
        elapsed = self.state.elapsed_time

        # Format elapsed time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        # Calculate percentage
        pct = (completed / total * 100) if total > 0 else 0

        progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=40),
            TextColumn(f"[bold]{pct:.0f}%"),
            TextColumn(f"({completed}/{total})"),
            TextColumn(f"| {time_str}"),
            console=self.console,
            expand=True,
        )
        task = progress.add_task("overall", total=total, completed=completed)
        progress.update(task, completed=completed)

        return Panel(progress, border_style="green")

    def _make_metrics_display(self, metrics: dict[str, float]) -> Table | None:
        """Create a dotted-leader style metrics display."""
        if not metrics:
            return None

        # Sort metrics with 'reward' first, then alphabetically
        sorted_names = sorted(metrics.keys(), key=lambda x: (x != "reward", x))

        # Create a grid with 3 columns for metrics
        num_cols = 3
        table = Table.grid(expand=True, padding=(0, 2))
        for _ in range(num_cols):
            table.add_column(ratio=1)

        # Build metric cells with dotted leaders
        cells = []
        for name in sorted_names:
            value = metrics[name]
            # Format value
            if isinstance(value, float):
                if value == int(value):
                    value_str = str(int(value))
                elif abs(value) < 0.01:
                    value_str = f"{value:.4f}"
                else:
                    value_str = f"{value:.3f}"
            else:
                value_str = str(value)

            # Create dotted leader text
            cell = Text()
            cell.append(name, style="dim")
            # Calculate dots needed (aim for ~20 chars total per cell)
            dots_needed = max(2, 18 - len(name) - len(value_str))
            cell.append(" " + "." * dots_needed + " ", style="dim")
            cell.append(value_str, style="bold")
            cells.append(cell)

        # Add rows (3 metrics per row)
        for i in range(0, len(cells), num_cols):
            row = cells[i : i + num_cols]
            # Pad row if needed
            while len(row) < num_cols:
                row.append(Text(""))
            table.add_row(*row)

        return table

    def _make_env_panel(self, env_id: str) -> Panel:
        """Create a full-width panel for a single environment with config and progress."""
        config = self.configs_by_id[env_id]
        env_state = self.state.envs[env_id]

        # Config info line
        config_line = Text()
        config_line.append("model: ", style="dim")
        config_line.append(config.model, style="white")
        config_line.append("  |  ", style="dim")
        config_line.append("examples: ", style="dim")
        config_line.append(str(config.num_examples), style="white")
        config_line.append("  |  ", style="dim")
        config_line.append("rollouts: ", style="dim")
        config_line.append(str(config.rollouts_per_example), style="white")
        config_line.append("  |  ", style="dim")
        config_line.append("concurrency: ", style="dim")
        config_line.append(str(config.max_concurrent), style="white")

        # Create progress bar with timing
        total_rollouts = config.num_examples * config.rollouts_per_example
        completed_rollouts = env_state.progress * config.rollouts_per_example
        pct = (completed_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0

        # Format elapsed time
        elapsed = env_state.elapsed_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        progress = Progress(
            SpinnerColumn() if env_state.status == "running" else TextColumn(""),
            BarColumn(bar_width=None),
            TextColumn(f"[bold]{pct:.0f}%"),
            TextColumn(f"({completed_rollouts}/{total_rollouts} rollouts)"),
            TextColumn(f"| {time_str}"),
            console=self.console,
            expand=True,
        )
        task = progress.add_task(
            "env", total=total_rollouts, completed=completed_rollouts
        )
        progress.update(task, completed=completed_rollouts)

        # Metrics display (below progress bar)
        metrics_content = self._make_metrics_display(env_state.metrics)

        # Error message if failed
        error_content = None
        if env_state.error:
            error_text = Text()
            error_text.append("ERROR: ", style="bold red")
            error_text.append(env_state.error, style="red")
            error_content = error_text

        # Combine all content
        content_items = [config_line, progress]
        if metrics_content:
            content_items.append(metrics_content)
        if error_content:
            content_items.append(error_content)

        # Border style based on status
        border_styles = {
            "pending": "dim",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }
        border_style = border_styles.get(env_state.status, "dim")

        # Build title with env name only (status shown via border color)
        title = Text()
        title.append(env_id, style="bold cyan")

        return Panel(
            Group(*content_items),
            title=title,
            title_align="left",
            border_style=border_style,
        )

    def _make_env_stack(self) -> Group:
        """Create a vertical stack of environment panels."""
        panels = [self._make_env_panel(env_id) for env_id in self.state.envs]
        return Group(*panels)

    def _make_footer(self) -> Panel:
        """Create the footer panel with instructions."""
        if self.state.all_completed:
            footer_text = Text()
            footer_text.append("Press ", style="dim")
            footer_text.append("q", style="bold cyan")
            footer_text.append(" or ", style="dim")
            footer_text.append("Enter", style="bold cyan")
            footer_text.append(" to exit", style="dim")
            return Panel(footer_text, border_style="dim")
        else:
            footer_text = Text()
            footer_text.append("Press ", style="dim")
            footer_text.append("Ctrl+C", style="bold yellow")
            footer_text.append(" to interrupt", style="dim")
            return Panel(footer_text, border_style="dim")

    def _make_layout(self) -> Layout:
        """Create the full TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="envs", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["progress"].update(self._make_global_progress())
        layout["envs"].update(self._make_env_stack())
        layout["footer"].update(self._make_footer())

        return layout

    def refresh(self) -> None:
        """Refresh the display."""
        if self._live:
            self._live.update(self._make_layout())

    async def wait_for_exit(self) -> None:
        """Wait for user to press a key to exit."""
        if not HAS_TERMINAL_CONTROL or not sys.stdin.isatty():
            # On Windows or non-TTY, just wait for a simple input
            await asyncio.get_event_loop().run_in_executor(None, input)
            return

        # These imports are guaranteed to exist when HAS_TERMINAL_CONTROL is True
        import select as select_module
        import termios as termios_module
        import tty as tty_module

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios_module.tcgetattr(fd)

        try:
            # Use cbreak mode (not raw) - allows single char input without corrupting display
            tty_module.setcbreak(fd)

            # Wait for key press in a non-blocking way
            while True:
                # Small delay to keep display responsive
                await asyncio.sleep(0.1)

                # Use select to check for input without blocking
                if select_module.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    # Exit on q, Q, Enter, or Escape
                    if char in ("q", "Q", "\r", "\n", "\x1b"):
                        break
        finally:
            # Restore terminal settings
            termios_module.tcsetattr(fd, termios_module.TCSADRAIN, old_settings)

    async def __aenter__(self) -> "EvalTUI":
        """Start the Live display."""
        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the Live display."""
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

    def print_final_summary(self) -> None:
        """Print a summary after the TUI closes."""
        self.console.print()

        # Determine overall status
        # completed = sum(1 for e in self.state.envs.values() if e.status == "completed")
        failed = sum(1 for e in self.state.envs.values() if e.status == "failed")
        pending = sum(
            1 for e in self.state.envs.values() if e.status in ("pending", "running")
        )

        if pending > 0:
            self.console.print("[yellow bold]Evaluation Interrupted[/yellow bold]")
        elif failed > 0:
            self.console.print(
                f"[red bold]Evaluation Complete ({failed} failed)[/red bold]"
            )
        else:
            self.console.print("[green bold]Evaluation Complete[/green bold]")
        self.console.print()

        # Summary table
        table = Table(title="Results Summary")
        table.add_column("Environment", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Progress", justify="right")
        table.add_column("Avg Reward", justify="right")
        table.add_column("Time", justify="right")

        for env_id, env_state in self.state.envs.items():
            status_styles = {
                "completed": "[green]DONE[/green]",
                "failed": "[red]FAILED[/red]",
                "running": "[yellow]RUNNING[/yellow]",
                "pending": "[dim]PENDING[/dim]",
            }
            status = status_styles.get(env_state.status, env_state.status)

            progress = f"{env_state.progress}/{env_state.total}"

            reward = (
                f"{env_state.metrics.get('reward', 0):.3f}"
                if "reward" in env_state.metrics
                else "-"
            )

            elapsed = env_state.elapsed_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

            table.add_row(env_id, status, progress, reward, time_str)

        self.console.print(table)

        # Print errors if any
        for env_id, env_state in self.state.envs.items():
            if env_state.error:
                self.console.print()
                self.console.print(f"[red]Error in {env_id}:[/red]")
                self.console.print(f"  {env_state.error}")


def is_tty() -> bool:
    """Check if stdout is a TTY (terminal)."""
    return sys.stdout.isatty()
