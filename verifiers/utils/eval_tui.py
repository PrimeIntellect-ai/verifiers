"""
Rich-based TUI for live multi-environment evaluation display.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Literal

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
    """State for a single environment evaluation."""

    env_id: str
    model: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: int = 0  # completed rollouts
    total: int = 0  # total rollouts
    reward_sum: float = 0.0
    reward_count: int = 0
    last_log: str = ""
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def avg_reward(self) -> float | None:
        if self.reward_count == 0:
            return None
        return self.reward_sum / self.reward_count

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
    def total_progress(self) -> int:
        return sum(env.progress for env in self.envs.values())

    @property
    def total_count(self) -> int:
        return sum(env.total for env in self.envs.values())

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
        self.state = EvalTUIState()
        self.console = Console()
        self._live: Live | None = None

        # Initialize env states
        for config in configs:
            total_rollouts = config.num_examples * config.rollouts_per_example
            self.state.envs[config.env_id] = EnvEvalState(
                env_id=config.env_id,
                model=config.model,
                total=total_rollouts,
            )

    def update_env_state(
        self,
        env_id: str,
        status: Literal["pending", "running", "completed", "failed"] | None = None,
        progress: int | None = None,
        avg_reward: float | None = None,
        reward_count: int | None = None,
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

        # Handle avg_reward by converting to sum/count
        if avg_reward is not None and reward_count is not None:
            env_state.reward_sum = avg_reward * reward_count
            env_state.reward_count = reward_count
        elif avg_reward is not None and env_state.progress > 0:
            # Estimate count from progress if not provided
            env_state.reward_sum = avg_reward * env_state.progress
            env_state.reward_count = env_state.progress

        if last_log is not None:
            env_state.last_log = last_log

        if error is not None:
            env_state.error = error

    def _make_header(self) -> Panel:
        """Create the header panel with config info."""
        # Get unique models
        models = list(set(c.model for c in self.configs))
        model_str = models[0] if len(models) == 1 else f"{len(models)} models"

        header_text = Text()
        header_text.append("vf-eval", style="bold magenta")
        header_text.append(" | ")
        header_text.append("Model: ", style="dim")
        header_text.append(model_str, style="cyan")
        header_text.append(" | ")
        header_text.append(f"{len(self.configs)} environment(s)", style="green")

        return Panel(header_text, border_style="blue")

    def _make_global_progress(self) -> Panel:
        """Create the global progress bar panel."""
        total = self.state.total_count
        completed = self.state.total_progress
        elapsed = self.state.elapsed_time

        # Format elapsed time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        # Calculate percentage
        pct = (completed / total * 100) if total > 0 else 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Overall Progress"),
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

    def _make_env_panel(self, env_state: EnvEvalState) -> Panel:
        """Create a panel for a single environment."""
        content = []

        # Status indicator
        status_styles = {
            "pending": ("dim", "PENDING"),
            "running": ("yellow", "RUNNING"),
            "completed": ("green", "DONE"),
            "failed": ("red", "FAILED"),
        }
        style, label = status_styles.get(env_state.status, ("dim", "?"))

        # Header with status
        header = Text()
        header.append(env_state.env_id, style="bold")
        header.append(" [", style="dim")
        header.append(label, style=style)
        header.append("]", style="dim")
        content.append(header)

        # Progress bar for running/completed
        if env_state.status in ("running", "completed", "failed"):
            pct = (
                (env_state.progress / env_state.total * 100)
                if env_state.total > 0
                else 0
            )
            progress_text = Text()
            progress_text.append(
                f"{env_state.progress}/{env_state.total}", style="cyan"
            )
            progress_text.append(f" ({pct:.0f}%)")
            content.append(progress_text)

            # Reward info
            if env_state.avg_reward is not None:
                reward_text = Text()
                reward_text.append("reward: ", style="dim")
                reward_text.append(f"{env_state.avg_reward:.3f}", style="bold green")
                content.append(reward_text)

            # Elapsed time
            elapsed = env_state.elapsed_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
            time_text = Text()
            time_text.append("time: ", style="dim")
            time_text.append(time_str)
            content.append(time_text)

        # Error message if failed
        if env_state.error:
            error_text = Text()
            error_text.append("error: ", style="bold red")
            error_text.append(env_state.error[:50], style="red")
            if len(env_state.error) > 50:
                error_text.append("...", style="red")
            content.append(error_text)

        # Last log message
        if env_state.last_log and env_state.status == "running":
            log_text = Text()
            log_text.append(env_state.last_log[:40], style="dim italic")
            if len(env_state.last_log) > 40:
                log_text.append("...", style="dim italic")
            content.append(log_text)

        # Border style based on status
        border_styles = {
            "pending": "dim",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }
        border_style = border_styles.get(env_state.status, "dim")

        return Panel(
            Group(*content),
            border_style=border_style,
            title=env_state.model[:20]
            if len(env_state.model) > 20
            else env_state.model,
            title_align="right",
        )

    def _make_env_grid(self) -> Panel:
        """Create a grid of environment panels."""
        env_states = list(self.state.envs.values())

        if len(env_states) == 1:
            # Single environment - full width
            return self._make_env_panel(env_states[0])

        # Create a table to hold environment panels
        # Use 2-3 columns depending on count
        num_cols = min(3, len(env_states))
        table = Table.grid(expand=True, padding=(0, 1))
        for _ in range(num_cols):
            table.add_column(ratio=1)

        # Add rows
        for i in range(0, len(env_states), num_cols):
            row = []
            for j in range(num_cols):
                if i + j < len(env_states):
                    row.append(self._make_env_panel(env_states[i + j]))
                else:
                    row.append("")
            table.add_row(*row)

        return Panel(table, title="Environments", border_style="blue")

    def _make_layout(self) -> Layout:
        """Create the full TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="envs", ratio=1),
        )

        layout["header"].update(self._make_header())
        layout["progress"].update(self._make_global_progress())
        layout["envs"].update(self._make_env_grid())

        return layout

    def refresh(self) -> None:
        """Refresh the display."""
        if self._live:
            self._live.update(self._make_layout())

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

        for env_state in self.state.envs.values():
            status_styles = {
                "completed": "[green]DONE[/green]",
                "failed": "[red]FAILED[/red]",
                "running": "[yellow]RUNNING[/yellow]",
                "pending": "[dim]PENDING[/dim]",
            }
            status = status_styles.get(env_state.status, env_state.status)

            progress = f"{env_state.progress}/{env_state.total}"

            reward = (
                f"{env_state.avg_reward:.3f}"
                if env_state.avg_reward is not None
                else "-"
            )

            elapsed = env_state.elapsed_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

            table.add_row(env_state.env_id, status, progress, reward, time_str)

        self.console.print(table)

        # Print errors if any
        for env_state in self.state.envs.values():
            if env_state.error:
                self.console.print()
                self.console.print(f"[red]Error in {env_state.env_id}:[/red]")
                self.console.print(f"  {env_state.error}")


def is_tty() -> bool:
    """Check if stdout is a TTY (terminal)."""
    return sys.stdout.isatty()
