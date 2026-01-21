"""
Rich-based TUI for live multi-environment evaluation display.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# check for unix-specific terminal control modules
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
    """Dynamic eval state for a single env."""

    status: Literal["pending", "running", "completed", "failed"] = "pending"
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    # updated by on_progress callback
    progress: int = 0  # completed rollouts
    total: int = 0  # total rollouts
    num_examples: int = -1  # num examples (-1 means "all", updated by on_start)
    rollouts_per_example: int = 1  # rollouts per example (from config)
    reward: float = 0.0  # reward (rolling avg)
    metrics: dict[str, float] = field(default_factory=dict)  # metrics (rolling avg)
    error_rate: float = 0.0  # error rate (rolling avg)

    # path where results were saved (if save_results=true)
    save_path: Path | None = None

    # log message for special events (updated by on_log callback)
    log_message: str | None = None

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class EvalTUIState:
    """Dynamic eval state for multiple envs."""

    envs: dict[int, EnvEvalState] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def all_completed(self) -> bool:
        return all(env.status in ("completed", "failed") for env in self.envs.values())


class EvalTUI:
    def __init__(self, configs: list[EvalConfig]):
        self.state = EvalTUIState()
        self.console = Console()
        self._live: Live | None = None
        self._old_terminal_settings: list | None = None

        # store configs by index to handle duplicate env_ids
        self.configs: list[EvalConfig] = list(configs)

        # initialize env states by index
        for idx, config in enumerate(configs):
            total = config.num_examples * config.rollouts_per_example
            self.state.envs[idx] = EnvEvalState(
                total=total,
                num_examples=config.num_examples,
                rollouts_per_example=config.rollouts_per_example,
            )

    def update_env_state(
        self,
        env_idx: int,
        status: Literal["pending", "running", "completed", "failed"] | None = None,
        progress: int | None = None,
        total: int | None = None,
        num_examples: int | None = None,
        reward: float | None = None,
        metrics: dict[str, float] | None = None,
        error_rate: float | None = None,
        error: str | None = None,
        save_path: Path | None = None,
        log_message: str | None = None,
    ) -> None:
        """Update the state of a specific environment evaluation."""
        assert env_idx in self.state.envs
        env_state = self.state.envs[env_idx]

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

        if num_examples is not None:
            env_state.num_examples = num_examples

        if reward is not None:
            env_state.reward = reward

        if metrics is not None:
            env_state.metrics = metrics

        if error_rate is not None:
            env_state.error_rate = error_rate

        if error is not None:
            env_state.error = error

        if save_path is not None:
            env_state.save_path = save_path

        if log_message is not None:
            env_state.log_message = log_message

        self.refresh()

    def _get_error_rate_color(self, error_rate: float) -> str:
        """Get color for error rate: red if > 10%, otherwise default."""
        if error_rate > 0.10:
            return "red"
        return "white"

    def _make_metrics_row(
        self, reward: float, metrics: dict[str, float], error_rate: float
    ) -> Table | None:
        """Create a metrics row with metrics left-aligned and error_rate right-aligned."""
        metrics = {"reward": reward, **metrics}

        # build the left-aligned metrics text
        metrics_text = Text()
        metrics_text.append("╰─ ", style="dim")

        for i, (name, value) in enumerate(metrics.items()):
            # format value
            if isinstance(value, float):
                if value == int(value):
                    value_str = str(int(value))
                elif abs(value) < 0.01:
                    value_str = f"{value:.4f}"
                else:
                    value_str = f"{value:.3f}"
            else:
                value_str = str(value)

            # add metric with dotted leader
            metrics_text.append(name, style="dim")
            metrics_text.append(" ", style="dim")
            metrics_text.append(value_str, style="bold")

            # add separator between metrics
            if i < len(metrics) - 1:
                metrics_text.append("   ")  # 3 spaces between metrics

        # build the right-aligned error_rate text
        error_text = Text()
        if error_rate is not None:
            error_rate_str = f"{error_rate:.3f}"
            error_color = self._get_error_rate_color(error_rate)
            error_text.append("error rate ", style="dim")
            error_text.append(error_rate_str, style=f"bold {error_color}")

        # create a table with two columns for left/right alignment
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right")
        table.add_row(metrics_text, error_text)

        return table

    def _make_env_panel(self, env_idx: int) -> Panel:
        """Create a full-width panel for a single environment with config and progress."""
        config = self.configs[env_idx]
        env_state = self.state.envs[env_idx]

        # config info line
        config_line = Text()
        config_line.append(config.model, style="white")
        config_line.append(" via ", style="dim")
        config_line.append(config.client_config.api_base_url, style="white")
        config_line.append("  |  ", style="dim")
        config_line.append(str(env_state.num_examples), style="white")
        config_line.append("x", style="white")
        config_line.append(str(env_state.rollouts_per_example), style="white")
        config_line.append(" rollouts", style="dim")

        def fmt_concurrency(val: int) -> str:
            return "∞" if val == -1 else str(val)

        config_line.append("  |  ", style="dim")
        if config.max_concurrent_generation or config.max_concurrent_scoring:
            gen_concurrency = config.max_concurrent_generation or config.max_concurrent
            sem_concurrency = config.max_concurrent_scoring or config.max_concurrent
            config_line.append(fmt_concurrency(gen_concurrency), style="white")
            config_line.append(" concurrent generation", style="dim")
            config_line.append(" and ", style="dim")
            config_line.append(fmt_concurrency(sem_concurrency), style="white")
            config_line.append(" concurrent scoring", style="dim")
        else:
            config_line.append(fmt_concurrency(config.max_concurrent), style="white")
            config_line.append(" concurrent rollouts", style="dim")

        if config.sampling_args and any(config.sampling_args.values()):
            config_line.append("  |  ", style="dim")
            config_line.append("custom sampling ", style="white")
            config_line.append("(", style="dim")
            for key, value in config.sampling_args.items():
                if value is not None:
                    config_line.append(f"{key}={value}", style="dim")
            config_line.append(")", style="dim")
        if config.save_results:
            config_line.append("  |  ", style="dim")
            config_line.append("saving results", style="white")
            if config.save_every > 0:
                config_line.append(" every ", style="dim")
                config_line.append(str(config.save_every), style="white")
                config_line.append(" steps", style="dim")

        # create progress bar with timing
        # use env_state.total which gets updated by on_start callback
        total_rollouts = env_state.total
        completed_rollouts = env_state.progress  # always rollout-based
        pct = (completed_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0

        # format elapsed time
        elapsed = env_state.elapsed_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        # show "..." for total if not yet known
        total_str = "..." if total_rollouts <= 0 else str(total_rollouts)
        progress = Progress(
            SpinnerColumn() if env_state.status == "running" else TextColumn(""),
            BarColumn(bar_width=None),
            TextColumn(f"[bold]{pct:.0f}%"),
            TextColumn(f"({completed_rollouts}/{total_str} rollouts)"),
            TextColumn(f"| {time_str}"),
            console=self.console,
            expand=True,
        )
        task = progress.add_task(
            "env", total=total_rollouts, completed=completed_rollouts
        )
        progress.update(task, completed=completed_rollouts)

        # metrics display
        metrics_content = self._make_metrics_row(
            env_state.reward, env_state.metrics, env_state.error_rate
        )

        # log message for special events
        log_content = Text()
        if env_state.log_message:
            log_content.append("› ", style="dim cyan")
            log_content.append(env_state.log_message, style="dim")

        # error message if failed
        error_content = None
        if env_state.error:
            error_text = Text()
            error_text.append("ERROR: ", style="bold red")
            error_text.append(env_state.error, style="red")
            error_content = error_text

        # combine all content
        space = Text("  ")
        content_items = [config_line, space, progress]
        if metrics_content:
            content_items.append(metrics_content)
        else:
            content_items.append(space)
        content_items.append(space)
        content_items.append(log_content)
        if error_content:
            content_items.append(error_content)

        # border style based on status
        border_styles = {
            "pending": "dim",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }
        border_style = border_styles.get(env_state.status, "dim")

        # build title with env name only
        title = Text()
        title.append(config.env_id, style="bold cyan")

        return Panel(
            Group(*content_items),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(1, 1),
        )

    def _make_env_stack(self) -> Group:
        """Create a vertical stack of environment panels."""
        if not self.configs:
            return Group()

        # create panels for each environment by index
        panels = [self._make_env_panel(idx) for idx in range(len(self.configs))]

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
            Layout(name="envs", ratio=1),
            Layout(name="footer", size=3),
        )

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
            # on windows or non-tty, just wait for a simple input
            await asyncio.get_event_loop().run_in_executor(None, input)
            return

        # these imports are guaranteed to exist when HAS_TERMINAL_CONTROL is true
        import select as select_module
        import termios as termios_module
        import tty as tty_module

        # save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios_module.tcgetattr(fd)

        def drain_escape_sequence() -> None:
            """Consume remaining chars of an escape sequence (mouse events, etc)."""
            while select_module.select([sys.stdin], [], [], 0.01)[0]:
                sys.stdin.read(1)

        try:
            # use cbreak mode (not raw) - allows single char input without corrupting display
            tty_module.setcbreak(fd)

            # wait for key press in a non-blocking way
            while True:
                # small delay to keep display responsive
                await asyncio.sleep(0.1)

                # use select to check for input without blocking
                if select_module.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)

                    # handle escape sequences (mouse scroll, arrow keys, etc)
                    if char == "\x1b":
                        # check if more chars follow (escape sequence vs standalone Esc)
                        if select_module.select([sys.stdin], [], [], 0.05)[0]:
                            # escape sequence - drain it and ignore
                            drain_escape_sequence()
                            continue
                        else:
                            # standalone Escape key - exit
                            break

                    # exit on q, Q, or enter
                    if char in ("q", "Q", "\r", "\n"):
                        break
        finally:
            # restore terminal settings
            termios_module.tcsetattr(fd, termios_module.TCSADRAIN, old_settings)

    async def __aenter__(self) -> "EvalTUI":
        """Start the Live display using alternate screen mode."""
        # disable terminal echo to prevent scroll/arrow keys from displaying
        if HAS_TERMINAL_CONTROL and sys.stdin.isatty():
            import termios

            fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(fd)
            new_settings = termios.tcgetattr(fd)
            # disable echo (ECHO flag in lflags)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the Live display and restore terminal settings."""
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

        # restore terminal settings
        if self._old_terminal_settings is not None:
            import termios

            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_terminal_settings)
            self._old_terminal_settings = None

    def print_final_summary(self) -> None:
        """Print a summary after the TUI closes."""
        self.console.print()

        # summary table
        table = Table(title="Evaluation Summary")
        table.add_column("Environment", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("N", justify="center")
        table.add_column("Avg Reward", justify="center")
        table.add_column("Time", justify="center")

        for idx, config in enumerate(self.configs):
            env_state = self.state.envs[idx]
            status_styles = {
                "completed": "[green]DONE[/green]",
                "failed": "[red]FAILED[/red]",
                "running": "[yellow]RUNNING[/yellow]",
                "pending": "[dim]PENDING[/dim]",
            }
            status = status_styles.get(env_state.status, env_state.status)

            # use env_state.total for actual resolved values
            total_rollouts = env_state.total
            num_examples = total_rollouts // config.rollouts_per_example
            n = f"{num_examples}x{config.rollouts_per_example} ({total_rollouts} rollouts)"

            reward = f"{env_state.reward:.3f}"
            elapsed = env_state.elapsed_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

            table.add_row(config.env_id, status, n, reward, time_str)

        self.console.print(table)

        # print save paths if any
        saved_envs = [
            (idx, env_state)
            for idx, env_state in self.state.envs.items()
            if env_state.save_path is not None
        ]
        if saved_envs:
            self.console.print()
            self.console.print("[bold]Results saved to:[/bold]")
            for idx, env_state in saved_envs:
                self.console.print(f"  [cyan]•[/cyan] {env_state.save_path}")

        # print errors if any
        for idx, config in enumerate(self.configs):
            env_state = self.state.envs[idx]
            if env_state.error:
                self.console.print()
                self.console.print(f"[red]Error in {config.env_id}:[/red]")
                self.console.print(f"  {env_state.error}")

        self.console.print()


def is_tty() -> bool:
    """Check if stdout is a TTY (terminal)."""
    return sys.stdout.isatty()
