"""
Rich-based display for live multi-environment evaluation.

Provides a visual progress display that works in two modes:
- Default (screen=False): Rich panels refresh in-place without screen hijacking
- TUI mode (screen=True): Alternate screen buffer with echo handling
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from verifiers.types import EvalConfig, GenerateOutputs, TokenUsage
from verifiers.utils.display_utils import BaseDisplay, format_numeric, make_aligned_row
from verifiers.utils.message_utils import format_messages


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
    usage: TokenUsage | None = None
    error_rate: float = 0.0  # error rate (rolling avg)

    # path where results were saved (if save_results=true)
    save_path: Path | None = None

    # log message for special events (updated by on_log callback)
    log_message: str | None = None

    # full results (stored after completion for summary)
    results: GenerateOutputs | None = None

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def _make_histogram(values: list[float], bins: int = 10, height: int = 8) -> Text:
    """Create a simple vertical text histogram of values."""
    if not values:
        return Text("no data", style="dim")

    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return Text(f"all values = {min_val:.2f}", style="dim")

    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    for v in values:
        bin_idx = min(int((v - min_val) / bin_width), bins - 1)
        counts[bin_idx] += 1

    max_count = max(counts)
    scaled = [
        int(round((c / max_count) * height)) if max_count > 0 else 0 for c in counts
    ]

    label_width = max(
        4,
        len(f"{min_val:.2f}"),
        len(f"{max_val:.2f}"),  # keep labels aligned
    )
    count_width = max(len(str(c)) for c in counts)
    col_width = max(label_width, count_width)
    spacer = " "
    bar_on = "█" * col_width
    bar_off = "░" * col_width

    out = Text()
    # Counts (top row)
    for i, count in enumerate(counts):
        out.append(str(count).center(col_width), style="dim")
        if i < bins - 1:
            out.append(spacer)
    out.append("\n")

    # Bars (top to bottom)
    for row in range(height, 0, -1):
        for i, h in enumerate(scaled):
            if h >= row:
                out.append(bar_on, style="cyan")
            else:
                out.append(bar_off, style="dim")
            if i < bins - 1:
                out.append(spacer)
        out.append("\n")

    # Baseline
    out.append("─" * (bins * col_width + (bins - 1)), style="dim")
    out.append("\n")

    # Bin labels (start values)
    for i in range(bins):
        bin_start = min_val + i * bin_width
        label = f"{bin_start:.2f}".center(col_width)
        out.append(label, style="dim")
        if i < bins - 1:
            out.append(spacer)

    return out


@dataclass
class EvalDisplayState:
    """Dynamic eval state for multiple envs."""

    envs: dict[int, EnvEvalState] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def all_completed(self) -> bool:
        return all(env.status in ("completed", "failed") for env in self.envs.values())


class EvalDisplay(BaseDisplay):
    """
    Rich-based display for multi-environment evaluation.

    Args:
        configs: List of EvalConfig objects for the environments being evaluated.
        screen: If True, use alternate screen buffer (TUI mode via --tui flag).
                If False (default), refresh in-place without screen hijacking.
    """

    def __init__(self, configs: list[EvalConfig], screen: bool = False) -> None:
        super().__init__(screen=screen, refresh_per_second=4)
        self.state = EvalDisplayState()

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

    @staticmethod
    def _display_max_concurrent(config: EvalConfig, total_rollouts: int) -> int:
        """Return rollout-level concurrency shown in the UI."""
        display_rollout_concurrency = config.max_concurrent
        if (
            not config.independent_scoring
            and config.max_concurrent > 0
            and config.rollouts_per_example > 1
        ):
            max_group_concurrency = math.ceil(
                config.max_concurrent / config.rollouts_per_example
            )
            display_rollout_concurrency = (
                max_group_concurrency * config.rollouts_per_example
            )

        if display_rollout_concurrency > 0 and total_rollouts > 0:
            return min(display_rollout_concurrency, total_rollouts)

        return display_rollout_concurrency

    def update_env_state(
        self,
        env_idx: int,
        status: Literal["pending", "running", "completed", "failed"] | None = None,
        progress: int | None = None,
        total: int | None = None,
        num_examples: int | None = None,
        reward: float | None = None,
        metrics: dict[str, float] | None = None,
        usage: TokenUsage | None = None,
        error_rate: float | None = None,
        error: str | None = None,
        save_path: Path | None = None,
        log_message: str | None = None,
        results: GenerateOutputs | None = None,
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

        if usage is not None:
            env_state.usage = usage

        if error_rate is not None:
            env_state.error_rate = error_rate

        if error is not None:
            env_state.error = error

        if save_path is not None:
            env_state.save_path = save_path

        if log_message is not None:
            env_state.log_message = log_message

        if results is not None:
            env_state.results = results

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
            value_str = format_numeric(value)

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

        return make_aligned_row(metrics_text, error_text)

    def _make_tokens_row(self, usage: TokenUsage) -> Table | None:
        """Create a tokens row with input/output values."""
        tokens_text = Text()
        tokens_text.append("╰─ ", style="dim")
        token_items = [
            ("input", usage.get("input_tokens", 0.0)),
            ("output", usage.get("output_tokens", 0.0)),
        ]
        for i, (name, value) in enumerate(token_items):
            value_str = format_numeric(value)
            tokens_text.append(name, style="dim")
            tokens_text.append(" ", style="dim")
            tokens_text.append(value_str, style="bold")
            if i < len(token_items) - 1:
                tokens_text.append("   ")
        return make_aligned_row(tokens_text, Text())

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

        display_max_concurrent = self._display_max_concurrent(config, env_state.total)
        config_line.append("  |  ", style="dim")
        config_line.append(fmt_concurrency(display_max_concurrent), style="white")
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
        tokens_content = (
            self._make_tokens_row(env_state.usage)
            if env_state.usage is not None
            else None
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
        if tokens_content:
            content_items.append(tokens_content)
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
            expand=True,
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
            if self.screen:
                # TUI mode - show exit instructions
                footer_text = Text()
                footer_text.append("Press ", style="dim")
                footer_text.append("q", style="bold cyan")
                footer_text.append(" or ", style="dim")
                footer_text.append("Enter", style="bold cyan")
                footer_text.append(" to exit", style="dim")
            else:
                # Normal mode - no exit prompt needed
                footer_text = Text()
                footer_text.append("Evaluation complete", style="dim")
            return Panel(footer_text, border_style="dim")
        else:
            if self.screen:
                # TUI mode - show interrupt instructions
                footer_text = Text()
                footer_text.append("Press ", style="dim")
                footer_text.append("Ctrl+C", style="bold yellow")
                footer_text.append(" to interrupt", style="dim")
            else:
                # Normal mode - show running status
                footer_text = Text()
                footer_text.append("Running...", style="dim")
            return Panel(footer_text, border_style="dim")

    def _render(self) -> Group:
        """Create the full display."""
        items: list[Group | Panel] = [self._make_env_stack()]

        # Always show log panel (with placeholder lines if no logs)
        items.append(self._make_log_panel())

        # Only show footer in TUI mode
        if self.screen:
            items.append(self._make_footer())

        return Group(*items)

    def print_final_summary(self) -> None:
        """Print a comprehensive summary after the display closes."""
        self.console.print()

        # Per-environment detailed sections
        for idx, config in enumerate(self.configs):
            env_state = self.state.envs[idx]
            results = env_state.results

            if results is None:
                continue

            self.console.print()
            self.console.print(
                Panel(
                    self._make_env_detail(config, env_state, results),
                    title=f"[bold blue]{config.env_id}[/bold blue]",
                    border_style="dim",
                )
            )

        # Print save paths if any
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

        # Print errors if any
        for idx, config in enumerate(self.configs):
            env_state = self.state.envs[idx]
            if env_state.error:
                self.console.print()
                self.console.print(f"[red]error in {config.env_id}:[/red]")
                self.console.print(f"  {env_state.error}")

        # Summary table with main metrics (printed last)
        table = Table(title="Evaluation Summary")
        table.add_column("env_id", style="cyan")
        table.add_column("status", justify="center")
        table.add_column("examples", justify="center")
        table.add_column("rollouts", justify="center")
        table.add_column("reward", justify="center")
        show_usage = any(
            env_state.usage is not None
            or (
                env_state.results is not None
                and env_state.results["metadata"].get("usage") is not None
            )
            for env_state in self.state.envs.values()
        )
        if show_usage:
            table.add_column("input", justify="center")
            table.add_column("output", justify="center")
        table.add_column("errors", justify="center")
        table.add_column("time", justify="center")

        for idx, config in enumerate(self.configs):
            env_state = self.state.envs[idx]
            status_styles = {
                "completed": "[green]done[/green]",
                "failed": "[red]failed[/red]",
                "running": "[yellow]running[/yellow]",
                "pending": "[dim]pending[/dim]",
            }
            status = status_styles.get(env_state.status, env_state.status)

            # use env_state.total for actual resolved values
            total_rollouts = env_state.total
            num_examples = total_rollouts // config.rollouts_per_example
            examples_str = str(num_examples)
            rollouts_str = str(config.rollouts_per_example)

            reward = f"{env_state.reward:.3f}"
            input_tokens = None
            output_tokens = None
            usage = None
            if env_state.results is not None:
                usage = env_state.results["metadata"].get("usage")
            else:
                usage = env_state.usage
            if usage is not None:
                input_tokens = format_numeric(usage.get("input_tokens", 0.0))
                output_tokens = format_numeric(usage.get("output_tokens", 0.0))

            # error rate with color coding
            error_rate = env_state.error_rate
            if error_rate > 0.10:
                error_str = f"[red]{error_rate:.1%}[/red]"
            elif error_rate > 0:
                error_str = f"[yellow]{error_rate:.1%}[/yellow]"
            else:
                error_str = f"[green]{error_rate:.1%}[/green]"

            elapsed = env_state.elapsed_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

            row = [config.env_id, status, examples_str, rollouts_str, reward]
            if show_usage:
                row.extend([input_tokens or "-", output_tokens or "-"])
            row.extend([error_str, time_str])
            table.add_row(*row)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _make_env_detail(
        self, config: EvalConfig, env_state: EnvEvalState, results: GenerateOutputs
    ) -> Group:
        """Create detailed content for a single environment's summary."""
        items: list[Panel] = []

        # Example 0 prompt/completion (already in printable format from state_to_output)
        outputs = results["outputs"]
        if outputs and outputs[0]["prompt"] and outputs[0]["completion"]:
            prompt = outputs[0]["prompt"]
            completion = outputs[0]["completion"]
            reward_0 = outputs[0]["reward"] if outputs[0]["reward"] else 0.0
            error_0 = outputs[0].get("error") if outputs[0] else None

            # Prompt panel
            items.append(
                Panel(
                    format_messages(prompt),
                    title="[dim]example 0 — prompt[/dim]",
                    border_style="dim",
                )
            )

            # Completion panel (with error if any)
            completion_text = format_messages(completion)
            if error_0 is not None:
                completion_text.append("\n\nerror: ", style="bold red")
                completion_text.append(error_0, style="bold red")
            completion_text.append("\n\nreward: ", style="bold cyan")
            completion_text.append(f"{reward_0:.3f}", style="bold cyan")

            items.append(
                Panel(
                    completion_text,
                    title="[dim]example 0 — completion[/dim]",
                    border_style="dim",
                )
            )

        # Reward distribution
        rewards = [o["reward"] for o in outputs]
        if rewards:
            # All rollouts histogram
            all_rollouts_content = Group(
                Text("all rollouts:", style="bold"),
                _make_histogram(rewards, bins=8, height=8),
            )

            # Per-example averages if multiple rollouts
            rollouts_per = config.rollouts_per_example
            if rollouts_per > 1 and len(rewards) >= rollouts_per:
                num_examples = len(rewards) // rollouts_per
                example_avgs = []
                for i in range(num_examples):
                    example_rewards = rewards[i * rollouts_per : (i + 1) * rollouts_per]
                    example_avgs.append(sum(example_rewards) / len(example_rewards))

                per_example_content = Group(
                    Text("per-example avg:", style="bold"),
                    _make_histogram(example_avgs, bins=8, height=8),
                )

                # Side by side
                reward_display = Columns(
                    [all_rollouts_content, per_example_content],
                    equal=True,
                    expand=True,
                )
            else:
                reward_display = all_rollouts_content

            items.append(
                Panel(
                    reward_display,
                    title="[dim]reward distribution[/dim]",
                    border_style="dim",
                )
            )

        # Metrics
        if env_state.metrics:
            metrics_text = Text()
            for name, value in env_state.metrics.items():
                value_str = format_numeric(value)
                metrics_text.append(f"• {name}: ", style="cyan")
                metrics_text.append(f"{value_str}\n")

            items.append(
                Panel(
                    metrics_text,
                    title="[dim]metrics (avg)[/dim]",
                    border_style="dim",
                )
            )

        usage = results["metadata"].get("usage")
        if usage is not None:
            tokens_text = Text()
            for name, value in usage.items():
                value_str = (
                    format_numeric(value)
                    if isinstance(value, (int, float, str))
                    else str(value)
                )
                label = name.replace("_", " ")
                tokens_text.append(f"• {label}: ", style="cyan")
                tokens_text.append(f"{value_str}\n")
            items.append(
                Panel(
                    tokens_text,
                    title="[dim]usage (avg)[/dim]",
                    border_style="dim",
                )
            )

        return Group(*items)


# Re-export is_tty for convenience
from verifiers.utils.display_utils import is_tty  # noqa: E402, F401
