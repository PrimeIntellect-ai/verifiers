"""
Rich-based display for GEPA optimization.

Shows:
1. Budget progress bar (metric calls used)
2. Current phase/step indicator
3. Per-valset-row pareto frontier (best score for each row) - only from full valset evals

Supports two modes:
- Default (screen=False): Rich panels refresh in-place without screen hijacking
- TUI mode (screen=True): Alternate screen buffer with echo handling
"""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from verifiers.utils.display_utils import BaseDisplay, make_aligned_row


@dataclass
class ValsetRowState:
    """Tracks best score for a single valset row."""

    best_score: float = 0.0
    best_candidate_idxs: list[int] = field(default_factory=list)


@dataclass
class GEPADisplayState:
    """Minimal state for display."""

    max_metric_calls: int = 500
    metric_calls_used: int = 0
    iteration: int = 0
    phase: str = "initializing"
    perfect_score: float | None = None
    completed: bool = False

    # Minibatch tracking
    minibatch_before: float | None = None  # Score before reflection
    minibatch_after: float | None = None  # Score after reflection
    minibatch_accepted: bool | None = None
    minibatch_skipped: bool = False

    # Per-valset-row tracking (only from full valset evals)
    valset_rows: dict[int, ValsetRowState] = field(default_factory=dict)
    num_valset_evals: int = 0


class GEPADisplay(BaseDisplay):
    """
    Rich-based display for GEPA optimization.

    Implements GEPA's LoggerProtocol (log method).
    Call update_eval() from adapter to track progress.

    Args:
        max_metric_calls: Maximum number of metric calls (budget).
        valset_size: Size of the validation set.
        valset_example_ids: Optional list of example IDs in the validation set.
        log_file: Optional path to write log messages.
        perfect_score: Optional perfect score threshold for skipping reflection.
        screen: If True, use alternate screen buffer (TUI mode via --tui flag).
                If False (default), refresh in-place without screen hijacking.
    """

    def __init__(
        self,
        env_id: str,
        model: str,
        reflection_model: str,
        max_metric_calls: int = 500,
        num_train: int | None = None,
        num_val: int | None = None,
        valset_size: int = 50,
        valset_example_ids: list[int] | None = None,
        log_file: str | Path | None = None,
        perfect_score: float | None = None,
        screen: bool = False,
    ) -> None:
        super().__init__(screen=screen, refresh_per_second=4)
        self.state = GEPADisplayState(
            max_metric_calls=max_metric_calls,
            perfect_score=perfect_score,
        )

        # Config metadata (known at start)
        self.env_id = env_id
        self.model = model
        self.reflection_model = reflection_model
        self.num_train = num_train
        self.num_val = num_val

        # Valset info (updated after env loads)
        self.valset_size = valset_size
        self.valset_example_ids: set[int] | None = (
            set(valset_example_ids) if valset_example_ids else None
        )
        self.log_file = Path(log_file) if log_file else None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def get_log_hint(self) -> Text | None:
        return None

    def set_valset_info(self, valset_size: int, valset_example_ids: list[int]) -> None:
        """Update valset info after environment is loaded."""
        self.valset_size = valset_size
        self.valset_example_ids = (
            set(valset_example_ids) if valset_example_ids else None
        )

    def start(self) -> None:
        """Start the live display."""
        super().start()

    def stop(self) -> None:
        """Stop the live display."""
        self.state.completed = True
        super().stop()
        self._print_final_summary()

    def log(self, message: str) -> None:
        """LoggerProtocol - receives GEPA log messages."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

        # Parse phase from messages
        if "Base program full valset score" in message:
            self.state.phase = "initial valset done"
        elif "Selected program" in message:
            if "Iteration" in message:
                try:
                    self.state.iteration = int(
                        message.split("Iteration")[1].split(":")[0].strip()
                    )
                except (ValueError, IndexError):
                    pass
            self.state.phase = "selecting"
        elif "Proposed new text" in message:
            self.state.phase = "re-evaluating"
        # Note: accepted/rejected phase is now set in update_eval()

        self.refresh()

    def update_eval(
        self,
        candidate_idx: int,
        scores: list[float],
        example_ids: list[int],
        capture_traces: bool = False,
    ) -> None:
        """
        Called by adapter after each evaluation to update progress.

        Args:
            candidate_idx: Which candidate was evaluated
            scores: Scores for each example
            example_ids: Which valset rows were evaluated
            capture_traces: True if this is a pre-reflection eval (baseline)
        """
        # Update budget
        self.state.metric_calls_used += len(scores)

        # Check if this is a valset eval by matching example_ids
        if self.valset_example_ids is not None:
            is_valset_eval = set(example_ids) == self.valset_example_ids
        else:
            is_valset_eval = len(scores) == self.valset_size

        if is_valset_eval:
            # Full valset evaluation - update frontier
            self.state.num_valset_evals += 1
            for example_id, score in zip(example_ids, scores):
                row = self.state.valset_rows.get(example_id)
                if row is None:
                    self.state.valset_rows[example_id] = ValsetRowState(
                        best_score=score,
                        best_candidate_idxs=[candidate_idx],
                    )
                elif score > row.best_score:
                    # New best - replace
                    row.best_score = score
                    row.best_candidate_idxs = [candidate_idx]
                elif (
                    score == row.best_score
                    and candidate_idx not in row.best_candidate_idxs
                ):
                    # Tie - add to list
                    row.best_candidate_idxs.append(candidate_idx)
        else:
            # Minibatch evaluation - track before/after for status display
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if capture_traces:
                # This is the baseline eval before reflection
                self.state.minibatch_before = avg_score
                self.state.minibatch_after = None
                self.state.minibatch_accepted = None

                # Detect skip based on perfect score
                if (
                    self.state.perfect_score is not None
                    and avg_score >= self.state.perfect_score
                ):
                    self.state.minibatch_skipped = True
                else:
                    self.state.minibatch_skipped = False
                    # Baseline done, now waiting for teacher LLM to reflect
                    self.state.phase = "reflecting"
            else:
                # This is the eval after reflection
                self.state.minibatch_after = avg_score
                if self.state.minibatch_before is not None:
                    self.state.minibatch_accepted = (
                        avg_score > self.state.minibatch_before
                    )
                    # Set phase based on result
                    self.state.phase = (
                        "accepted" if self.state.minibatch_accepted else "rejected"
                    )

        self.refresh()

    def _make_main_panel(self) -> Panel:
        """Create main panel with config, progress, metrics, and frontier."""
        s = self.state

        # Config line (model → reflection, train/val, budget)
        config_line = Text()
        config_line.append(self.model, style="white")
        config_line.append(" → ", style="dim")
        config_line.append(self.reflection_model, style="white")
        config_line.append(" (reflection)", style="dim")
        config_line.append("  |  ", style="dim")
        train_str = str(self.num_train) if self.num_train is not None else "..."
        val_str = str(self.num_val) if self.num_val is not None else "..."
        config_line.append(train_str, style="white")
        config_line.append(" train", style="dim")
        config_line.append(" / ", style="dim")
        config_line.append(val_str, style="white")
        config_line.append(" val", style="dim")
        config_line.append("  |  ", style="dim")
        config_line.append("budget ", style="dim")
        config_line.append(str(s.max_metric_calls), style="white")

        # Budget progress bar
        budget_used = s.metric_calls_used
        budget_total = s.max_metric_calls
        pct = (budget_used / budget_total * 100) if budget_total > 0 else 0

        progress = Progress(
            SpinnerColumn() if not s.completed else TextColumn(""),
            BarColumn(bar_width=None),
            TextColumn(f"[bold]{pct:.0f}%[/bold]"),
            TextColumn(f"({budget_used}/{budget_total} calls)"),
            TextColumn(f"| iter {s.iteration}"),
            console=self.console,
            expand=True,
        )
        task = progress.add_task("budget", total=budget_total, completed=budget_used)
        progress.update(task, completed=budget_used)

        # Metrics row
        metrics_line = Text()
        metrics_line.append("╰─ ", style="dim")

        # Phase with color
        phase_styles = {
            "initializing": "dim",
            "initial valset done": "green",
            "selecting": "blue",
            "reflecting": "magenta",
            "re-evaluating": "cyan",
            "accepted": "bold green",
            "rejected": "red",
        }
        phase_style = phase_styles.get(s.phase, "white")
        metrics_line.append("phase ", style="dim")
        metrics_line.append(s.phase, style=phase_style)

        # Average score
        if s.valset_rows:
            avg_score = sum(r.best_score for r in s.valset_rows.values()) / len(
                s.valset_rows
            )
            metrics_line.append("   ", style="dim")
            metrics_line.append("avg ", style="dim")
            metrics_line.append(f"{avg_score:.3f}", style="bold")

        # Minibatch info
        if s.minibatch_before is not None:
            metrics_line.append("   ", style="dim")
            metrics_line.append("minibatch ", style="dim")
            if s.minibatch_skipped:
                metrics_line.append(f"{s.minibatch_before:.2f} ", style="white")
                metrics_line.append("✓ perfect", style="bold green")
            elif s.minibatch_after is None:
                metrics_line.append(f"{s.minibatch_before:.2f} → ...", style="white")
            elif s.minibatch_accepted:
                metrics_line.append(
                    f"{s.minibatch_before:.2f} → {s.minibatch_after:.2f} ",
                    style="white",
                )
                metrics_line.append("✓", style="bold green")
            else:
                metrics_line.append(
                    f"{s.minibatch_before:.2f} → {s.minibatch_after:.2f} ",
                    style="white",
                )
                metrics_line.append("✗", style="red")

        # Valset evals count (right side)
        evals_text = Text()
        n = s.num_valset_evals
        evals_label = "eval" if n == 1 else "evals"
        evals_text.append(f"{n} valset {evals_label}", style="dim")

        # Left/right aligned row
        metrics_table = make_aligned_row(metrics_line, evals_text)

        # Frontier section (compact)
        frontier_content = self._make_frontier_content()

        # Combine all content
        content = Group(
            config_line, Text(""), progress, metrics_table, frontier_content
        )

        # Border style based on phase
        border_styles = {
            "initializing": "dim",
            "initial valset done": "yellow",
            "selecting": "yellow",
            "reflecting": "yellow",
            "re-evaluating": "yellow",
            "accepted": "green",
            "rejected": "yellow",
        }
        border_style = border_styles.get(s.phase, "yellow")
        if s.completed:
            border_style = "green"

        # Panel with env_id as title (left-aligned)
        title = f"[cyan bold]{self.env_id}[/cyan bold]"
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=border_style,
            expand=True,
        )

    def _make_frontier_content(self) -> Table:
        """Create compact frontier table for inside the main panel."""
        table = Table(
            show_header=True, header_style="bold dim", box=None, padding=(0, 1)
        )
        table.add_column("row", style="dim", width=5, justify="right")
        table.add_column("best", style="green", width=5, justify="right")
        table.add_column("prompt#", style="dim yellow", justify="left")

        rows = self.state.valset_rows
        if not rows:
            table.add_row("-", "-", "[dim]waiting for valset eval...[/]")
        else:
            for row_idx in sorted(rows.keys()):
                row = rows[row_idx]
                score_style = (
                    "green"
                    if row.best_score >= 1.0
                    else "yellow"
                    if row.best_score > 0
                    else "red"
                )
                prompts_str = ",".join(str(idx) for idx in row.best_candidate_idxs)
                table.add_row(
                    str(row_idx),
                    f"[{score_style}]{row.best_score:.2f}[/]",
                    prompts_str,
                )

        return table

    def _render(self) -> Group:
        """Render the full display."""
        items = [
            self._make_main_panel(),
            self._make_log_panel(),  # Always show, with placeholder if empty
        ]

        # Only show footer in TUI mode
        if self.screen:
            items.append(self._make_footer())

        return Group(*items)

    def _make_footer(self) -> Panel:
        """Create the footer panel with instructions."""
        if self.state.completed:
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
                footer_text.append("Optimization complete", style="dim")
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

    def set_result(
        self, best_prompt: str | None = None, save_path: str | None = None
    ) -> None:
        """Set final result info for summary display."""
        self._best_prompt = best_prompt
        self._save_path = save_path

    def _print_final_summary(self) -> None:
        """Print final summary in Rich styling."""
        s = self.state
        self.console.print()

        # Summary table (horizontal like eval)
        table = Table(title="Optimization Summary")
        table.add_column("env_id", style="cyan")
        table.add_column("status", justify="center")
        table.add_column("budget", justify="center")
        table.add_column("iterations", justify="center")
        table.add_column("avg_score", justify="center")
        table.add_column("perfect", justify="center")

        # Calculate stats
        avg_score = "—"
        perfect_str = "—"
        if s.valset_rows:
            avg_best = sum(r.best_score for r in s.valset_rows.values()) / len(
                s.valset_rows
            )
            perfect_count = sum(
                1 for r in s.valset_rows.values() if r.best_score >= 1.0
            )
            avg_score = f"{avg_best:.3f}"
            perfect_str = f"{perfect_count}/{len(s.valset_rows)}"

        status = "[green]done[/green]" if s.completed else "[yellow]running[/yellow]"
        budget_str = f"{s.metric_calls_used}/{s.max_metric_calls}"

        table.add_row(
            self.env_id,
            status,
            budget_str,
            str(s.iteration),
            avg_score,
            perfect_str,
        )

        self.console.print(table)

        # Best prompt
        best_prompt = getattr(self, "_best_prompt", None)
        if best_prompt:
            self.console.print()
            self.console.print("[bold]Best system prompt:[/bold]")
            self.console.print(Panel(best_prompt, border_style="dim"))

        # Save path
        save_path = getattr(self, "_save_path", None)
        if save_path:
            self.console.print()
            self.console.print("[bold]Results saved to:[/bold]")
            self.console.print(f"  [cyan]•[/cyan] {save_path}")

        self.console.print()
