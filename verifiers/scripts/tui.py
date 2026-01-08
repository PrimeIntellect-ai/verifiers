"""
Textual-based TUI for viewing verifiers eval results.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markup import escape as safe_escape
from rich.text import Text
from rich.console import Console
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import Footer, Input, Label, OptionList, Static
from textual.widgets._option_list import Option


# ----------------------------
# Discovery and data loading
# ----------------------------
@dataclass
class RunInfo:
    env_id: str
    model: str
    run_id: str
    path: Path
    metadata: Dict[str, Any]


def _iter_eval_roots(env_dir: Path, global_outputs_dir: Path) -> List[Path]:
    roots: List[Path] = []
    if env_dir.exists():
        for child in env_dir.iterdir():
            if child.is_dir():
                candidate = child / "outputs" / "evals"
                if candidate.exists():
                    roots.append(candidate)
    if (global_outputs_dir / "evals").exists():
        roots.append(global_outputs_dir / "evals")
    return roots


def _parse_env_and_model(dir_name: str) -> Optional[Tuple[str, str]]:
    if "--" not in dir_name:
        return None
    env, model_part = dir_name.split("--", 1)
    model = model_part.replace("--", "/")
    return env, model


def discover_results(
    env_dir_path: str = "./environments", outputs_dir_path: str = "./outputs"
) -> Dict[str, Dict[str, List[RunInfo]]]:
    """
    Returns mapping: env_id -> model -> list[RunInfo]
    """
    env_dir = Path(env_dir_path)
    global_outputs_dir = Path(outputs_dir_path)
    roots = _iter_eval_roots(env_dir, global_outputs_dir)

    discovered: Dict[str, Dict[str, List[RunInfo]]] = {}
    for root in roots:
        for env_model_dir in sorted(
            root.iterdir() if root.exists() else [], key=lambda p: p.name
        ):
            if not env_model_dir.is_dir():
                continue
            parsed = _parse_env_and_model(env_model_dir.name)
            if parsed is None:
                continue
            env_id, model = parsed
            for run_dir in sorted(env_model_dir.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir():
                    continue
                meta = run_dir / "metadata.json"
                results = run_dir / "results.jsonl"
                if meta.exists() and results.exists():
                    try:
                        metadata = json.loads(meta.read_text())
                    except Exception:
                        metadata = {}
                    run = RunInfo(
                        env_id=env_id,
                        model=model,
                        run_id=run_dir.name,
                        path=run_dir,
                        metadata=metadata,
                    )
                    discovered.setdefault(env_id, {}).setdefault(model, []).append(run)

    # Sort runs by time
    for env_id, models in discovered.items():
        for model, runs in models.items():
            runs.sort(
                key=lambda r: (
                    r.metadata.get("date", ""),
                    r.metadata.get("time", ""),
                    r.run_id,
                )
            )
    return discovered


def load_run_results(run: RunInfo) -> List[Dict[str, Any]]:
    """Load results.jsonl into memory."""
    data: List[Dict[str, Any]] = []
    results_path = run.path / "results.jsonl"
    with results_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


# ----------------------------
# Formatting helpers
# ----------------------------


def format_prompt_or_completion(prompt_or_completion) -> Text:
    """Format completion for display."""
    out = Text()
    if isinstance(prompt_or_completion, list):
        for msg in prompt_or_completion:
            if not isinstance(msg, dict):
                out.append(str(msg))
                out.append("\n\n")
                continue
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            # Style by role
            if role == "assistant":
                out.append("assistant: ", style="bold")
                out.append(content)
            elif role == "tool":
                out.append("tool result: ", style="bold dim")
                out.append(content)
            else:
                out.append(f"{role}: ", style="bold dim")
                out.append(content)
            out.append("\n")
            # Tool calls
            tool_calls_data = msg.get("tool_calls", [])
            if isinstance(tool_calls_data, list) and tool_calls_data:
                if isinstance(tool_calls_data[0], str):
                    parsed = []
                    for tc_str in tool_calls_data:
                        try:
                            parsed.append(json.loads(tc_str))
                        except Exception:
                            parsed.append(tc_str)
                    tool_calls_data = parsed

                for tc in tool_calls_data:
                    out.append("\ntool call: ", style="bold")
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        out.append(str(fn.get("name", "")))
                        out.append("\n")
                        out.append(str(fn.get("arguments", "")))
                    else:
                        out.append(str(tc))
                    out.append("\n")
            out.append("\n")
        return out
    out.append(str(prompt_or_completion))
    return out


# ----------------------------
# Custom Panel Widget
# ----------------------------
class Panel(Container):
    """A rounded panel container."""

    DEFAULT_CSS = """
    Panel {
        border: round white;
        padding: 1 2;
        margin: 1;
    }
    """


# ----------------------------
# Search helpers
# ----------------------------
@dataclass(frozen=True)
class SearchHit:
    column: str
    line_index: int
    line_text: str


@dataclass(frozen=True)
class SearchResult:
    column: str
    line_index: int
    pattern: str


def _stylize_matches(text: Text, pattern: re.Pattern, style: str) -> Text:
    plain = text.plain
    for match in pattern.finditer(plain):
        text.stylize(style, match.start(), match.end())
    return text


def _line_wrap_count(line: str, width: int, console: Console) -> int:
    if width <= 0:
        return 1
    if not line:
        return 1
    try:
        wrapped = Text(line).wrap(console, width)
        return max(1, len(wrapped))
    except Exception:
        return max(1, (len(line) - 1) // width + 1)


def _compute_line_offsets(lines: List[str], width: int, console: Console) -> List[int]:
    offsets: List[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += _line_wrap_count(line, width, console)
    return offsets


# ----------------------------
# Screens
# ----------------------------
class SelectEnvScreen(Screen):
    """Screen for selecting an environment."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index
        self.env_ids = sorted(index.keys())

    def compose(self) -> ComposeResult:
        with Container():
            yield Panel(
                Label(Text("Select Environment", style="bold"), classes="title"),
                OptionList(id="env-list"),
            )
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one("#env-list", OptionList)

        if not self.env_ids:
            option_list.add_option(
                Option("No completed evals found", id="__none__", disabled=True)
            )
            return

        for env_id in self.env_ids:
            models = self.index[env_id]
            total_runs = sum(len(runs) for runs in models.values())
            option_list.add_option(
                Option(
                    f"{safe_escape(env_id)} - Models: {len(models)}, Runs: {total_runs}",
                    id=env_id,
                )
            )

        option_list.focus()

    @on(OptionList.OptionSelected, "#env-list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option_id and event.option_id in self.env_ids:
            self.app.push_screen(SelectModelScreen(self.index, event.option_id))

    def action_select(self) -> None:
        """Handle Enter key."""
        option_list = self.query_one("#env-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option and option.id in self.env_ids:
                self.app.push_screen(SelectModelScreen(self.index, option.id))


class SelectModelScreen(Screen):
    """Screen for selecting a model."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]], env_id: str):
        super().__init__()
        self.index = index
        self.env_id = env_id
        self.models = sorted(index[env_id].keys())

    def compose(self) -> ComposeResult:
        with Container():
            yield Panel(
                Label(Text.assemble(("Environment: ", "bold"), str(self.env_id))),
                Label(Text("Select Model")),
                OptionList(id="model-list"),
            )
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one("#model-list", OptionList)

        for model in self.models:
            runs = self.index[self.env_id][model]
            option_list.add_option(
                Option(f"{safe_escape(model)} - Runs: {len(runs)}", id=model)
            )

        option_list.focus()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(OptionList.OptionSelected, "#model-list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option_id and event.option_id in self.models:
            self.app.push_screen(
                SelectRunScreen(self.index, self.env_id, event.option_id)
            )

    def action_select(self) -> None:
        """Handle Enter key."""
        option_list = self.query_one("#model-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option and option.id in self.models:
                self.app.push_screen(
                    SelectRunScreen(self.index, self.env_id, option.id)
                )


class SelectRunScreen(Screen):
    """Screen for selecting a run."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(
        self, index: Dict[str, Dict[str, List[RunInfo]]], env_id: str, model: str
    ):
        super().__init__()
        self.index = index
        self.env_id = env_id
        self.model = model
        self.runs = index[env_id][model]

    def compose(self) -> ComposeResult:
        with Container():
            yield Panel(
                Label(Text.assemble(("Environment: ", "bold"), str(self.env_id))),
                Label(Text.assemble(("Model: ", "bold"), str(self.model))),
                Label(Text("Select Run")),
                OptionList(id="run-list"),
                classes="run-list-panel",
            )
            yield Panel(
                VerticalScroll(
                    Static("", id="run-details", markup=False), id="run-details-scroll"
                ),
                classes="run-details-panel",
            )
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one("#run-list", OptionList)

        for i, run in enumerate(self.runs):
            meta = run.metadata
            datetime_str = f"{meta.get('date', '')} {meta.get('time', '')}".strip()
            reward = meta.get("avg_reward", "")
            if isinstance(reward, (int, float)):
                reward_str = f"Reward: {reward:.3f}"
            else:
                reward_str = f"Reward: {reward}"

            option_list.add_option(
                Option(
                    f"{safe_escape(run.run_id)} - {safe_escape(datetime_str)} | {safe_escape(reward_str)}",
                    id=str(i),
                )
            )

        option_list.focus()
        details_widget = self.query_one("#run-details", Static)
        details_widget.update(Text("Select a run to see details", style="dim"))

    @on(OptionList.OptionHighlighted, "#run-list")
    def on_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is not None:
            self._update_details_for_index(int(event.option_id))

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(OptionList.OptionSelected, "#run-list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option_id is not None:
            idx = int(event.option_id)
            if 0 <= idx < len(self.runs):
                self.app.push_screen(ViewRunScreen(self.runs[idx]))

    def action_select(self) -> None:
        """Handle Enter key."""
        option_list = self.query_one("#run-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option and option.id is not None:
                idx = int(option.id)
                if 0 <= idx < len(self.runs):
                    self.app.push_screen(ViewRunScreen(self.runs[idx]))

    def _update_details_for_index(self, idx: int) -> None:
        if not (0 <= idx < len(self.runs)):
            return
        run = self.runs[idx]
        details = self._build_run_details(run)
        details_widget = self.query_one("#run-details", Static)
        details_widget.update(details)

    def _build_run_details(self, run: RunInfo) -> Text:
        meta = run.metadata
        out = Text()
        out.append("Run ID: ", style="bold")
        out.append(str(run.run_id))
        out.append("\n")
        out.append("Environment: ", style="bold")
        out.append(str(run.env_id))
        out.append("\n")
        out.append("Model: ", style="bold")
        out.append(str(run.model))
        out.append("\n")
        base_url = meta.get("base_url", "")
        if base_url:
            out.append("Base URL: ", style="bold")
            out.append(str(base_url))
            out.append("\n")

        avg_reward = meta.get("avg_reward")
        if isinstance(avg_reward, (int, float)):
            out.append("Avg reward: ", style="bold")
            out.append(f"{avg_reward:.3f}")
            out.append("\n")

        avg_metrics = meta.get("avg_metrics", {})
        if isinstance(avg_metrics, dict) and avg_metrics:
            out.append("Avg metrics: ", style="bold")
            out.append("\n")
            for key in sorted(avg_metrics.keys()):
                value = avg_metrics.get(key)
                if isinstance(value, (int, float)):
                    out.append(f"  {key}: {value:.3f}\n")
                else:
                    out.append(f"  {key}: {value}\n")

        time_ms = meta.get("time_ms")
        if isinstance(time_ms, (int, float)):
            seconds = time_ms / 1000.0
            if seconds >= 60:
                minutes = int(seconds // 60)
                rem = seconds - minutes * 60
                runtime_str = f"{minutes}m {rem:.1f}s"
            else:
                runtime_str = f"{seconds:.1f}s"
            out.append("Runtime: ", style="bold")
            out.append(runtime_str)
            out.append("\n")

        env_args = meta.get("env_args", {})
        out.append("\nEnv args:\n", style="bold")
        try:
            out.append(json.dumps(env_args, ensure_ascii=False, indent=2))
        except Exception:
            out.append(str(env_args))

        sampling_args = meta.get("sampling_args", {})
        out.append("\n\nSampling args:\n", style="bold")
        try:
            out.append(json.dumps(sampling_args, ensure_ascii=False, indent=2))
        except Exception:
            out.append(str(sampling_args))

        return out


class ViewRunScreen(Screen):
    """Screen for viewing run details and rollouts."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("left,h", "prev_record", "Previous"),
        Binding("right,l", "next_record", "Next"),
        Binding("s", "search", "Search"),
    ]

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = load_run_results(run)
        self.current_record_idx = 0
        self._prompt_lines: List[str] = []
        self._completion_lines: List[str] = []
        self._prompt_offsets: List[int] = []
        self._completion_offsets: List[int] = []
        self._highlight_regex: Optional[re.Pattern] = None
        self._highlight_column: Optional[str] = None
        self._highlight_timer = None

    def compose(self) -> ComposeResult:
        with Container():
            # Metadata section
            yield Panel(
                Static(self._get_metadata_text(), id="metadata", markup=False),
                classes="metadata-panel",
            )

            # Rollout section with two columns
            with Horizontal(classes="rollout-container"):
                with Panel(classes="column-panel"):
                    yield Label(Text("Prompt", style="bold"), classes="column-header")
                    yield VerticalScroll(
                        Static("", id="prompt-content", markup=False),
                        id="prompt-scroll",
                    )

                with Panel(classes="column-panel"):
                    yield Label(
                        Text("Completion", style="bold"), classes="column-header"
                    )
                    yield VerticalScroll(
                        Static("", id="completion-content", markup=False),
                        id="completion-scroll",
                    )

            # Details section (horizontal scroll)
            yield Panel(Static("", id="details", markup=False), classes="details-panel")

        yield Footer()

    def _get_metadata_text(self) -> Text:
        meta = self.run.metadata
        sampling_args = meta.get("sampling_args", {})
        avg_reward = meta.get("avg_reward", "")
        if isinstance(avg_reward, (int, float)):
            avg_reward_str = f"{avg_reward:.3f}"
        else:
            avg_reward_str = str(avg_reward) if avg_reward else "N/A"

        def format_sampling_param(value: Any) -> str:
            return str(value) if value is not None else "N/A"

        temperature_str = format_sampling_param(sampling_args.get("temperature"))
        max_tokens_str = format_sampling_param(sampling_args.get("max_tokens"))

        # Create three columns of information without markup, with styled labels
        col1_items = [
            ("Environment: ", str(self.run.env_id)),
            ("Model: ", str(self.run.model)),
            ("Run ID: ", str(self.run.run_id)),
            (
                "Date: ",
                f"{str(meta.get('date', ''))} {str(meta.get('time', ''))}".strip(),
            ),
        ]

        col2_items = [
            ("Record: ", f"{self.current_record_idx + 1}/{len(self.records)}"),
            ("Examples: ", str(meta.get("num_examples", ""))),
            ("Rollouts/ex: ", str(meta.get("rollouts_per_example", ""))),
            ("", ""),
        ]

        col3_items = [
            ("Avg reward: ", avg_reward_str),
            ("Max tokens: ", max_tokens_str),
            ("Temperature: ", temperature_str),
            ("", ""),
        ]

        def build_padded(label: str, value: str, width: int) -> Text:
            combined = f"{label}{value}"
            pad_len = max(0, width - len(combined))
            t = Text()
            if label:
                t.append(label, style="bold")
            if value:
                t.append(value)
            if pad_len:
                t.append(" " * pad_len)
            return t

        lines: List[Text] = []
        num_rows = max(len(col1_items), len(col2_items), len(col3_items))
        for i in range(num_rows):
            left_label, left_value = col1_items[i] if i < len(col1_items) else ("", "")
            mid_label, mid_value = col2_items[i] if i < len(col2_items) else ("", "")
            right_label, right_value = (
                col3_items[i] if i < len(col3_items) else ("", "")
            )

            row = Text()
            row += build_padded(left_label, left_value, 45)
            row += build_padded(mid_label, mid_value, 35)
            if right_label or right_value:
                row.append(right_label, style="bold")
                row.append(right_value)
            lines.append(row)

        return Text("\n").join(lines)

    def on_mount(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        """Update the display with current record."""
        if not self.records:
            return

        record = self.records[self.current_record_idx]

        # Update prompt
        prompt = record.get("prompt", "")
        prompt_widget = self.query_one("#prompt-content", Static)
        prompt_text = format_prompt_or_completion(prompt)
        if self._highlight_regex and self._highlight_column == "prompt":
            _stylize_matches(prompt_text, self._highlight_regex, "reverse")
        prompt_widget.update(prompt_text)

        # Update completion
        completion = record.get("completion", "")
        completion_widget = self.query_one("#completion-content", Static)
        completion_text = format_prompt_or_completion(completion)
        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")
        if self._highlight_regex and self._highlight_column == "completion":
            _stylize_matches(completion_text, self._highlight_regex, "reverse")
        completion_widget.update(completion_text)

        # Update details
        details_lines = Text()
        reward = record.get("reward", None)
        if reward is not None:
            reward_str = (
                f"{reward:.3f}" if isinstance(reward, (int, float)) else str(reward)
            )
            details_lines.append("Reward: ", style="bold")
            details_lines.append(f"{reward_str}\n")

        answer = record.get("answer", None)
        if answer not in (None, ""):
            details_lines.append("Answer: ", style="bold")
            details_lines.append(str(answer))
            details_lines.append("\n")

        info = record.get("info", None)
        if info not in (None, {}):
            details_lines.append("Info: ", style="bold")
            try:
                details_lines.append(json.dumps(info, ensure_ascii=False, indent=2))
            except Exception:
                details_lines.append(str(info))

        task = record.get("task", None)
        if task not in (None, ""):
            details_lines.append("Task: ", style="bold")
            details_lines.append(str(task))

        details_widget = self.query_one("#details", Static)
        details_widget.update(
            details_lines
            if details_lines.plain.strip()
            else Text("No additional details", style="dim")
        )
        # Update metadata with current record index
        metadata_widget = self.query_one("#metadata", Static)
        metadata_widget.update(self._get_metadata_text())
        self._prompt_lines = prompt_text.plain.split("\n")
        self._completion_lines = completion_text.plain.split("\n")

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_prev_record(self) -> None:
        if self.records:
            self.current_record_idx = (self.current_record_idx - 1) % len(self.records)
            self.update_display()
            # Reset scroll positions
            self.query_one("#prompt-scroll").scroll_y = 0
            self.query_one("#completion-scroll").scroll_y = 0

    def action_next_record(self) -> None:
        if self.records:
            self.current_record_idx = (self.current_record_idx + 1) % len(self.records)
            self.update_display()
            # Reset scroll positions
            self.query_one("#prompt-scroll").scroll_y = 0
            self.query_one("#completion-scroll").scroll_y = 0

    def action_search(self) -> None:
        if not self.records:
            return
        self.app.push_screen(
            SearchScreen(self._prompt_lines, self._completion_lines),
            self._handle_search_result,
        )

    def _handle_search_result(self, result: Optional[SearchResult]) -> None:
        if result is None:
            return
        try:
            compiled = re.compile(result.pattern, re.IGNORECASE)
        except re.error:
            return
        self._highlight_regex = compiled
        self._highlight_column = result.column
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
        self.update_display()
        self._recompute_line_offsets()
        self._scroll_to_line(result.column, result.line_index)
        self._highlight_timer = self.set_timer(3.0, self._clear_highlight)

    def _clear_highlight(self) -> None:
        if not self.is_mounted:
            return
        self._highlight_regex = None
        self._highlight_column = None
        self.update_display()

    def _get_scroll_width(self, scroll: VerticalScroll) -> int:
        width = scroll.size.width
        padding = getattr(scroll.styles, "padding", None)
        if padding is not None:
            width -= padding.left + padding.right
        if not width:
            content_size = getattr(scroll, "content_size", None)
            if content_size is not None:
                width = scroll.content_size.width
        return max(1, width)

    def _recompute_line_offsets(self) -> None:
        if not self._prompt_lines and not self._completion_lines:
            return
        console = getattr(self.app, "console", None) or Console()
        prompt_scroll = self.query_one("#prompt-scroll", VerticalScroll)
        completion_scroll = self.query_one("#completion-scroll", VerticalScroll)
        prompt_width = self._get_scroll_width(prompt_scroll)
        completion_width = self._get_scroll_width(completion_scroll)
        self._prompt_offsets = _compute_line_offsets(
            self._prompt_lines, prompt_width, console
        )
        self._completion_offsets = _compute_line_offsets(
            self._completion_lines, completion_width, console
        )

    def _scroll_to_line(self, column: str, line_index: int) -> None:
        if column == "prompt":
            offsets = self._prompt_offsets
            scroll = self.query_one("#prompt-scroll", VerticalScroll)
        else:
            offsets = self._completion_offsets
            scroll = self.query_one("#completion-scroll", VerticalScroll)

        if not offsets or line_index < 0 or line_index >= len(offsets):
            return
        target = offsets[line_index]
        max_scroll = getattr(scroll, "max_scroll_y", None)
        if callable(max_scroll):
            max_scroll = max_scroll()
        if max_scroll is None:
            max_scroll = getattr(scroll, "scroll_y_max", None)
        if max_scroll is None:
            target = max(0, target)
        else:
            target = max(0, min(target, max_scroll))

        if hasattr(scroll, "scroll_to"):
            try:
                scroll.scroll_to(y=target, animate=False)
            except TypeError:
                scroll.scroll_to(y=target)
        else:
            scroll.scroll_y = target


# ----------------------------
# Main App
# ----------------------------
class VerifiersTUI(App):
    """Textual-based TUI for viewing verifiers eval results."""

    # Custom dark theme with a modern color palette
    ENABLE_COMMAND_PALETTE = False  # Disable command palette for cleaner UI

    # Define custom dark theme
    BLACK_WARM_THEME = Theme(
        name="black-warm",
        primary="#d4a373",  # Warm tan/beige
        secondary="#808080",  # Gray
        accent="#c9ada7",  # Muted rose
        warning="#ffa500",  # Orange
        error="#ff6b6b",  # Soft red
        success="#98c379",  # Soft green
        background="#141414",
        surface="#141414",
        panel="#141414",
        foreground="#ffffff",
        dark=True,
    )

    # Define custom light theme with matching warm tones
    WHITE_WARM_THEME = Theme(
        name="white-warm",
        primary="#8b6f47",  # Darker warm brown (darker than dark theme for contrast)
        secondary="#606060",  # Medium gray
        accent="#a08b87",  # Muted warm brown-rose
        warning="#ff8c00",  # Dark orange
        error="#dc143c",  # Crimson
        success="#6b8e23",  # Olive green
        background="#f5f5f5",  # Light warm grey
        surface="#f5f5f5",  # Light warm grey
        panel="#f5f5f5",  # Light warm grey
        foreground="#1a1a1a",  # Near black
        dark=False,
    )

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    CSS = """
    /* Clean black theme */
    Screen {
        layout: vertical;
        background: $background;
    }
    
    Panel {
        border: round $primary;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $panel;
    }
    
    Label {
        color: $text;
    }
    
    Static {
        color: $text;
    }
    
    .title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    OptionList {
        height: auto;
        max-height: 20;
        background: $surface;
        color: $text;
    }
    
    OptionList > .option-list--option-highlighted {
        background: $primary 20%;
    }
    
    #view-container {
        layout: vertical;
        height: 100%;
    }
    
    .metadata-panel {
        height: auto;
        min-height: 6;
        max-height: 8;
    }
    
    .rollout-container {
        height: 1fr;
        layout: horizontal;
    }
    
    .column-panel {
        width: 50%;
        height: 100%;
        layout: vertical;
    }
    
    .column-header {
        height: auto;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    #prompt-scroll, #completion-scroll {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-color: $secondary;
        scrollbar-background: $panel;
        scrollbar-corner-color: $panel;
    }
    
    .details-panel {
        height: auto;
        min-height: 3;
        max-height: 6;
    }
    
    .run-list-panel {
        height: 1fr;
    }
    
    #run-list {
        height: 1fr;
        max-height: 100%;
    }
    
    .run-details-panel {
        height: 1fr;
    }
    
    #run-details-scroll {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-color: $secondary;
        scrollbar-background: $panel;
        scrollbar-corner-color: $panel;
    }
    
    Footer {
        background: $panel;
    }
    
    .search-header {
        height: auto;
    }
    
    .search-columns {
        height: 1fr;
        layout: horizontal;
    }
    
    .search-panel {
        width: 50%;
        height: 100%;
        layout: vertical;
    }
    
    .search-input {
        background: $surface;
        color: $text;
    }
    """

    def __init__(
        self, env_dir_path: str = "./environments", outputs_dir_path: str = "./outputs"
    ):
        super().__init__()
        self.env_dir_path = env_dir_path
        self.outputs_dir_path = outputs_dir_path
        self.index = discover_results(env_dir_path, outputs_dir_path)

    def on_mount(self) -> None:
        # Register both custom themes
        self.register_theme(self.BLACK_WARM_THEME)
        self.register_theme(self.WHITE_WARM_THEME)
        # Start with dark theme
        self.theme = "black-warm"
        self.push_screen(SelectEnvScreen(self.index))

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_toggle_dark(self) -> None:
        """Toggle between dark and light themes."""
        # Toggle between our custom dark and light themes
        if self.theme == "black-warm":
            self.theme = "white-warm"
        else:
            self.theme = "black-warm"


class SearchScreen(ModalScreen[Optional[SearchResult]]):
    """Modal screen for searching prompt/completion text."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, prompt_lines: List[str], completion_lines: List[str]):
        super().__init__()
        self._prompt_lines = prompt_lines
        self._completion_lines = completion_lines
        self._prompt_hits: List[SearchHit] = []
        self._completion_hits: List[SearchHit] = []
        self._active_column: Optional[str] = None
        self._prompt_cursor: Optional[int] = None
        self._completion_cursor: Optional[int] = None

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="search-header"):
                yield Label(Text("Search (regex, case-insensitive)", style="bold"))
                yield Input(
                    placeholder="regex...", id="search-input", classes="search-input"
                )
                yield Label("", id="search-error", classes="subtitle")

            with Horizontal(classes="search-columns"):
                with Panel(classes="search-panel"):
                    yield Label(Text("Prompt results", style="bold"), id="prompt-count")
                    yield OptionList(id="prompt-results")
                with Panel(classes="search-panel"):
                    yield Label(
                        Text("Completion results", style="bold"),
                        id="completion-count",
                    )
                    yield OptionList(id="completion-results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()
        self._update_results("")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_results(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_select()

    @on(OptionList.OptionHighlighted, "#prompt-results")
    def on_prompt_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        self._active_column = "prompt"
        self._prompt_cursor = int(event.option_id)
        self._sync_highlights()

    @on(OptionList.OptionHighlighted, "#completion-results")
    def on_completion_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        self._active_column = "completion"
        self._completion_cursor = int(event.option_id)
        self._sync_highlights()

    @on(OptionList.OptionSelected, "#prompt-results")
    def on_prompt_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        self._active_column = "prompt"
        self._prompt_cursor = int(event.option_id)
        self.action_select()

    @on(OptionList.OptionSelected, "#completion-results")
    def on_completion_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        self._active_column = "completion"
        self._completion_cursor = int(event.option_id)
        self.action_select()

    def on_key(self, event) -> None:
        if event.key in ("left", "right", "up", "down"):
            if event.key == "left":
                self._switch_column("prompt")
            elif event.key == "right":
                self._switch_column("completion")
            elif event.key == "up":
                self._move_selection(-1)
            elif event.key == "down":
                self._move_selection(1)
            event.prevent_default()
            event.stop()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        selection = self._current_selection()
        if selection is None:
            return
        pattern = self.query_one("#search-input", Input).value
        self.dismiss(
            SearchResult(
                column=selection.column,
                line_index=selection.line_index,
                pattern=pattern,
            )
        )

    def _update_results(self, pattern: str) -> None:
        prompt_list = self.query_one("#prompt-results", OptionList)
        completion_list = self.query_one("#completion-results", OptionList)
        error_label = self.query_one("#search-error", Label)
        prompt_label = self.query_one("#prompt-count", Label)
        completion_label = self.query_one("#completion-count", Label)

        prompt_list.clear_options()
        completion_list.clear_options()
        self._prompt_hits = []
        self._completion_hits = []
        self._prompt_cursor = None
        self._completion_cursor = None

        if not pattern:
            error_label.update("")
            prompt_label.update(Text("Prompt results", style="bold"))
            completion_label.update(Text("Completion results", style="bold"))
            self._active_column = None
            return

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            error_label.update(Text(f"Invalid regex: {exc}", style="red"))
            prompt_label.update(Text("Prompt results", style="bold"))
            completion_label.update(Text("Completion results", style="bold"))
            self._active_column = None
            return

        error_label.update("")
        self._prompt_hits = self._find_hits("prompt", self._prompt_lines, compiled)
        self._completion_hits = self._find_hits(
            "completion", self._completion_lines, compiled
        )

        for idx, hit in enumerate(self._prompt_hits):
            prompt_list.add_option(self._build_option(hit, compiled, idx))
        for idx, hit in enumerate(self._completion_hits):
            completion_list.add_option(self._build_option(hit, compiled, idx))

        prompt_label.update(
            Text(f"Prompt results ({len(self._prompt_hits)})", style="bold")
        )
        completion_label.update(
            Text(f"Completion results ({len(self._completion_hits)})", style="bold")
        )

        if self._completion_hits:
            self._active_column = "completion"
            self._completion_cursor = 0
        elif self._prompt_hits:
            self._active_column = "prompt"
            self._prompt_cursor = 0
        else:
            self._active_column = None

        self._sync_highlights()

    def _find_hits(
        self, column: str, lines: List[str], pattern: re.Pattern
    ) -> List[SearchHit]:
        hits: List[SearchHit] = []
        for idx, line in enumerate(lines):
            if pattern.search(line):
                hits.append(SearchHit(column=column, line_index=idx, line_text=line))
        return hits

    def _build_option(
        self, hit: SearchHit, pattern: re.Pattern, option_index: int
    ) -> Option:
        prefix = Text(f"{hit.line_index + 1:>5} | ", style="dim")
        content = Text(hit.line_text)
        _stylize_matches(content, pattern, "reverse")
        return Option(prefix + content, id=str(option_index))

    def _sync_highlights(self) -> None:
        prompt_list = self.query_one("#prompt-results", OptionList)
        completion_list = self.query_one("#completion-results", OptionList)

        if self._active_column == "prompt" and self._prompt_cursor is not None:
            prompt_list.highlighted = self._prompt_cursor
            completion_list.highlighted = None
            if hasattr(prompt_list, "scroll_to_highlight"):
                prompt_list.scroll_to_highlight()
        elif (
            self._active_column == "completion" and self._completion_cursor is not None
        ):
            completion_list.highlighted = self._completion_cursor
            prompt_list.highlighted = None
            if hasattr(completion_list, "scroll_to_highlight"):
                completion_list.scroll_to_highlight()
        else:
            prompt_list.highlighted = None
            completion_list.highlighted = None

    def _switch_column(self, target: str) -> None:
        if target == "prompt" and self._prompt_hits:
            self._active_column = "prompt"
            if self._prompt_cursor is None:
                self._prompt_cursor = 0
        elif target == "completion" and self._completion_hits:
            self._active_column = "completion"
            if self._completion_cursor is None:
                self._completion_cursor = 0
        self._sync_highlights()

    def _move_selection(self, delta: int) -> None:
        if self._active_column == "prompt" and self._prompt_hits:
            if self._prompt_cursor is None:
                self._prompt_cursor = 0
            else:
                self._prompt_cursor = max(
                    0, min(len(self._prompt_hits) - 1, self._prompt_cursor + delta)
                )
        elif self._active_column == "completion" and self._completion_hits:
            if self._completion_cursor is None:
                self._completion_cursor = 0
            else:
                self._completion_cursor = max(
                    0,
                    min(
                        len(self._completion_hits) - 1, self._completion_cursor + delta
                    ),
                )
        self._sync_highlights()

    def _current_selection(self) -> Optional[SearchHit]:
        if self._active_column == "prompt" and self._prompt_hits:
            if self._prompt_cursor is None:
                return None
            return self._prompt_hits[self._prompt_cursor]
        if self._active_column == "completion" and self._completion_hits:
            if self._completion_cursor is None:
                return None
            return self._completion_hits[self._completion_cursor]
        return None


def main() -> None:
    # Optional args via env vars
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    app = VerifiersTUI(env_dir, outputs_dir)
    app.run()


if __name__ == "__main__":
    main()
