"""
Textual-based TUI for viewing verifiers eval results.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markup import escape as safe_escape
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Footer, Label, OptionList, Static
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


class ViewRunScreen(Screen):
    """Screen for viewing run details and rollouts."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("left,h", "prev_record", "Previous"),
        Binding("right,l", "next_record", "Next"),
    ]

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = load_run_results(run)
        self.current_record_idx = 0

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
        prompt_widget.update(format_prompt_or_completion(prompt))

        # Update completion
        completion = record.get("completion", "")
        completion_widget = self.query_one("#completion-content", Static)
        completion_text = format_prompt_or_completion(completion)
        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")
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
    
    Footer {
        background: $panel;
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


def main() -> None:
    # Optional args via env vars
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    app = VerifiersTUI(env_dir, outputs_dir)
    app.run()


if __name__ == "__main__":
    main()
