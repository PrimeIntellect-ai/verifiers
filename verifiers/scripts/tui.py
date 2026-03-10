"""
Textual-based TUI for viewing verifiers eval results.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import (
    Collapsible,
    Footer,
    Input,
    Label,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual.widgets._option_list import Option

from verifiers.utils.display_utils import format_numeric

AnimationLevel = Literal["none", "basic", "full"]
TreeBinding = Binding | tuple[str, str] | tuple[str, str, str]


def _binding_key(binding: TreeBinding) -> str:
    if isinstance(binding, Binding):
        return binding.key
    return binding[0]


# ----------------------------
# Discovery and data loading
# ----------------------------
@dataclass
class RunInfo:
    env_id: str
    model: str
    run_id: str
    path: Path
    metadata: Optional[Dict[str, Any]] = None

    def load_metadata(self) -> Dict[str, Any]:
        if self.metadata is not None:
            return self.metadata
        meta_path = self.path / "metadata.json"
        try:
            self.metadata = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.metadata = {}
        return self.metadata


@dataclass(frozen=True)
class BrowserNodeData:
    kind: str
    env_id: str = ""
    model: str = ""
    run: Optional[RunInfo] = None


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    avg: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class RunOverviewStats:
    rewards: List[float]
    metric_summaries: List[MetricSummary]


class RunBrowserTree(Tree[BrowserNodeData]):
    """Tree with footer-visible shortcuts for the eval browser."""

    BINDINGS = [
        *(
            binding
            for binding in Tree.BINDINGS
            if _binding_key(binding) not in {"enter", "space"}
        ),
        Binding("enter", "select_cursor", "Open/toggle", show=True),
        Binding("space", "toggle_node", "Toggle folder", show=True),
    ]


def discover_results(
    env_dir_path: str = "./environments", outputs_dir_path: str = "./outputs"
) -> Dict[str, Dict[str, List[RunInfo]]]:
    """
    Returns mapping: env_id -> model -> list[RunInfo]
    """
    roots: List[Path] = []
    env_dir = Path(env_dir_path)
    if env_dir.is_dir():
        for env_path in sorted(env_dir.iterdir(), key=lambda path: path.name):
            candidate = env_path / "outputs" / "evals"
            if candidate.is_dir():
                roots.append(candidate)

    global_root = Path(outputs_dir_path) / "evals"
    if global_root.is_dir():
        roots.append(global_root)

    discovered: Dict[str, Dict[str, List[RunInfo]]] = {}
    for root in roots:
        for env_model_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not env_model_dir.is_dir() or "--" not in env_model_dir.name:
                continue
            env_id, model_part = env_model_dir.name.split("--", 1)
            model = model_part.replace("--", "/")
            for run_dir in sorted(env_model_dir.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "metadata.json").is_file() and (
                    run_dir / "results.jsonl"
                ).is_file():
                    run = RunInfo(
                        env_id=env_id,
                        model=model,
                        run_id=run_dir.name,
                        path=run_dir,
                    )
                    discovered.setdefault(env_id, {}).setdefault(model, []).append(run)

    return discovered


class LazyRunResults:
    """Lazy loader for results.jsonl with optional metadata count."""

    def __init__(self, run: RunInfo):
        self._path = run.path / "results.jsonl"
        self._fh = self._path.open("r", encoding="utf-8")
        self._offsets: List[int] = []
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._eof = False
        self._count_hint: Optional[int] = None
        self._count: Optional[int] = None

        meta = run.load_metadata()
        num_examples = meta.get("num_examples")
        rollouts_per_example = meta.get("rollouts_per_example")
        if isinstance(num_examples, int) and num_examples >= 0:
            if isinstance(rollouts_per_example, int) and rollouts_per_example >= 0:
                self._count_hint = num_examples * rollouts_per_example
            else:
                self._count_hint = num_examples

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _read_next_line(self) -> Optional[str]:
        if self._eof:
            return None
        pos = self._fh.tell()
        line = self._fh.readline()
        if not line:
            self._eof = True
            self._count = len(self._offsets)
            return None
        self._offsets.append(pos)
        return line

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        while len(self._offsets) <= index and not self._eof:
            line = self._read_next_line()
            if line is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        if self._count is not None:
            return self._count
        while not self._eof:
            line = self._read_next_line()
            if line is None:
                break
        self._count = len(self._offsets)
        return self._count

    def get(self, index: int) -> Dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        if not self._ensure_index(index):
            return {}
        pos = self._fh.tell()
        try:
            self._fh.seek(self._offsets[index])
            line = self._fh.readline()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {}
        finally:
            self._fh.seek(pos)
        self._cache[index] = data
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index)

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        line = self._read_next_line()
        return line is not None

    def count_hint(self) -> Optional[int]:
        if self._count is not None:
            return self._count
        return self._count_hint


# ----------------------------
# Formatting helpers
# ----------------------------


def _stringify_message_content(content: Any) -> str:
    """Render message content into readable plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(_pretty_json_or_str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        return _pretty_json_or_str(content)
    return str(content)


def _parse_tool_calls(tool_calls: Any) -> List[Any]:
    if not isinstance(tool_calls, list):
        return []
    return [_parse_jsonish_string(tool_call) for tool_call in tool_calls]


def _truncate_preview(text: str, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _count_result_records(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _compute_run_overview_stats(run: RunInfo) -> RunOverviewStats:
    rewards: List[float] = []
    metric_values: Dict[str, List[float]] = defaultdict(list)
    try:
        with (run.path / "results.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                reward = _numeric_reward(record.get("reward"))
                if reward is not None:
                    rewards.append(reward)
                for name, value in _extract_numeric_metric_values(record).items():
                    metric_values[name].append(value)
    except OSError:
        pass

    return RunOverviewStats(
        rewards=rewards,
        metric_summaries=[
            MetricSummary(
                name=name,
                count=len(values),
                avg=sum(values) / len(values),
                min_value=min(values),
                max_value=max(values),
            )
            for name, values in sorted(metric_values.items())
            if values
        ],
    )


def _format_message_preview(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = _stringify_message_content(message.get("content", ""))
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if content:
        return _truncate_preview(content, 56)
    if tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            function = first.get("function", {})
            name = function.get("name") or first.get("name") or ""
            return f"calls {name}" if name else ""
        return f"calls {_truncate_preview(str(first), 48)}"
    return ""


def _reward_style(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value >= 0.9:
            return "bold green"
        if value >= 0.5:
            return "bold yellow"
        return "bold red"
    return "bold"


def _format_reward_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _format_compact_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _numeric_reward(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _pretty_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _tool_call_parts(tool_call: Any) -> Tuple[str, str, Optional[str]]:
    if not isinstance(tool_call, dict):
        return str(tool_call), "", None

    function = tool_call.get("function")
    payload = function if isinstance(function, dict) else tool_call
    name = str(payload.get("name") or tool_call.get("name") or "")
    raw_arguments = payload.get("arguments", tool_call.get("arguments", ""))
    parsed_arguments = _parse_jsonish_string(raw_arguments)
    if isinstance(parsed_arguments, dict):
        arguments = (
            str(parsed_arguments["code"])
            if set(parsed_arguments.keys()) == {"code"}
            else _pretty_json_or_str(parsed_arguments)
        )
    elif isinstance(parsed_arguments, list):
        arguments = _pretty_json_or_str(parsed_arguments)
    else:
        arguments = str(raw_arguments) if raw_arguments not in (None, "") else ""
    call_id = tool_call.get("id")
    return name, arguments, str(call_id) if call_id not in (None, "") else None


def _tool_output_preview(message: Any) -> str:
    if not isinstance(message, dict):
        return _truncate_preview(str(message), 44)
    content = _stringify_message_content(message.get("content", ""))
    for line in content.splitlines():
        if line.strip():
            return _truncate_preview(line.strip(), 44)
    return _truncate_preview(content, 44)


def _tool_group_preview(message: Any, tool_outputs: List[Any]) -> str:
    base = _format_message_preview(message)
    if not tool_outputs:
        return base
    output_preview = _tool_output_preview(tool_outputs[0])
    if not base:
        return output_preview
    return _truncate_preview(f"{base} ... {output_preview}", 68)


def _raw_preview(value: Any, *, limit: int = 56) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return _truncate_preview(value, limit)
    if isinstance(value, list):
        for item in value:
            preview = _raw_preview(item, limit=limit)
            if preview:
                return preview
        return ""
    if isinstance(value, dict):
        content = _stringify_message_content(value.get("content", ""))
        if content:
            return _truncate_preview(content, limit)
        for key in ("text", "message", "error", "detail", "details", "type", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _truncate_preview(candidate, limit)
        return ""
    return _truncate_preview(str(value), limit)


def _error_preview(error: Any) -> str:
    parsed = _parse_jsonish_string(error)
    if isinstance(parsed, dict):
        chain = parsed.get("error_chain_str")
        if isinstance(chain, str) and chain.strip():
            return _truncate_preview(chain, 56)

        name = parsed.get("error")
        if isinstance(name, str) and name.strip():
            return _truncate_preview(name, 56)

        chain_repr = parsed.get("error_chain_repr")
        if isinstance(chain_repr, str) and chain_repr.strip():
            return _truncate_preview(chain_repr, 56)

    return _raw_preview(parsed, limit=56)


def _parse_jsonish_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def format_info_for_details(info: Any) -> str:
    """Format record info for the details panel in rollout view."""
    info_value = _parse_jsonish_string(info)
    if isinstance(info_value, (dict, list)):
        return _pretty_json_or_str(info_value)
    return str(info_value)


_STANDARD_NUMERIC_FIELDS = {
    "example_id",
    "prompt",
    "completion",
    "answer",
    "task",
    "info",
    "reward",
    "error",
    "timing",
    "is_completed",
    "is_truncated",
    "stop_condition",
    "metrics",
    "tool_defs",
    "token_usage",
    "error_chain",
    "long_error_chain",
}


def _extract_numeric_metric_values(record: Dict[str, Any]) -> Dict[str, float]:
    metric_values: Dict[str, float] = {}

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values[key] = float(value)

    info = _parse_jsonish_string(record.get("info"))
    if isinstance(info, dict):
        reward_signals = info.get("reward_signals")
        if isinstance(reward_signals, dict):
            for key, value in reward_signals.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values.setdefault(key, float(value))

    for key, value in record.items():
        if key in _STANDARD_NUMERIC_FIELDS:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metric_values.setdefault(key, float(value))

    return metric_values


def _build_reward_distribution_table(values: List[float], heading: str) -> Group | Text:
    if not values:
        return Text()

    avg_reward = sum(values) / len(values)
    summary = Text()
    summary.append(heading, style="bold dim")
    summary.append("\n")
    summary.append("count ", style="bold")
    summary.append(f"{len(values):,}")
    summary.append("   avg ", style="bold")
    summary.append(f"{avg_reward:.3f}", style=_reward_style(avg_reward))
    summary.append("   min ", style="bold")
    summary.append(f"{min(values):.3f}", style=_reward_style(min(values)))
    summary.append("   max ", style="bold")
    summary.append(f"{max(values):.3f}", style=_reward_style(max(values)))

    buckets = [
        ("<0", lambda reward: reward < 0, "bold red"),
        ("0-0.25", lambda reward: 0 <= reward < 0.25, "red"),
        ("0.25-0.5", lambda reward: 0.25 <= reward < 0.5, "yellow"),
        ("0.5-0.75", lambda reward: 0.5 <= reward < 0.75, "yellow"),
        ("0.75-1.0", lambda reward: 0.75 <= reward < 1.0, "green"),
        (">=1.0", lambda reward: reward >= 1.0, "bold green"),
    ]
    bucket_counts = [
        (label, sum(1 for reward in values if predicate(reward)), style)
        for label, predicate, style in buckets
    ]
    peak_count = max(count for _, count, _ in bucket_counts) or 1

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column("Range", style="dim", width=10, no_wrap=True)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Share", justify="right", width=8)
    table.add_column("Distribution", ratio=1, min_width=24)

    for label, count, style in bucket_counts:
        share = (count / len(values)) if values else 0.0
        fraction = count / peak_count if peak_count else 0.0
        filled_cells = round(max(0.0, min(1.0, fraction)) * 24)
        bar = Text()
        if filled_cells:
            bar.append("█" * filled_cells, style=style)
        if filled_cells < 24:
            bar.append("░" * (24 - filled_cells), style="dim")
        table.add_row(
            label,
            f"{count:,}",
            f"{share:.1%}",
            bar,
        )

    return Group(summary, table)


def _format_metric_stat_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"

    magnitude = abs(value)
    if magnitude >= 1000:
        precision = 1
    elif magnitude >= 100:
        precision = 2
    elif magnitude >= 1:
        precision = 3
    elif magnitude >= 0.01:
        precision = 3
    else:
        precision = 4
    return f"{value:,.{precision}f}".rstrip("0").rstrip(".")


def _build_metric_summary_table(metric_summaries: List[MetricSummary]) -> Table | Text:
    if not metric_summaries:
        return Text()

    counts = {summary.count for summary in metric_summaries}
    show_count_column = len(counts) > 1
    title_suffix = f" (n={next(iter(counts)):,})" if len(counts) == 1 else ""

    category_order = {
        "Tokens": 0,
        "Calls": 1,
        "Flow": 2,
        "Errors": 3,
        "Timing": 4,
        "Scores": 5,
        "Other": 6,
    }
    replacements = (
        ("sub llm", "sub-LLM"),
        ("main rlm", "main RLM"),
        ("root rlm", "root RLM"),
        ("llm", "LLM"),
        ("repl", "REPL"),
    )
    prepared: List[Tuple[int, int, str, str, MetricSummary]] = []
    for summary in metric_summaries:
        lowered = summary.name.lower()
        if "token" in lowered:
            category = "Tokens"
        elif "call" in lowered:
            category = "Calls"
        elif "turn" in lowered or "step" in lowered or "batch" in lowered:
            category = "Flow"
        elif "error" in lowered:
            category = "Errors"
        elif "time" in lowered or lowered.endswith("_ms"):
            category = "Timing"
        elif "reward" in lowered or "score" in lowered or "task" in lowered:
            category = "Scores"
        else:
            category = "Other"

        prefix_rank = 4
        if lowered.startswith("sub_llm_"):
            prefix_rank = 0
        elif lowered.startswith("main_rlm_"):
            prefix_rank = 1
        elif lowered.startswith("root_"):
            prefix_rank = 2
        elif lowered.startswith("repl_"):
            prefix_rank = 3

        display_name = summary.name.replace("_", " ")
        for source, target in replacements:
            display_name = display_name.replace(source, target)
        prepared.append(
            (
                category_order.get(category, 99),
                prefix_rank,
                display_name,
                category,
                summary,
            )
        )

    rows: List[Tuple[str, str, str, str, Optional[str], bool]] = []
    metric_width = len("Metric")
    average_width = len("Average")
    min_width = len("Min")
    max_width = len("Max")
    count_width = len("N")

    previous_category: str | None = None
    for _, _, display_name, category, summary in sorted(prepared):
        avg_text = _format_metric_stat_value(summary.avg)
        min_text = _format_metric_stat_value(summary.min_value)
        max_text = _format_metric_stat_value(summary.max_value)
        count_text = f"{summary.count:,}" if show_count_column else None
        rows.append(
            (
                display_name,
                avg_text,
                min_text,
                max_text,
                count_text,
                previous_category is not None and category != previous_category,
            )
        )
        metric_width = max(metric_width, len(display_name))
        average_width = max(average_width, len(avg_text))
        min_width = max(min_width, len(min_text))
        max_width = max(max_width, len(max_text))
        if count_text is not None:
            count_width = max(count_width, len(count_text))
        previous_category = category

    table = Table(
        title=f"Rollout metrics{title_suffix}",
        title_style="bold dim",
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column(
        "Metric",
        style="bold cyan",
        ratio=1,
        min_width=min(max(metric_width, 28), 38),
        no_wrap=True,
        overflow="ellipsis",
    )
    table.add_column("Average", justify="right", width=average_width, no_wrap=True)
    table.add_column("Min", justify="right", width=min_width, no_wrap=True)
    table.add_column("Max", justify="right", width=max_width, no_wrap=True)
    if show_count_column:
        table.add_column(
            "N", justify="right", style="dim", width=count_width, no_wrap=True
        )

    for display_name, avg_text, min_text, max_text, count_text, add_section in rows:
        if add_section:
            table.add_section()
        row = [
            display_name,
            avg_text,
            min_text,
            max_text,
        ]
        if count_text is not None:
            row.append(count_text)
        table.add_row(*row)

    return table


# ----------------------------
# Custom Panel Widget
# ----------------------------
class Panel(Container):
    """A rounded panel container."""

    pass


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
    pattern: str


@dataclass(frozen=True)
class HistorySectionData:
    title: str
    body: str
    column: str
    collapsed: bool
    classes: str
    nested_sections: Tuple["HistorySectionData", ...] = ()


@dataclass(frozen=True)
class RolloutCopyItem:
    key: str
    label: str
    body: str


def _stylize_matches(text: Text, pattern: re.Pattern, style: str) -> Text:
    plain = text.plain
    for match in pattern.finditer(plain):
        text.stylize(style, match.start(), match.end())
    return text


def _sorted_runs(runs: List[RunInfo]) -> List[RunInfo]:
    return sorted(runs, key=lambda run: run.run_id)


def _format_run_datetime(meta: Dict[str, Any]) -> str:
    return f"{meta.get('date', '')} {meta.get('time', '')}".strip()


def _text_to_plain(text: Text) -> str:
    return text.plain.rstrip()


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


# ----------------------------
# Screens
# ----------------------------
class BrowseRunsScreen(Screen):
    """Single-screen browser for environments, models, and runs."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index
        self._run_overview_cache: Dict[Path, RunOverviewStats] = {}

    def compose(self) -> ComposeResult:
        with Container():
            with Horizontal(classes="browser-columns"):
                yield Panel(
                    Label(Text("Eval Browser", style="bold"), classes="title"),
                    RunBrowserTree("Completed evals", id="run-browser-tree"),
                    classes="browser-tree-panel",
                )
                yield Panel(
                    Label(Text("Selection Details", style="bold"), classes="title"),
                    VerticalScroll(
                        Static("", id="run-browser-details", markup=False),
                        id="run-browser-details-scroll",
                        classes="surface-scroll",
                    ),
                    classes="browser-details-panel",
                )
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#run-browser-tree", Tree)
        tree.show_root = False
        tree.auto_expand = False
        tree.guide_depth = 2

        first_run_node = self._populate_tree(tree)
        tree.focus()

        if first_run_node is None:
            self.query_one("#run-browser-details", Static).update(
                Text("No completed evals found", style="dim")
            )
            return

        self.call_after_refresh(lambda: tree.move_cursor(first_run_node))

    def action_focus_next_pane(self) -> None:
        self.focus_next()

    def action_focus_prev_pane(self) -> None:
        self.focus_previous()

    def action_copy(self) -> None:
        tree = self.query_one("#run-browser-tree", Tree)
        node = tree.cursor_node
        payload = getattr(node, "data", None)
        if not isinstance(payload, BrowserNodeData):
            return
        label = getattr(node, "label", "")
        label_text = label.plain if isinstance(label, Text) else str(label)
        buffer = StringIO()
        Console(
            file=buffer,
            force_terminal=False,
            color_system=None,
            width=180,
        ).print(self._details_for(payload))
        self.app.push_screen(
            CopyScreen(
                label_text,
                buffer.getvalue().rstrip(),
                "completion",
                prompt_label="Selection",
                completion_label="Details",
                title="Copy Details",
            )
        )

    def _populate_tree(self, tree: Tree) -> Any:
        root = tree.root
        root.expand()

        if not self.index:
            root.add("No completed evals found", allow_expand=False)
            return None

        first_run_node = None
        sorted_env_ids = sorted(self.index.keys())
        for env_idx, env_id in enumerate(sorted_env_ids):
            models = self.index[env_id]
            total_runs = sum(len(runs) for runs in models.values())
            env_label = Text()
            env_label.append(env_id, style="bold")
            env_label.append("  ")
            env_label.append(f"{len(models)} models", style="dim")
            env_label.append("  ")
            env_label.append(f"{total_runs} runs", style="dim")
            env_node = root.add(
                env_label,
                data=BrowserNodeData(kind="env", env_id=env_id),
                expand=env_idx == 0,
            )
            for model_idx, model in enumerate(sorted(models.keys())):
                runs = _sorted_runs(models[model])
                model_label = Text()
                model_label.append(model, style="bold")
                model_label.append("  ")
                model_label.append(f"{len(runs)} runs", style="dim")
                model_node = env_node.add(
                    model_label,
                    data=BrowserNodeData(kind="model", env_id=env_id, model=model),
                    expand=env_idx == 0 and model_idx == 0,
                )
                for run in runs:
                    meta = run.load_metadata()
                    run_label = Text()
                    run_label.append(run.run_id, style="bold")
                    avg_reward = _numeric_reward(meta.get("avg_reward"))
                    if avg_reward is not None:
                        run_label.append("  ")
                        run_label.append(
                            _format_reward_value(avg_reward),
                            style=_reward_style(avg_reward),
                        )
                    run_node = model_node.add(
                        run_label,
                        data=BrowserNodeData(
                            kind="run",
                            env_id=env_id,
                            model=model,
                            run=run,
                        ),
                        allow_expand=False,
                    )
                    if first_run_node is None:
                        first_run_node = run_node
        return first_run_node

    @on(Tree.NodeHighlighted, "#run-browser-tree")
    def on_tree_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self.query_one("#run-browser-details", Static).update(
            self._details_for(getattr(event.node, "data", None))
        )

    @on(Tree.NodeSelected, "#run-browser-tree")
    def on_tree_selected(self, event: Tree.NodeSelected) -> None:
        payload = event.node.data
        if not isinstance(payload, BrowserNodeData):
            return
        if payload.kind == "run" and payload.run is not None:
            self.app.push_screen(ViewRunScreen(payload.run))
            return
        if event.node.allow_expand:
            event.node.toggle()

    def _details_for(self, payload: Any) -> Any:
        if not isinstance(payload, BrowserNodeData):
            return Text()

        if payload.kind == "run" and payload.run is not None:
            stats = self._run_overview_cache.get(payload.run.path)
            if stats is None:
                self._load_run_overview_stats(payload.run)
            return self._build_run_details(payload.run, stats)

        if payload.kind == "env":
            return self._build_env_details(payload.env_id)

        if payload.kind == "model":
            return self._build_model_details(payload.env_id, payload.model)

        return Text()

    @work(
        thread=True,
        group="run-overview",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_run_overview_stats(self, run: RunInfo) -> None:
        if run.path in self._run_overview_cache:
            return
        stats = _compute_run_overview_stats(run)
        self.app.call_from_thread(self._finish_loading_run_overview_stats, run, stats)

    def _finish_loading_run_overview_stats(
        self, run: RunInfo, stats: RunOverviewStats
    ) -> None:
        if not self.is_mounted:
            return
        self._run_overview_cache[run.path] = stats
        tree = self.query_one("#run-browser-tree", Tree)
        payload = getattr(getattr(tree, "cursor_node", None), "data", None)
        if not isinstance(payload, BrowserNodeData):
            return
        if payload.kind != "run" or payload.run is None:
            return
        if payload.run.path != run.path:
            return
        self.query_one("#run-browser-details", Static).update(
            self._build_run_details(run, stats)
        )

    def _build_env_details(self, env_id: str) -> Group:
        models = self.index.get(env_id, {})
        runs = [run for model_runs in models.values() for run in model_runs]
        rewards = [
            reward
            for run in runs
            for reward in [_numeric_reward(run.load_metadata().get("avg_reward"))]
            if reward is not None
        ]

        summary = Text()
        summary.append("Environment\n", style="bold dim")
        summary.append(env_id, style="bold")
        summary.append("\n")
        summary.append(f"{len(models)} models   {len(runs)} runs", style="dim")
        items: List[Any] = [
            summary,
            Text(""),
            _build_reward_distribution_table(rewards, "Run avg rewards"),
        ]

        if models:
            ranked_models = sorted(
                models.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:4]
            model_activity = Text()
            model_activity.append("Model activity\n", style="bold dim")
            for model, model_runs in ranked_models:
                model_rewards = [
                    reward
                    for run in model_runs
                    for reward in [
                        _numeric_reward(run.load_metadata().get("avg_reward"))
                    ]
                    if reward is not None
                ]
                model_activity.append(model, style="bold")
                model_activity.append(f"  {len(model_runs)} runs", style="dim")
                if model_rewards:
                    avg_reward = sum(model_rewards) / len(model_rewards)
                    model_activity.append("  avg ", style="dim")
                    model_activity.append(
                        f"{avg_reward:.3f}",
                        style=_reward_style(avg_reward),
                    )
                model_activity.append("\n")
            items.extend([Text(""), model_activity])

        return Group(*items)

    def _build_model_details(self, env_id: str, model: str) -> Group:
        runs = _sorted_runs(self.index.get(env_id, {}).get(model, []))
        rewards = [
            reward
            for run in runs
            for reward in [_numeric_reward(run.load_metadata().get("avg_reward"))]
            if reward is not None
        ]

        summary = Text()
        summary.append("Model\n", style="bold dim")
        summary.append(model, style="bold")
        summary.append("\n")
        summary.append(f"{env_id}   {len(runs)} runs", style="dim")
        items: List[Any] = [
            summary,
            Text(""),
            _build_reward_distribution_table(rewards, "Run avg rewards"),
        ]

        if runs:
            latest = runs[-1]
            best = max(
                runs,
                key=lambda run: (
                    _numeric_reward(run.load_metadata().get("avg_reward"))
                    if _numeric_reward(run.load_metadata().get("avg_reward"))
                    is not None
                    else float("-inf")
                ),
            )
            recent = Text()
            recent.append("Recent runs\n", style="bold dim")
            for label, run in (("latest", latest), ("best", best)):
                reward = _numeric_reward(run.load_metadata().get("avg_reward"))
                recent.append(label, style="bold")
                recent.append("  ")
                recent.append(run.run_id)
                if reward is not None:
                    recent.append("  reward ", style="dim")
                    recent.append(f"{reward:.3f}", style=_reward_style(reward))
                recent.append("\n")
            items.extend([Text(""), recent])

        return Group(*items)

    def _build_run_details(
        self,
        run: RunInfo,
        stats: Optional[RunOverviewStats] = None,
    ) -> Group:
        meta = run.load_metadata()
        rewards = stats.rewards if stats is not None else []

        summary = Text()
        summary.append("Run\n", style="bold dim")
        summary.append(run.run_id, style="bold")
        summary.append("\n")
        summary.append(f"{run.env_id}   {run.model}", style="dim")

        summary_parts: List[Tuple[str, str, Optional[str]]] = []
        created = _format_run_datetime(meta)
        if created:
            summary_parts.append(("created", created, None))
        avg_reward = _numeric_reward(meta.get("avg_reward"))
        if avg_reward is not None:
            summary_parts.append(
                ("avg reward", f"{avg_reward:.3f}", _reward_style(avg_reward))
            )
        if rewards:
            summary_parts.append(("rollouts", str(len(rewards)), None))
        elif meta.get("num_examples") not in (None, ""):
            summary_parts.append(("examples", str(meta.get("num_examples")), None))
        if summary_parts:
            summary.append("\n\n")
            for idx, (label, value, style) in enumerate(summary_parts):
                if idx:
                    summary.append("   ")
                summary.append(f"{label} ", style="bold")
                summary.append(value, style=style or "")

        pass_rates = []
        for key, prefix in (("pass_at_k", "pass@"), ("pass_all_k", "pass-all@")):
            values = meta.get(key)
            if isinstance(values, dict):
                for bucket, value in sorted(
                    values.items(), key=lambda item: str(item[0])
                ):
                    numeric = _numeric_reward(value)
                    if numeric is None:
                        continue
                    pass_rates.append((f"{prefix}{bucket}", numeric))
        pass_rate_text = Text()
        if pass_rates:
            pass_rate_text.append("Pass rates\n", style="bold dim")
            for idx, (label, value) in enumerate(pass_rates[:6]):
                if idx and idx % 3 == 0:
                    pass_rate_text.append("\n")
                elif idx:
                    pass_rate_text.append("   ")
                pass_rate_text.append(f"{label} ", style="bold")
                pass_rate_text.append(f"{value:.3f}", style=_reward_style(value))

        items: List[Any] = [summary, Text("")]
        if stats is None:
            loading = Text("Loading rollout metrics…", style="dim")
            loading.append(
                "\nOpen the run to inspect rollouts immediately.", style="dim"
            )
            items.append(loading)
        else:
            reward_summary = _build_reward_distribution_table(
                stats.rewards,
                "Rollout rewards",
            )
            items.extend(
                [
                    reward_summary,
                    Text(""),
                    _build_metric_summary_table(stats.metric_summaries),
                ]
            )
        if pass_rate_text.plain:
            items.extend([Text(""), pass_rate_text])

        return Group(*items)


class ViewRunScreen(Screen):
    """Screen for viewing run details and rollouts."""

    COMPACT_LAYOUT_WIDTH = 150

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("left,p", "prev_record", "Prev rollout"),
        Binding("right,n", "next_record", "Next rollout"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("e", "expand_all", "Expand all"),
        Binding("x", "collapse_all", "Collapse all"),
        Binding("s", "search", "Search"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = LazyRunResults(run)
        self._record_count = self.records.count_hint()
        self.current_record_idx = 0
        self._prompt_lines: List[str] = []
        self._completion_lines: List[str] = []
        self._prompt_text: str = ""
        self._completion_text: str = ""
        self._highlight_regex: Optional[re.Pattern] = None
        self._highlight_column: Optional[str] = None
        self._highlight_timer = None
        self._previous_animation_level: Optional[AnimationLevel] = None
        if self.records:
            self._set_record_text_state(self.records[self.current_record_idx])

    def compose(self) -> ComposeResult:
        completion_sections = (
            self._completion_sections(self.records[self.current_record_idx])
            if self.records
            else []
        )
        with Container(id="view-container"):
            with Panel(classes="metadata-panel"):
                with Horizontal(classes="metadata-layout"):
                    yield Static("", id="metadata-summary", markup=False)
                    yield Static("", id="metadata-metrics", markup=False)
                    yield Static("", id="metadata-reward", markup=False)
            with Horizontal(classes="view-columns"):
                with Panel(id="rollouts-panel", classes="rollouts-panel"):
                    yield Label(Text("Rollouts", style="bold"), classes="column-header")
                    yield Label("", id="rollout-summary", classes="subtitle")
                    yield OptionList(id="rollout-list")
                with Panel(id="history-panel", classes="history-panel"):
                    yield Label(
                        Text("Completion History", style="bold"),
                        classes="column-header",
                    )
                    yield Static(
                        "", id="history-summary", classes="subtitle", markup=False
                    )
                    yield VerticalScroll(*completion_sections, id="completion-scroll")
                with Panel(id="details-panel", classes="details-panel"):
                    yield Label(Text("Details", style="bold"), classes="column-header")
                    with TabbedContent(initial="details-task", id="details-tabs"):
                        with TabPane("Task", id="details-task"):
                            yield VerticalScroll(
                                Static("", id="task-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Score", id="details-score"):
                            yield VerticalScroll(
                                Static("", id="score-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Usage", id="details-usage"):
                            yield VerticalScroll(
                                Static("", id="usage-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Info", id="details-info"):
                            yield VerticalScroll(
                                Static("", id="info-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
        yield Footer()

    def _build_header_summary_text(self) -> Text:
        meta = self.run.load_metadata()
        lines: List[Text] = []

        lines.append(Text("Run Summary", style="bold dim"))

        identity = Text()
        identity.append("Environment: ", style="bold")
        identity.append(str(self.run.env_id))
        identity.append("   ")
        identity.append("Model: ", style="bold")
        identity.append(str(self.run.model))
        identity.append("   ")
        identity.append("Run ID: ", style="bold")
        identity.append(str(self.run.run_id))
        lines.append(identity)

        progress = Text()
        progress.append("Record: ", style="bold")
        progress.append(self._record_progress_label())
        progress.append("   ")
        progress.append("Examples: ", style="bold")
        progress.append(str(meta.get("num_examples", "")))
        progress.append("   ")
        progress.append("Rollouts/ex: ", style="bold")
        progress.append(str(meta.get("rollouts_per_example", "")))
        date_text = f"{str(meta.get('date', ''))} {str(meta.get('time', ''))}".strip()
        if date_text:
            progress.append("   ")
            progress.append("Date: ", style="bold")
            progress.append(date_text)
        lines.append(progress)

        usage = meta.get("usage")
        sampling_args = meta.get("sampling_args", {})
        usage_items: List[Tuple[str, str]] = []
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if input_tokens is not None:
                usage_items.append(("Avg input tokens", format_numeric(input_tokens)))
            if output_tokens is not None:
                usage_items.append(("Avg output tokens", format_numeric(output_tokens)))
        max_tokens = sampling_args.get("max_tokens")
        if max_tokens not in (None, ""):
            usage_items.append(("Max tokens", str(max_tokens)))
        temperature = sampling_args.get("temperature")
        if temperature not in (None, ""):
            usage_items.append(("Temperature", str(temperature)))

        if usage_items:
            usage_line = Text()
            for idx, (label, value) in enumerate(usage_items):
                if idx:
                    usage_line.append("   ")
                usage_line.append(f"{label}: ", style="bold")
                usage_line.append(value)
            lines.append(usage_line)

        return Text("\n").join(lines)

    def _build_history_summary_text(
        self, record: Dict[str, Any], *, include_hints: bool = True
    ) -> Text:
        completion = record.get("completion")
        if not isinstance(completion, list) or not completion:
            return Text()

        groups = self._history_groups(completion)
        tool_groups = sum(
            1 for group in groups if group.get("kind") == "assistant-tools"
        )
        user_messages = sum(
            1
            for group in groups
            if isinstance(group.get("message"), dict)
            and group["message"].get("role") == "user"
        )
        parts: List[Tuple[str, str]] = [
            (f"{len(groups)} events", "bold"),
            ("  ", ""),
            (f"{tool_groups} tool exchanges", "dim"),
            ("  ", ""),
            (f"{user_messages} user turns", "dim"),
        ]
        if include_hints:
            parts.extend(
                [
                    ("  ", ""),
                    ("Enter toggles", "dim"),
                    ("  ", ""),
                    ("PgUp/PgDn scroll", "dim"),
                ]
            )
        return Text.assemble(*parts)

    def _build_header_metric_text(self) -> Text:
        meta = self.run.load_metadata()
        stats: List[Tuple[str, Any]] = []

        pass_at_k = meta.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            for key in sorted(pass_at_k.keys(), key=lambda item: str(item)):
                stats.append((f"pass@{key}", pass_at_k[key]))

        pass_all_k = meta.get("pass_all_k")
        if isinstance(pass_all_k, dict):
            for key in sorted(pass_all_k.keys(), key=lambda item: str(item)):
                stats.append((f"pass-all@{key}", pass_all_k[key]))

        avg_metrics = meta.get("avg_metrics")
        preferred_metric_keys = [
            ("evaluate_tau2_task", "task"),
            ("num_turns", "turns"),
            ("total_tool_calls", "tools"),
            ("num_steps", "steps"),
            ("num_errors", "errors"),
        ]
        if isinstance(avg_metrics, dict):
            for key, label in preferred_metric_keys:
                if key in avg_metrics:
                    stats.append((label, avg_metrics[key]))

        if not stats:
            return Text()

        out = Text()
        out.append("Run Metrics\n", style="bold dim")
        for idx, (label, value) in enumerate(stats[:6]):
            if idx and idx % 3 == 0:
                out.append("\n")
            elif idx:
                out.append("   ")
            out.append(f"{label} ", style="bold")
            out.append(_format_compact_metric(value))

        pass_threshold = meta.get("pass_threshold")
        if pass_threshold not in (None, ""):
            out.append("\n")
            out.append("threshold ", style="bold")
            out.append(_format_compact_metric(pass_threshold))
        return out

    def _build_reward_text(
        self,
        record: Dict[str, Any],
        *,
        heading: str,
        multiline: bool,
        limit: Optional[int] = None,
    ) -> Text:
        reward = record.get("reward")
        out = Text()
        out.append(f"{heading}\n", style="bold dim")
        out.append(_format_reward_value(reward), style=_reward_style(reward))

        breakdown = self._extract_reward_metrics(record)
        if breakdown:
            breakdown = breakdown[:limit] if limit is not None else breakdown
            if multiline:
                out.append("\n\nBreakdown\n", style="bold dim")
                width = max(len(name) for name, _ in breakdown)
                for name, value in breakdown:
                    out.append(name.ljust(width), style="bold")
                    out.append("  ")
                    out.append(_format_reward_value(value), style=_reward_style(value))
                    out.append("\n")
            else:
                out.append("\n")
                for idx, (name, value) in enumerate(breakdown):
                    if idx:
                        out.append("   ")
                    out.append(f"{name} ", style="bold")
                    out.append(_format_reward_value(value), style=_reward_style(value))
        return out

    def _build_header_reward_text(self, record: Dict[str, Any]) -> Text:
        return self._build_reward_text(
            record,
            heading="Current Reward",
            multiline=False,
            limit=3,
        )

    def on_mount(self) -> None:
        app = cast(App[Any], self.app)
        self._previous_animation_level = app.animation_level
        app.animation_level = "none"
        self._populate_rollout_list()
        self.query_one("#rollout-list", OptionList).focus()
        self.update_display()
        self._update_responsive_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(event.size.width)

    def on_unmount(self) -> None:
        self.records.close()
        if self._previous_animation_level is not None:
            cast(App[Any], self.app).animation_level = self._previous_animation_level

    def _available_record_count(self) -> int:
        if self.is_mounted:
            return self.query_one("#rollout-list", OptionList).option_count
        if self._record_count is not None:
            return self._record_count
        return 1 if self.records else 0

    def _record_progress_label(self) -> str:
        total = "?" if self._record_count is None else str(self._record_count)
        return f"{self.current_record_idx + 1}/{total}"

    def _set_rollout_option_count(self, count: int) -> None:
        rollout_list = self.query_one("#rollout-list", OptionList)
        while rollout_list.option_count < count:
            idx = rollout_list.option_count
            rollout_list.add_option(
                Option(self._build_rollout_prompt(idx), id=str(idx))
            )

    def _hydrate_rollout_option(self, index: int) -> None:
        rollout_list = self.query_one("#rollout-list", OptionList)
        if not (0 <= index < rollout_list.option_count):
            return
        rollout_list.replace_option_prompt_at_index(
            index,
            self._build_rollout_prompt(index, self.records[index]),
        )

    @work(
        thread=True,
        group="rollout-count",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_record_count(self) -> None:
        count = _count_result_records(self.run.path / "results.jsonl")
        self.app.call_from_thread(self._finish_loading_record_count, count)

    def _finish_loading_record_count(self, count: int) -> None:
        self._record_count = count
        if not self.is_mounted:
            return
        self._set_rollout_option_count(count)
        rollout_list = self.query_one("#rollout-list", OptionList)
        rollout_list.highlighted = self.current_record_idx
        rollout_list.scroll_to_highlight()
        self.update_display()

    def _populate_rollout_list(self) -> None:
        rollout_list = self.query_one("#rollout-list", OptionList)
        rollout_list.clear_options()

        if not self.records:
            return

        self._record_count = len(self.records)
        for idx in range(self._record_count):
            rollout_list.add_option(
                Option(self._build_rollout_prompt(idx, self.records[idx]), id=str(idx))
            )
        rollout_list.highlighted = self.current_record_idx
        rollout_list.scroll_to_highlight()

    def _build_rollout_prompt(
        self,
        idx: int,
        record: Optional[Dict[str, Any]] = None,
    ) -> Text:
        label = Text()
        label.append(f"#{idx}", style="bold")
        if not record:
            return label

        reward = record.get("reward")
        label.append("  ")
        label.append("reward ", style="dim")
        label.append(_format_reward_value(reward), style=_reward_style(reward))
        label.append("\n")
        label.append(_truncate_preview(self._record_preview(record), 38), style="dim")
        return label

    def _record_preview(self, record: Dict[str, Any]) -> str:
        completion = record.get("completion")
        if isinstance(completion, list) and completion:
            for group in reversed(self._history_groups(completion)):
                message = group["message"]
                if group.get("kind") == "assistant-tools":
                    preview = _tool_group_preview(
                        message,
                        group["tool_outputs"],
                    )
                else:
                    preview = _format_message_preview(message)
                if preview:
                    return preview
        completion_preview = _raw_preview(completion, limit=56)
        if completion_preview:
            return completion_preview

        error_preview = _error_preview(record.get("error"))
        if error_preview:
            return error_preview

        prompt = record.get("prompt")
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[-1], dict):
                preview = _format_message_preview(prompt[-1])
                if preview:
                    return preview
            prompt_preview = _raw_preview(prompt[-1], limit=56)
            if prompt_preview:
                return prompt_preview
        prompt_preview = _raw_preview(prompt, limit=56)
        if prompt_preview:
            return prompt_preview
        return ""

    def _format_prompt_or_completion(self, prompt_or_completion: Any) -> Text:
        out = Text()
        if not isinstance(prompt_or_completion, list):
            out.append(str(prompt_or_completion))
            return out

        for message in prompt_or_completion:
            if not isinstance(message, dict):
                out.append(str(message))
                out.append("\n\n")
                continue
            role = str(message.get("role", ""))
            content = _stringify_message_content(message.get("content", ""))
            if role == "assistant":
                out.append("assistant: ", style="bold")
            elif role == "tool":
                out.append("tool result: ", style="bold dim")
            else:
                out.append(f"{role}: ", style="bold dim")
            out.append(content)
            out.append("\n")

            for tool_call in _parse_tool_calls(message.get("tool_calls")):
                name, arguments, _ = _tool_call_parts(tool_call)
                out.append("\ntool call: ", style="bold")
                out.append(name)
                out.append("\n")
                out.append(arguments)
                out.append("\n")

            out.append("\n")

        return out

    def _set_record_text_state(self, record: Dict[str, Any]) -> None:
        prompt_text = self._format_prompt_or_completion(record.get("prompt", ""))
        completion_text = self._format_prompt_or_completion(
            record.get("completion", "")
        )

        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")

        self._prompt_text = prompt_text.plain
        self._completion_text = completion_text.plain
        self._prompt_lines = prompt_text.plain.split("\n")
        self._completion_lines = completion_text.plain.split("\n")

    def update_display(self, *, focus_history: bool = False) -> None:
        if not self.records:
            return

        record = self.records[self.current_record_idx]
        self._set_record_text_state(record)
        task_text = self._build_task_text(record)
        score_text = self._build_score_text(record)
        usage_text = self._build_usage_text(record)
        info_text = self._build_info_text(record)

        self.query_one("#metadata-summary", Static).update(
            self._build_header_summary_text()
        )
        self.query_one("#metadata-metrics", Static).update(
            self._build_header_metric_text()
        )
        self.query_one("#metadata-reward", Static).update(
            self._build_header_reward_text(record)
        )
        self.query_one("#history-summary", Static).update(
            self._build_history_summary_text(record)
        )
        self.query_one("#task-content", Static).update(task_text)
        self.query_one("#score-content", Static).update(score_text)
        self.query_one("#usage-content", Static).update(usage_text)
        self.query_one("#info-content", Static).update(info_text)
        self._update_rollout_summary(record)
        self._rebuild_completion_sections(record, focus_history)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_prev_record(self) -> None:
        self._move_record_cursor(-1)

    def action_next_record(self) -> None:
        self._move_record_cursor(1)

    def _move_record_cursor(self, delta: int) -> None:
        record_count = self._available_record_count()
        if record_count <= 0:
            return
        new_index = (self.current_record_idx + delta) % record_count
        rollout_list = self.query_one("#rollout-list", OptionList)
        rollout_list.highlighted = new_index
        rollout_list.scroll_to_highlight()
        self._set_current_record(new_index)

    def action_search(self) -> None:
        if not self.records:
            return
        self.app.push_screen(
            SearchScreen(self._prompt_lines, self._completion_lines),
            self._handle_search_result,
        )

    def action_copy(self) -> None:
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        self.app.push_screen(
            RolloutCopyScreen(
                self._build_rollout_copy_items(record),
                start_key=self._rollout_copy_start_key(),
                title=f"Copy Rollout #{self.current_record_idx}",
            )
        )

    def action_expand_all(self) -> None:
        container = self.query_one("#completion-scroll", VerticalScroll)
        for section in container.query(Collapsible):
            section.collapsed = False
        self._focus_primary_content()

    def action_collapse_all(self) -> None:
        container = self.query_one("#completion-scroll", VerticalScroll)
        for section in container.query(Collapsible):
            section.collapsed = True
        self._focus_primary_content(prefer_expanded=False)

    def action_focus_next_pane(self) -> None:
        self.focus_next()

    def action_focus_prev_pane(self) -> None:
        self.focus_previous()

    def action_history_page_up(self) -> None:
        self.query_one("#completion-scroll", VerticalScroll).scroll_page_up(
            animate=False
        )

    def action_history_page_down(self) -> None:
        self.query_one("#completion-scroll", VerticalScroll).scroll_page_down(
            animate=False
        )

    def action_history_home(self) -> None:
        self.query_one("#completion-scroll", VerticalScroll).scroll_home(animate=False)

    def action_history_end(self) -> None:
        self.query_one("#completion-scroll", VerticalScroll).scroll_end(animate=False)

    def _handle_search_result(self, result: Optional[SearchResult]) -> None:
        if result is not None:
            self._set_highlight(result)

    def _set_highlight(
        self, result: Optional[SearchResult], *, repaint: bool = True
    ) -> None:
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
            self._highlight_timer = None
        self._highlight_regex = None
        self._highlight_column = None

        if result is not None:
            try:
                self._highlight_regex = re.compile(result.pattern, re.IGNORECASE)
            except re.error:
                return
            self._highlight_column = result.column
            self._highlight_timer = self.set_timer(
                3.0, lambda: self._set_highlight(None)
            )

        if repaint and self.is_mounted:
            self.update_display()

    def _build_rollout_summary_text(self, record: Dict[str, Any]) -> Text:
        return Text.assemble(
            (self._record_progress_label(), "bold"),
            ("  ", ""),
            ("reward ", "dim"),
            (
                _format_reward_value(record.get("reward")),
                _reward_style(record.get("reward")),
            ),
        )

    def _update_rollout_summary(self, record: Dict[str, Any]) -> None:
        self.query_one("#rollout-summary", Label).update(
            self._build_rollout_summary_text(record)
        )

    def _update_responsive_layout(self, width: int) -> None:
        compact = width < self.COMPACT_LAYOUT_WIDTH
        rollouts_panel = self.query_one("#rollouts-panel", Panel)
        details_panel = self.query_one("#details-panel", Panel)
        rollouts_panel.display = not compact
        details_panel.display = not compact
        if compact and (
            rollouts_panel.has_focus_within or details_panel.has_focus_within
        ):
            self.call_after_refresh(
                lambda: self._focus_primary_content(prefer_expanded=False)
            )

    def _set_current_record(self, index: int, *, focus_history: bool = False) -> None:
        if not (0 <= index < self._available_record_count()):
            return
        self.current_record_idx = index
        self._hydrate_rollout_option(index)
        self._set_highlight(None, repaint=False)
        self.update_display(focus_history=focus_history)
        self.query_one("#completion-scroll", VerticalScroll).scroll_y = 0
        for scroll in self.query(".details-scroll"):
            scroll.scroll_y = 0

    @on(OptionList.OptionHighlighted, "#rollout-list")
    def on_rollout_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        idx = int(event.option_id)
        if idx != self.current_record_idx:
            self._set_current_record(idx)

    @on(OptionList.OptionSelected, "#rollout-list")
    def on_rollout_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        self._set_current_record(int(event.option_id), focus_history=True)

    def _history_section_data(self, record: Dict[str, Any]) -> List[HistorySectionData]:
        sections: List[HistorySectionData] = [
            HistorySectionData(
                title="Initial Prompt",
                body=self._prompt_text,
                column="prompt",
                collapsed=True,
                classes="history-section prompt-section",
            )
        ]
        completion = record.get("completion")
        if not isinstance(completion, list) or not completion:
            sections.append(
                HistorySectionData(
                    title="Completion",
                    body=self._completion_text,
                    column="completion",
                    collapsed=False,
                    classes="history-section assistant-section",
                )
            )
            return sections

        for idx, group in enumerate(self._history_groups(completion), start=1):
            message = group["message"]
            if group["kind"] != "assistant-tools":
                role = str(message.get("role", "message"))
                title = f"{idx}. {role}"
                preview = _format_message_preview(message)
                if preview:
                    title += f"  {preview}"

                sections.append(
                    HistorySectionData(
                        title=title,
                        body=_stringify_message_content(
                            message.get("content", "")
                        ).strip(),
                        column="completion",
                        collapsed=True,
                        classes=(
                            "history-section tool-section"
                            if role == "tool"
                            else (
                                "history-section prompt-section"
                                if role not in ("assistant", "tool")
                                else "history-section assistant-section"
                            )
                        ),
                    )
                )
                continue

            tool_calls = group["tool_calls"]
            tool_outputs = group["tool_outputs"]
            preview = _tool_group_preview(
                message, tool_outputs
            ) or _format_message_preview(message)
            title = f"{idx}. assistant"
            if preview:
                title += f"  {preview}"

            body = _stringify_message_content(message.get("content", "")).strip()
            collapsed = True
            if self._highlight_regex and self._highlight_column == "completion":
                collapsed = not (body and self._highlight_regex.search(body))
                if collapsed:
                    for tool_call in tool_calls:
                        name, arguments, _ = _tool_call_parts(tool_call)
                        if self._highlight_regex.search(
                            name
                        ) or self._highlight_regex.search(arguments):
                            collapsed = False
                            break
                if collapsed:
                    for output in tool_outputs:
                        output_text = (
                            _stringify_message_content(output.get("content", ""))
                            if isinstance(output, dict)
                            else str(output)
                        )
                        if self._highlight_regex.search(output_text):
                            collapsed = False
                            break

            nested_sections: List[HistorySectionData] = []
            used_output_indexes: set[int] = set()
            for tool_idx, tool_call in enumerate(tool_calls, start=1):
                name, arguments, call_id = _tool_call_parts(tool_call)
                matched_output = None
                if call_id is not None:
                    for output_idx, candidate in enumerate(tool_outputs):
                        if (
                            isinstance(candidate, dict)
                            and candidate.get("tool_call_id") == call_id
                        ):
                            matched_output = candidate
                            used_output_indexes.add(output_idx)
                            break
                if matched_output is None:
                    for output_idx, candidate in enumerate(tool_outputs):
                        if output_idx not in used_output_indexes:
                            matched_output = candidate
                            used_output_indexes.add(output_idx)
                            break

                output_text = (
                    _stringify_message_content(matched_output.get("content", ""))
                    if isinstance(matched_output, dict)
                    else (str(matched_output) if matched_output is not None else "")
                )
                nested_sections.append(
                    HistorySectionData(
                        title=(
                            f"tool {tool_idx}  {name}  ... "
                            f"{_tool_output_preview(matched_output)}"
                        ),
                        body="\n".join(
                            [
                                "Call",
                                arguments,
                                "",
                                "Output",
                                output_text,
                            ]
                        ),
                        column="completion",
                        collapsed=collapsed or tool_idx > 1,
                        classes="history-section tool-call-section nested-section",
                    )
                )

            for extra_idx, output_message in enumerate(tool_outputs, start=1):
                if (extra_idx - 1) in used_output_indexes:
                    continue
                output_text = (
                    _stringify_message_content(output_message.get("content", ""))
                    if isinstance(output_message, dict)
                    else str(output_message)
                )
                nested_sections.append(
                    HistorySectionData(
                        title=(
                            f"tool output {len(nested_sections) + 1}  "
                            f"{_tool_output_preview(output_message)}"
                        ),
                        body=output_text,
                        column="completion",
                        collapsed=True,
                        classes="history-section tool-section nested-section",
                    )
                )

            sections.append(
                HistorySectionData(
                    title=title,
                    body=body,
                    column="completion",
                    collapsed=collapsed,
                    classes="history-section assistant-section",
                    nested_sections=tuple(nested_sections),
                )
            )

        return sections

    def _completion_sections(self, record: Dict[str, Any]) -> List[Collapsible]:
        return [
            self._make_section(section)
            for section in self._history_section_data(record)
        ]

    def _rebuild_completion_sections(
        self, record: Dict[str, Any], focus_history: bool = False
    ) -> None:
        if not self.is_mounted:
            return

        container = self.query_one("#completion-scroll", VerticalScroll)
        container.remove_children()
        container.mount(*self._completion_sections(record))
        if focus_history:
            self.call_after_refresh(self._focus_primary_content)

    def _rollout_copy_start_key(self) -> str:
        if self.is_mounted and self.query_one("#details-panel", Panel).has_focus_within:
            return f"details:{self._active_details_tab_id()}"
        if self.is_mounted and self.query_one("#history-panel", Panel).has_focus_within:
            return "history"
        if (
            self.is_mounted
            and self.query_one("#rollouts-panel", Panel).has_focus_within
        ):
            return "rollout"
        return "snapshot"

    def _active_details_tab_id(self) -> str:
        if not self.is_mounted:
            return "details-task"
        active = self.query_one("#details-tabs", TabbedContent).active
        return active or "details-task"

    def _detail_copy_sections(
        self, record: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        sections = [
            ("details-task", "Task", _text_to_plain(self._build_task_text(record))),
            ("details-score", "Score", _text_to_plain(self._build_score_text(record))),
            ("details-usage", "Usage", _text_to_plain(self._build_usage_text(record))),
            ("details-info", "Info", _text_to_plain(self._build_info_text(record))),
        ]
        return [section for section in sections if section[2]]

    def _render_detail_copy_text(self, sections: List[Tuple[str, str, str]]) -> str:
        return "\n\n".join(f"{label}\n{body}" for _, label, body in sections if body)

    def _render_history_section_copy_text(
        self, section: HistorySectionData, *, depth: int = 0
    ) -> str:
        indent = "  " * depth
        parts = [f"{indent}{section.title}"]
        if section.body:
            parts.append(_indent_block(section.body, f"{indent}  "))
        parts.extend(
            self._render_history_section_copy_text(child, depth=depth + 1)
            for child in section.nested_sections
        )
        return "\n\n".join(part for part in parts if part)

    def _render_history_copy_text(self, sections: List[HistorySectionData]) -> str:
        return "\n\n".join(
            self._render_history_section_copy_text(section) for section in sections
        )

    def _append_history_copy_items(
        self,
        items: List[RolloutCopyItem],
        sections: List[HistorySectionData],
        *,
        depth: int = 0,
        prefix: str = "history",
    ) -> None:
        for idx, section in enumerate(sections, start=1):
            key = f"{prefix}:{idx}"
            indent = "  " * depth
            items.append(
                RolloutCopyItem(
                    key=key,
                    label=f"History: {indent}{section.title}",
                    body=self._render_history_section_copy_text(section),
                )
            )
            self._append_history_copy_items(
                items,
                list(section.nested_sections),
                depth=depth + 1,
                prefix=key,
            )

    def _build_rollout_snapshot_text(
        self,
        record: Dict[str, Any],
        history_sections: List[HistorySectionData],
        detail_sections: List[Tuple[str, str, str]],
    ) -> str:
        blocks = [
            _text_to_plain(self._build_header_summary_text()),
            _text_to_plain(self._build_header_metric_text()),
            _text_to_plain(self._build_header_reward_text(record)),
            f"Current Rollout\n{self._build_rollout_prompt(self.current_record_idx, record).plain}",
        ]

        history_summary = _text_to_plain(
            self._build_history_summary_text(record, include_hints=False)
        )
        history_text = self._render_history_copy_text(history_sections)
        history_parts = ["Completion History"]
        if history_summary:
            history_parts.append(history_summary)
        if history_text:
            history_parts.append(history_text)
        blocks.append("\n\n".join(history_parts))

        active_tab_id = self._active_details_tab_id()
        active_tab_label = next(
            (
                label
                for detail_id, label, _ in [
                    ("details-task", "Task", ""),
                    ("details-score", "Score", ""),
                    ("details-usage", "Usage", ""),
                    ("details-info", "Info", ""),
                ]
                if detail_id == active_tab_id
            ),
            "Task",
        )
        detail_text = self._render_detail_copy_text(detail_sections)
        detail_heading = f"Details (active: {active_tab_label})"
        if detail_text:
            blocks.append(f"{detail_heading}\n\n{detail_text}")
        else:
            blocks.append(detail_heading)

        return "\n\n".join(block for block in blocks if block)

    def _build_rollout_copy_items(
        self, record: Dict[str, Any]
    ) -> List[RolloutCopyItem]:
        history_sections = self._history_section_data(record)
        detail_sections = self._detail_copy_sections(record)
        items: List[RolloutCopyItem] = [
            RolloutCopyItem(
                key="snapshot",
                label="Full rollout snapshot",
                body=self._build_rollout_snapshot_text(
                    record,
                    history_sections,
                    detail_sections,
                ),
            ),
            RolloutCopyItem(
                key="rollout",
                label="Rollout card",
                body=self._build_rollout_prompt(self.current_record_idx, record).plain,
            ),
            RolloutCopyItem(
                key="summary",
                label="Run summary",
                body=_text_to_plain(self._build_header_summary_text()),
            ),
        ]

        run_metrics = _text_to_plain(self._build_header_metric_text())
        if run_metrics:
            items.append(
                RolloutCopyItem(
                    key="metrics",
                    label="Run metrics",
                    body=run_metrics,
                )
            )

        reward_text = _text_to_plain(self._build_header_reward_text(record))
        if reward_text:
            items.append(
                RolloutCopyItem(
                    key="reward",
                    label="Current reward",
                    body=reward_text,
                )
            )

        history_text = self._render_history_copy_text(history_sections)
        if history_text:
            items.append(
                RolloutCopyItem(
                    key="history",
                    label="Completion history",
                    body=history_text,
                )
            )

        detail_text = self._render_detail_copy_text(detail_sections)
        if detail_text:
            items.append(
                RolloutCopyItem(
                    key="details",
                    label="Details panel",
                    body=detail_text,
                )
            )

        for detail_id, label, body in detail_sections:
            items.append(
                RolloutCopyItem(
                    key=f"details:{detail_id}",
                    label=f"Details: {label}",
                    body=body,
                )
            )

        self._append_history_copy_items(items, history_sections)
        return items

    def _history_groups(self, completion: List[Any]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(completion):
            message = completion[idx]
            if not isinstance(message, dict):
                idx += 1
                continue
            if message.get("role") == "assistant":
                tool_calls = _parse_tool_calls(message.get("tool_calls"))
                if tool_calls:
                    tool_outputs: List[Any] = []
                    next_idx = idx + 1
                    while next_idx < len(completion):
                        next_message = completion[next_idx]
                        if not isinstance(next_message, dict):
                            break
                        if next_message.get("role") != "tool":
                            break
                        tool_outputs.append(next_message)
                        next_idx += 1
                    groups.append(
                        {
                            "kind": "assistant-tools",
                            "message": message,
                            "tool_calls": tool_calls,
                            "tool_outputs": tool_outputs,
                        }
                    )
                    idx = next_idx
                    continue
            groups.append({"kind": "message", "message": message})
            idx += 1
        return groups

    def _make_section(self, section: HistorySectionData) -> Collapsible:
        collapsed = section.collapsed
        if (
            self._highlight_regex
            and self._highlight_column == section.column
            and self._highlight_regex.search(section.body)
        ):
            collapsed = False
        children: List[Any] = []
        if section.body or not section.nested_sections:
            text = Text(section.body)
            if self._highlight_regex and self._highlight_column == section.column:
                _stylize_matches(text, self._highlight_regex, "reverse")
            content = Static(
                text,
                classes="section-body",
                markup=False,
            )
            children.append(content)
        children.extend(
            self._make_section(nested_section)
            for nested_section in section.nested_sections
        )
        return Collapsible(
            *children,
            title=section.title,
            collapsed=collapsed,
            classes=section.classes,
        )

    def _focus_primary_content(self, *, prefer_expanded: bool = True) -> None:
        container = self.query_one("#completion-scroll", VerticalScroll)
        sections = [
            child for child in container.children if isinstance(child, Collapsible)
        ]
        if not sections:
            self.query_one("#rollout-list", OptionList).focus()
            return
        target = sections[0]
        if prefer_expanded:
            target = next(
                (section for section in sections if not section.collapsed),
                target,
            )
        title_widget = next(iter(target.children), None)
        if title_widget is not None and getattr(title_widget, "can_focus", False):
            title_widget.focus()

    @on(Collapsible.Expanded)
    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        collapsible = event.collapsible
        if not collapsible.has_class("history-section"):
            return
        collapsible.remove_class("expand-settle")
        collapsible.add_class("just-expanded")
        self.set_timer(
            0.04,
            lambda: self._shift_expand_pulse(collapsible),
        )
        self.set_timer(
            0.10,
            lambda: self._clear_expand_pulse(collapsible),
        )
        collapsible.call_after_refresh(
            lambda: collapsible.scroll_visible(duration=0.06, easing="out_cubic")
        )

    def _shift_expand_pulse(self, collapsible: Collapsible) -> None:
        if not collapsible.is_mounted:
            return
        collapsible.remove_class("just-expanded")
        collapsible.add_class("expand-settle")

    def _clear_expand_pulse(self, collapsible: Collapsible) -> None:
        if not collapsible.is_mounted:
            return
        collapsible.remove_class("just-expanded")
        collapsible.remove_class("expand-settle")

    def _build_score_text(self, record: Dict[str, Any]) -> Text:
        out = self._build_reward_text(record, heading="Reward", multiline=True)

        record_metrics = record.get("metrics")
        if isinstance(record_metrics, dict) and record_metrics:
            out.append("\nRecord metrics\n", style="bold dim")
            for key in sorted(record_metrics.keys()):
                value = record_metrics[key]
                out.append(f"{key}: ", style="bold")
                out.append(_format_compact_metric(value))
                out.append("\n")

        return out

    def _extract_reward_metrics(self, record: Dict[str, Any]) -> List[Tuple[str, Any]]:
        return sorted(_extract_numeric_metric_values(record).items())

    def _build_task_text(self, record: Dict[str, Any]) -> Text:
        out = Text()
        self._append_context_section(out, "Task", record.get("task"))
        self._append_context_section(out, "Answer", record.get("answer"))
        self._append_context_section(
            out, "Stop condition", record.get("stop_condition")
        )
        return out

    def _build_usage_text(self, record: Dict[str, Any]) -> Text:
        out = Text()
        token_usage = record.get("token_usage")
        if isinstance(token_usage, dict):
            usage_lines = []
            input_tokens = token_usage.get("input_tokens")
            output_tokens = token_usage.get("output_tokens")
            if input_tokens is not None:
                usage_lines.append(f"input_tokens: {format_numeric(input_tokens)}")
            if output_tokens is not None:
                usage_lines.append(f"output_tokens: {format_numeric(output_tokens)}")
            self._append_context_section(out, "Tokens", "\n".join(usage_lines))

        timing = record.get("timing")
        if isinstance(timing, dict):
            timing_lines = []
            for key in ("generation_ms", "scoring_ms", "total_ms"):
                value = timing.get(key)
                if value is not None:
                    timing_lines.append(f"{key}: {_format_compact_metric(value)}")
            self._append_context_section(out, "Timing", "\n".join(timing_lines))
        return out

    def _build_info_text(self, record: Dict[str, Any]) -> Text:
        out = Text()
        error = record.get("error")
        if error not in (None, ""):
            self._append_context_section(out, "Error", error)

        info = record.get("info")
        if info not in (None, {}, ""):
            self._append_context_section(out, "Info", format_info_for_details(info))
        return out

    def _append_context_section(self, out: Text, title: str, value: Any) -> None:
        if value in (None, "", {}):
            return
        if out.plain:
            out.append("\n\n")
        out.append(f"{title}\n", style="bold dim")
        if isinstance(value, Text):
            out += value
        else:
            out.append(str(value))


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

    .copy-hint {
        color: $text-muted;
        margin-bottom: 0;
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

    .metadata-layout {
        height: auto;
        width: 100%;
    }

    #metadata-summary {
        width: 2fr;
        padding: 0 1;
    }

    #metadata-metrics {
        width: 1.5fr;
        padding: 0 1;
        color: $text;
    }

    #metadata-reward {
        width: 1fr;
        padding: 0 1;
        text-align: left;
    }
    
    .view-columns {
        height: 1fr;
        layout: horizontal;
    }
    
    .rollouts-panel {
        width: 34;
        height: 100%;
        layout: vertical;
    }

    #rollout-list {
        height: 1fr;
        max-height: 100%;
        background: $surface;
    }

    .history-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }
    
    .column-header {
        height: auto;
        margin-bottom: 0;
        text-align: left;
        text-style: bold;
    }
    
    #completion-scroll {
        layout: vertical;
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-size-vertical: 2;
        scrollbar-color: $primary 40%;
        scrollbar-color-hover: $primary 70%;
        scrollbar-color-active: $accent;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-corner-color: $panel;
    }

    .history-section {
        margin: 0 0 1 0;
        background: $surface;
        border: round $secondary;
    }

    .history-section:focus-within {
        background-tint: $foreground 4%;
    }

    .history-section.just-expanded > CollapsibleTitle {
        background: $primary 18%;
        color: $text;
    }

    .history-section.expand-settle > CollapsibleTitle {
        background: $primary 10%;
        color: $text;
    }

    .history-section > CollapsibleTitle {
        text-style: bold;
        padding: 0 1;
    }

    .history-section > CollapsibleTitle:hover {
        background: $primary 12%;
        color: $text;
    }

    .history-section > CollapsibleTitle:focus {
        background: $primary 28%;
        color: $text;
    }

    .assistant-section {
        background: $success 6%;
        border: round $success;
    }

    .assistant-section > CollapsibleTitle {
        color: $success;
    }

    .tool-section {
        background: $warning 6%;
        border: round $warning;
    }

    .tool-section > CollapsibleTitle {
        color: $warning;
    }

    .prompt-section {
        background: $secondary 4%;
        border: round $secondary;
    }

    .prompt-section > CollapsibleTitle {
        color: $secondary;
    }

    .prompt-section .section-body {
        color: $text-muted;
    }

    .tool-call-section {
        background: $accent 8%;
        border: round $accent;
    }

    .tool-call-section > CollapsibleTitle {
        color: $accent;
    }

    .nested-section {
        margin: 0 0 0 1;
    }

    .section-body {
        padding: 0 1 0 1;
        color: $text;
    }

    .details-panel {
        width: 38;
        height: 1fr;
    }

    #details-tabs {
        height: 1fr;
    }

    #details-tabs > ContentTabs {
        background: $panel;
        margin: 0 0 1 0;
    }

    #details-tabs Tab {
        background: $surface;
        color: $text-muted;
        min-width: 8;
    }

    #details-tabs Tab.-active {
        color: $text;
    }

    #details-tabs ContentSwitcher {
        height: 1fr;
    }

    #details-tabs TabPane {
        height: 1fr;
        padding: 0;
    }

    .surface-scroll {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-color: $secondary;
        scrollbar-background: $panel;
        scrollbar-corner-color: $panel;
    }

    #run-browser-details-scroll {
        padding: 0 1 0 2;
        scrollbar-size-vertical: 2;
        scrollbar-gutter: stable;
    }

    #run-browser-details {
        margin-right: 8;
    }

    .browser-columns {
        height: 1fr;
        layout: horizontal;
    }

    .browser-tree-panel {
        width: 48;
        height: 1fr;
        layout: vertical;
    }

    #run-browser-tree {
        height: 1fr;
        background: $surface;
        color: $text;
    }

    #run-browser-tree:focus {
        background-tint: $foreground 4%;
    }

    .browser-details-panel {
        height: 1fr;
        width: 1fr;
    }
    
    Footer {
        background: $panel;
    }
    
    .modal-header {
        height: auto;
    }
    
    .modal-columns {
        height: 1fr;
        layout: horizontal;
    }
    
    .modal-panel {
        width: 50%;
        height: 100%;
        layout: vertical;
    }

    .search-input {
        background: $surface;
        color: $text;
    }

    .copy-targets {
        height: 1fr;
        background: $surface;
        color: $text;
    }

    .copy-textarea {
        height: 1fr;
        background: $surface;
        color: $text;
    }

    """

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index

    def on_mount(self) -> None:
        # Register both custom themes
        self.register_theme(self.BLACK_WARM_THEME)
        self.register_theme(self.WHITE_WARM_THEME)
        # Start with dark theme
        self.theme = "black-warm"
        self.push_screen(BrowseRunsScreen(self.index))

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
        self._lines: Dict[str, List[str]] = {
            "prompt": prompt_lines,
            "completion": completion_lines,
        }
        self._hits: Dict[str, List[SearchHit]] = {
            "prompt": [],
            "completion": [],
        }
        self._cursors: Dict[str, Optional[int]] = {
            "prompt": None,
            "completion": None,
        }
        self._active_column: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text("Search (regex, case-insensitive)", style="bold"))
                yield Input(
                    placeholder="regex...", id="search-input", classes="search-input"
                )
                yield Label("", id="search-error", classes="subtitle")

            with Horizontal(classes="modal-columns"):
                with Panel(classes="modal-panel"):
                    yield Label(Text("Prompt results", style="bold"), id="prompt-count")
                    yield OptionList(id="prompt-results")
                with Panel(classes="modal-panel"):
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
        self._set_active_hit("prompt", event.option_id)

    @on(OptionList.OptionHighlighted, "#completion-results")
    def on_completion_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._set_active_hit("completion", event.option_id)

    @on(OptionList.OptionSelected, "#prompt-results")
    def on_prompt_selected(self, event: OptionList.OptionSelected) -> None:
        self._set_active_hit("prompt", event.option_id, select=True)

    @on(OptionList.OptionSelected, "#completion-results")
    def on_completion_selected(self, event: OptionList.OptionSelected) -> None:
        self._set_active_hit("completion", event.option_id, select=True)

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
        self.dismiss(SearchResult(column=selection.column, pattern=pattern))

    def _set_active_hit(
        self, column: str, option_id: Optional[str], *, select: bool = False
    ) -> None:
        if option_id is None:
            return
        self._active_column = column
        self._cursors[column] = int(option_id)
        self._sync_highlights()
        if select:
            self.action_select()

    def _update_results(self, pattern: str) -> None:
        option_lists = {
            "prompt": self.query_one("#prompt-results", OptionList),
            "completion": self.query_one("#completion-results", OptionList),
        }
        labels = {
            "prompt": self.query_one("#prompt-count", Label),
            "completion": self.query_one("#completion-count", Label),
        }
        error_label = self.query_one("#search-error", Label)

        for column, option_list in option_lists.items():
            option_list.clear_options()
            self._hits[column] = []
            self._cursors[column] = None

        if not pattern:
            error_label.update("")
            labels["prompt"].update(Text("Prompt results", style="bold"))
            labels["completion"].update(Text("Completion results", style="bold"))
            self._active_column = None
            return

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            error_label.update(Text(f"Invalid regex: {exc}", style="red"))
            labels["prompt"].update(Text("Prompt results", style="bold"))
            labels["completion"].update(Text("Completion results", style="bold"))
            self._active_column = None
            return

        error_label.update("")
        for column, lines in self._lines.items():
            hits: List[SearchHit] = []
            for line_index, line in enumerate(lines):
                if not compiled.search(line):
                    continue
                hits.append(
                    SearchHit(column=column, line_index=line_index, line_text=line)
                )
                content = Text(line)
                _stylize_matches(content, compiled, "reverse")
                option_lists[column].add_option(
                    Option(
                        Text(f"{line_index + 1:>5} | ", style="dim") + content,
                        id=str(len(hits) - 1),
                    )
                )
            self._hits[column] = hits
            labels[column].update(
                Text(
                    f"{column.capitalize()} results ({len(hits)})",
                    style="bold",
                )
            )

        if self._hits["completion"]:
            self._active_column = "completion"
            self._cursors["completion"] = 0
        elif self._hits["prompt"]:
            self._active_column = "prompt"
            self._cursors["prompt"] = 0
        else:
            self._active_column = None

        self._sync_highlights()

    def _sync_highlights(self) -> None:
        for column, option_list in (
            ("prompt", self.query_one("#prompt-results", OptionList)),
            ("completion", self.query_one("#completion-results", OptionList)),
        ):
            if self._active_column == column and self._cursors[column] is not None:
                option_list.highlighted = self._cursors[column]
                option_list.scroll_to_highlight()
            else:
                option_list.highlighted = None

    def _switch_column(self, target: str) -> None:
        if target == "prompt" and self._hits["prompt"]:
            self._active_column = "prompt"
            if self._cursors["prompt"] is None:
                self._cursors["prompt"] = 0
        elif target == "completion" and self._hits["completion"]:
            self._active_column = "completion"
            if self._cursors["completion"] is None:
                self._cursors["completion"] = 0
        self._sync_highlights()

    def _move_selection(self, delta: int) -> None:
        if self._active_column is None:
            return
        hits = self._hits[self._active_column]
        cursor = self._cursors[self._active_column]
        if not hits:
            return
        if cursor is None:
            self._cursors[self._active_column] = 0
        else:
            self._cursors[self._active_column] = max(
                0,
                min(len(hits) - 1, cursor + delta),
            )
        self._sync_highlights()

    def _current_selection(self) -> Optional[SearchHit]:
        if self._active_column is None:
            return None
        cursor = self._cursors[self._active_column]
        hits = self._hits[self._active_column]
        if cursor is None or not hits:
            return None
        return hits[cursor]


class RolloutCopyScreen(ModalScreen[None]):
    """Modal screen for copying rollout viewer sections."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("enter", "copy", "Copy"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(
        self,
        items: List[RolloutCopyItem],
        *,
        start_key: Optional[str] = None,
        title: str = "Copy Rollout",
    ):
        super().__init__()
        self._items = items
        self._items_by_key = {item.key: item for item in items}
        self._start_key = (
            start_key
            if start_key in self._items_by_key
            else (items[0].key if items else None)
        )
        self._title = title
        self._current_key = self._start_key
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text(self._title, style="bold"))
                yield Label(
                    Text(
                        "Choose a viewer section on the left. Tab switches panes; enter or c copies the current target.",
                        style="dim",
                    ),
                    id="rollout-copy-hint",
                    classes="copy-hint",
                )
                yield Label("", id="rollout-copy-status", classes="subtitle")

            with Horizontal(classes="modal-columns"):
                with Panel(classes="modal-panel"):
                    yield Label(Text("Copy targets", style="bold"))
                    yield OptionList(id="rollout-copy-targets", classes="copy-targets")
                with Panel(classes="modal-panel"):
                    yield Label(
                        Text("Preview", style="bold"), id="rollout-copy-preview-label"
                    )
                    preview = TextArea(
                        "", id="rollout-copy-preview", classes="copy-textarea"
                    )
                    preview.read_only = True
                    yield preview
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one("#rollout-copy-targets", OptionList)
        for item in self._items:
            option_list.add_option(Option(Text(item.label), id=item.key))

        if self._start_key is not None:
            for idx, item in enumerate(self._items):
                if item.key == self._start_key:
                    option_list.highlighted = idx
                    option_list.scroll_to_highlight()
                    break

        self._sync_preview()
        option_list.focus()

    @on(OptionList.OptionHighlighted, "#rollout-copy-targets")
    def _on_target_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        self._current_key = event.option_id
        self._sync_preview()

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area.id != "rollout-copy-preview":
            return
        selected = event.text_area.selected_text or ""
        if selected and selected != self._last_copied_selection:
            self.app.copy_to_clipboard(selected)
            self._last_copied_selection = selected
            self.query_one("#rollout-copy-status", Label).update(
                Text(f"Copied selection ({len(selected):,} chars).", style="dim")
            )
        self._update_hint()

    def on_key(self, event) -> None:
        if event.key in ("tab", "shift+tab", "backtab"):
            if event.key == "tab":
                self.action_focus_next_pane()
            else:
                self.action_focus_prev_pane()
            event.prevent_default()
            event.stop()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_focus_next_pane(self) -> None:
        self.focus_next()

    def action_focus_prev_pane(self) -> None:
        self.focus_previous()

    def action_copy(self) -> None:
        item = self._current_item()
        if item is None:
            return

        preview = self.query_one("#rollout-copy-preview", TextArea)
        selected = preview.selected_text or ""
        copied_text = selected or item.body
        if not copied_text:
            self.query_one("#rollout-copy-status", Label).update(
                Text("Nothing to copy.", style="dim")
            )
            self._update_hint()
            return

        self.app.copy_to_clipboard(copied_text)
        self._last_copied_selection = copied_text
        if selected:
            message = f"Copied selection ({len(copied_text):,} chars)."
        else:
            message = f"Copied {item.label} ({len(copied_text):,} chars)."
        self.query_one("#rollout-copy-status", Label).update(Text(message, style="dim"))
        self._update_hint()

    def _current_item(self) -> Optional[RolloutCopyItem]:
        if self._current_key is None:
            return self._items[0] if self._items else None
        return self._items_by_key.get(self._current_key)

    def _sync_preview(self) -> None:
        item = self._current_item()
        preview_label = self.query_one("#rollout-copy-preview-label", Label)
        preview = self.query_one("#rollout-copy-preview", TextArea)
        if item is None:
            preview_label.update(Text("Preview", style="bold"))
            preview.load_text("")
            self._update_hint()
            return

        preview_label.update(
            Text(f"{item.label}  ({len(item.body):,} chars)", style="bold")
        )
        preview.load_text(item.body)
        self._update_hint()

    def _update_hint(self) -> None:
        item = self._current_item()
        preview = self.query_one("#rollout-copy-preview", TextArea)
        selected = preview.selected_text or ""
        if selected:
            hint = f"Enter/c copies the selection ({len(selected):,} chars)."
        elif item is not None:
            hint = f"Enter/c copies {item.label.lower()}. Tab switches panes."
        else:
            hint = "Choose a copy target. Tab switches panes."
        self.query_one("#rollout-copy-hint", Label).update(Text(hint, style="dim"))


class CopyScreen(ModalScreen[None]):
    """Modal screen for selecting and copying prompt/completion text."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("tab", "cycle_column", "Next column"),
        Binding("shift+tab", "cycle_column", show=False),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(
        self,
        prompt_text: str,
        completion_text: str,
        start_column: str,
        *,
        prompt_label: str = "Prompt",
        completion_label: str = "Completion",
        title: str = "Copy Mode",
    ):
        super().__init__()
        self._prompt_text = prompt_text
        self._completion_text = completion_text
        self._prompt_label = prompt_label
        self._completion_label = completion_label
        self._title = title
        self._active_column = (
            start_column if start_column in ("prompt", "completion") else "completion"
        )
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text(self._title, style="bold"))
                yield Label(
                    Text(
                        "Highlight text to auto-copy. Tab switches columns, c copies the active column.",
                        style="dim",
                    ),
                    id="copy-hint",
                    classes="copy-hint",
                )
                yield Label("", id="copy-status", classes="subtitle")

            with Horizontal(classes="modal-columns"):
                with Panel(classes="modal-panel"):
                    yield Label(
                        Text(self._prompt_label, style="bold"),
                        id="copy-prompt-label",
                    )
                    prompt_area = TextArea(
                        self._prompt_text,
                        id="copy-prompt",
                        classes="copy-textarea",
                    )
                    prompt_area.read_only = True
                    yield prompt_area
                with Panel(classes="modal-panel"):
                    yield Label(
                        Text(self._completion_label, style="bold"),
                        id="copy-completion-label",
                    )
                    completion_area = TextArea(
                        self._completion_text,
                        id="copy-completion",
                        classes="copy-textarea",
                    )
                    completion_area.read_only = True
                    yield completion_area
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_ui(focus_text_area=True)

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area is self._active_text_area():
            selected = event.text_area.selected_text or ""
            if selected and selected != self._last_copied_selection:
                self.app.copy_to_clipboard(selected)
                self._last_copied_selection = selected
                self.query_one("#copy-status", Label).update(
                    Text(f"Copied selection ({len(selected)} chars).", style="dim")
                )
            self._refresh_ui()

    def action_close(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key in ("tab", "shift+tab", "backtab"):
            self.action_cycle_column()
            event.prevent_default()
            event.stop()

    def action_cycle_column(self) -> None:
        self._active_column = (
            "completion" if self._active_column == "prompt" else "prompt"
        )
        self._refresh_ui(focus_text_area=True)

    def action_copy(self) -> None:
        text_area = self._active_text_area()
        selected = text_area.selected_text or ""
        if not selected:
            selected = text_area.text
            copied_label = "full column"
        else:
            copied_label = "selection"
        if selected:
            self.app.copy_to_clipboard(selected)
            self._last_copied_selection = selected
        self.query_one("#copy-status", Label).update(
            Text(f"Copied {copied_label} ({len(selected)} chars).", style="dim")
            if selected
            else Text()
        )
        self._refresh_ui()

    def _active_text_area(self) -> TextArea:
        if self._active_column == "prompt":
            return self.query_one("#copy-prompt", TextArea)
        return self.query_one("#copy-completion", TextArea)

    def _refresh_ui(self, *, focus_text_area: bool = False) -> None:
        prompt_label = self.query_one("#copy-prompt-label", Label)
        completion_label = self.query_one("#copy-completion-label", Label)
        if self._active_column == "prompt":
            prompt_label.update(Text(f"{self._prompt_label} (active)", style="bold"))
            completion_label.update(Text(self._completion_label, style="bold"))
        else:
            prompt_label.update(Text(self._prompt_label, style="bold"))
            completion_label.update(
                Text(f"{self._completion_label} (active)", style="bold")
            )
        text_area = self._active_text_area()
        if focus_text_area:
            text_area.focus()
        selected = text_area.selected_text or ""
        if selected:
            count = len(selected)
            unit = "char" if count == 1 else "chars"
            copy_text = f"c / ctrl+c: copy selection ({count} {unit})"
        else:
            active_label = (
                self._prompt_label
                if self._active_column == "prompt"
                else self._completion_label
            ).lower()
            copy_text = f"c / ctrl+c: copy {active_label}"
        self.query_one("#copy-hint", Label).update(Text(copy_text, style="dim"))


def main() -> None:
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    VerifiersTUI(discover_results(env_dir, outputs_dir)).run()


if __name__ == "__main__":
    main()
