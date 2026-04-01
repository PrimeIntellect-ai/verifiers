"""
Textual-based TUI for viewing verifiers eval results.
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.dom import DOMNode
from textual.widget import Widget
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.content import Content, Span
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.style import Style
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
from textual.widgets._markdown import (
    Markdown as BaseMarkdown,
    MarkdownBlock,
    MarkdownH1,
    MarkdownH2,
    MarkdownH3,
    MarkdownH4,
    MarkdownH5,
    MarkdownH6,
    MarkdownParagraph,
    MarkdownTD,
    MarkdownTH,
)
from textual.widgets._option_list import Option
from textual.widgets._tabbed_content import ContentTabs
from textual.widgets._tree import TreeNode

from verifiers.utils.display_utils import format_numeric

AnimationLevel = Literal["none", "basic", "full"]
TreeBinding = Binding | tuple[str, str] | tuple[str, str, str]


def _binding_key(binding: TreeBinding) -> str:
    if isinstance(binding, Binding):
        return binding.key
    return binding[0]


def _int_like_sort_key(value: Any) -> Tuple[int, int, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except (TypeError, ValueError):
        return (1, 0, text)


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
    tree_name: str = ""
    tree_suffix: Tuple[Tuple[str, str], ...] = ()


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
        Binding("left", "cursor_parent", "Parent folder", show=True),
        Binding("right", "cursor_right", "Expand/next folder", show=True),
        Binding("enter", "select_cursor", "Open/toggle", show=True),
        Binding("space", "toggle_node", "Toggle folder", show=True),
    ]

    def _visible_depth(self, node: Any) -> int:
        depth = 0
        parent = node.parent
        while parent is not None and (self.show_root or not parent.is_root):
            depth += 1
            parent = parent.parent
        return depth

    def _render_browser_label(
        self, payload: BrowserNodeData, style: Style, max_width: int
    ) -> Text:
        label = Text()
        label.append(payload.tree_name or "", style="bold")
        for text, segment_style in payload.tree_suffix:
            label.append(text, style=segment_style or None)

        if max_width <= 0:
            label.truncate(1, overflow="ellipsis")
            label.stylize(cast(Any, style))
            return label

        suffix = Text()
        for text, segment_style in payload.tree_suffix:
            suffix.append(text, style=segment_style or None)

        if suffix.cell_len < max_width:
            name = Text(payload.tree_name or "", style="bold")
            name.truncate(max_width - suffix.cell_len, overflow="ellipsis")
            label = Text.assemble(name, suffix)
        else:
            label.truncate(max_width, overflow="ellipsis")

        label.stylize(cast(Any, style))
        return label

    def render_label(  # ty: ignore[invalid-method-override]
        self,
        node: TreeNode[Any],
        base_style: Style,
        style: Style,
    ) -> Text:
        payload = node.data
        available_width = self.size.width - (
            self._visible_depth(node) * self.guide_depth
        )
        prefix_text = (
            self.ICON_NODE_EXPANDED
            if node.allow_expand and node.is_expanded
            else self.ICON_NODE
            if node.allow_expand
            else ""
        )
        content_width = max(1, available_width - len(prefix_text))

        if isinstance(payload, BrowserNodeData) and payload.tree_name:
            label = self._render_browser_label(payload, style, content_width)
        else:
            label = node._label.copy()
            label.stylize(cast(Any, style))
            label.truncate(content_width, overflow="ellipsis")

        return Text.assemble((prefix_text, cast(Any, base_style)), label)

    def action_cursor_parent(self) -> None:
        """Move the cursor to the nearest visible parent folder."""
        cursor_node = self.cursor_node
        if cursor_node is None:
            return
        parent = cursor_node.parent
        if parent is None or (not self.show_root and parent.parent is None):
            return
        self.move_cursor(parent, animate=True)

    def action_cursor_right(self) -> None:
        """Expand the current folder or move to the next visible parent folder."""
        cursor_node = self.cursor_node
        if cursor_node is None:
            return
        if cursor_node.allow_expand:
            if cursor_node.is_collapsed:
                cursor_node.expand()
                return
            if cursor_node.children:
                self.move_cursor(cursor_node.children[0], animate=True)
                return

        node = cursor_node.parent if not cursor_node.allow_expand else cursor_node
        while node is not None:
            next_sibling = node.next_sibling
            if next_sibling is not None:
                self.move_cursor(next_sibling, animate=True)
                return
            node = node.parent
            if node is not None and not self.show_root and node.is_root:
                return

    def action_toggle_node(self) -> None:
        """Toggle the current folder, or the nearest ancestor folder for a leaf."""
        node = self.cursor_node
        while node is not None and not node.allow_expand:
            node = node.parent
        if node is None or (not self.show_root and node.parent is None):
            return
        if node is not self.cursor_node:
            self.move_cursor(node, animate=False)
        self._toggle_node(node)


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


class LazyLogFile:
    """Lazy loader for log files with line-level random access."""

    MAX_DISPLAY_LINES = 10_000

    def __init__(self, path: Path):
        self._path = path
        self._fh = path.open("r", encoding="utf-8", errors="replace")
        self._offsets: List[int] = []
        self._cache: Dict[int, str] = {}
        self._eof = False
        self._count: Optional[int] = None

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
            if self._read_next_line() is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        if self._count is not None:
            return self._count
        while not self._eof:
            if self._read_next_line() is None:
                break
        self._count = len(self._offsets)
        return self._count

    def get_line(self, index: int) -> str:
        if index in self._cache:
            return self._cache[index]
        if not self._ensure_index(index):
            return ""
        pos = self._fh.tell()
        try:
            self._fh.seek(self._offsets[index])
            line = self._fh.readline().rstrip("\n\r")
        finally:
            self._fh.seek(pos)
        self._cache[index] = line
        return line

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        return self._read_next_line() is not None


# ----------------------------
# Log styling helpers
# ----------------------------

_LOG_LEVEL_STYLES: Dict[str, str] = {
    "DEBUG": "dim blue",
    "INFO": "bold green",
    "WARNING": "bold yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold red reverse",
}


def _parse_log_header(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse a log line into (timestamp, source, level, message).

    Expected format: '2026-03-03 22:57:21 - source.name - LEVEL ...'
    """
    if len(line) < 22 or line[19:22] != " - ":
        return None
    rest = line[22:]
    sep_idx = rest.find(" - ")
    if sep_idx < 0:
        return None
    source = rest[:sep_idx]
    after_source = rest[sep_idx + 3 :]
    space_idx = after_source.find(" ")
    if space_idx < 0:
        level = after_source
        message = ""
    else:
        level = after_source[:space_idx]
        message = after_source[space_idx:]
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return None
    return line[:19], source, level, message


def _append_styled_log_line(log_text: Text, line: str) -> None:
    """Append a log line to a Text object with colored header parts."""
    parsed = _parse_log_header(line)
    if parsed is None:
        log_text.append(line, style="dim")
        return
    timestamp, source, level, message = parsed
    level_style = _LOG_LEVEL_STYLES.get(level, "dim")
    log_text.append(timestamp, style="bold dim")
    log_text.append(" - ", style="dim")
    log_text.append(source, style="dim cyan")
    log_text.append(" - ", style="dim")
    log_text.append(level, style=level_style)
    log_text.append(message, style="dim")


def _log_tab_label(path: Path) -> str:
    """Derive a display label from a log file path."""
    stem = path.stem
    if stem.startswith("env_"):
        stem = stem[4:]
    return stem


def _discover_log_files(run_path: Path) -> List[Path]:
    """Find log files in a run directory, sorted with env_server first."""
    log_files = sorted(run_path.glob("*.log"))
    # Put env_server.log first, then workers in natural order
    server_logs = [p for p in log_files if p.name == "env_server.log"]
    worker_logs = sorted(
        [p for p in log_files if p.name.startswith("env_worker_")],
        key=lambda p: (
            int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
        ),
    )
    other_logs = [p for p in log_files if p not in server_logs and p not in worker_logs]
    return server_logs + worker_logs + other_logs


def _merge_log_files(log_files: List[Path]) -> List[str]:
    """Merge lines from multiple log files, sorted by timestamp.

    Lines without a parseable timestamp are attached to the preceding
    timestamped line (continuation lines from multi-line log messages).
    """
    # Collect all lines with their timestamps
    entries: List[Tuple[str, int, str]] = []  # (timestamp, file_idx, line)
    for file_idx, path in enumerate(log_files):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        current_ts = ""
        for line in lines:
            parsed = _parse_log_header(line)
            if parsed is not None:
                current_ts = parsed[0]  # timestamp string
            entries.append((current_ts, file_idx, line))
    # Stable sort by timestamp — preserves original order for same-timestamp lines
    entries.sort(key=lambda e: e[0])
    return [line for _, _, line in entries]


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
                item_type = item.get("type")
                if item_type == "text":
                    chunks.append(str(item.get("text", "")))
                elif item_type in {"input_audio", "audio"}:
                    chunks.append("[audio]")
                elif item_type in {"image", "image_url"}:
                    chunks.append("[image]")
                elif item_type in {"thinking", "redacted_thinking"}:
                    continue
                else:
                    chunks.append(_pretty_json_or_str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        return _pretty_json_or_str(content)
    return str(content)


def _thinking_block_to_text(block: Any) -> str:
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking")
            return str(thinking).strip() if thinking else ""
        if block_type == "redacted_thinking":
            return "[reasoning redacted]"
        return ""

    block_type = getattr(block, "type", None)
    if block_type == "thinking":
        thinking = getattr(block, "thinking", None)
        return str(thinking).strip() if thinking else ""
    if block_type == "redacted_thinking":
        return "[reasoning redacted]"
    return ""


def _stringify_message_reasoning(message: Any) -> str:
    if not isinstance(message, dict):
        return ""

    parts: List[str] = []

    def add_part(value: str) -> None:
        text = value.strip()
        if text and text not in parts:
            parts.append(text)

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        add_part(reasoning_content)

    thinking_blocks = message.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        for block in thinking_blocks:
            add_part(_thinking_block_to_text(block))

    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            add_part(_thinking_block_to_text(item))

    return "\n\n".join(parts)


def _stringify_message(message: Any) -> str:
    if not isinstance(message, dict):
        return _stringify_message_content(message)

    content = _stringify_message_content(message.get("content", "")).strip()
    reasoning = _stringify_message_reasoning(message)
    if reasoning and content:
        return f"Reasoning\n{reasoning}\n\n{content}"
    if reasoning:
        return f"Reasoning\n{reasoning}"
    return content


def _parse_tool_calls(tool_calls: Any) -> List[Any]:
    if not isinstance(tool_calls, list):
        return []
    return [_parse_jsonish_string(tool_call) for tool_call in tool_calls]


def _truncate_preview(text: str, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _compute_prompt_hash(prompt: list | None) -> str | None:
    """MD5 hash of JSON-serialized prompt for deduplication."""
    if prompt is None:
        return None
    return hashlib.md5(
        json.dumps(prompt, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


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
    reasoning = _stringify_message_reasoning(message)
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if content:
        return _truncate_preview(content, 56)
    if reasoning:
        return f"reasoning: {_truncate_preview(reasoning, 45)}"
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


def _compact_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _format_setting_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_compact_metric(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(
            isinstance(item, (str, int, float, bool)) and not isinstance(item, dict)
            for item in value
        ):
            return ", ".join(_format_setting_value(item) for item in value)
    return _compact_json_or_str(value)


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return str(getattr(tool, "name", "") or "")
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _run_setting_rows(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []

    ordered_settings: List[Tuple[str, Any]] = []
    if meta.get("base_url") not in (None, ""):
        ordered_settings.append(("endpoint", meta["base_url"]))
    if meta.get("num_examples") not in (None, ""):
        ordered_settings.append(("examples", meta["num_examples"]))
    if meta.get("rollouts_per_example") not in (None, ""):
        ordered_settings.append(("rollouts/example", meta["rollouts_per_example"]))
    if meta.get("pass_threshold") not in (None, ""):
        ordered_settings.append(("pass threshold", meta["pass_threshold"]))

    sampling_args = meta.get("sampling_args")
    if isinstance(sampling_args, dict):
        for key in sorted(sampling_args):
            value = sampling_args[key]
            if value not in (None, ""):
                ordered_settings.append((f"sampling.{key}", value))

    env_args = meta.get("env_args")
    if isinstance(env_args, dict):
        for key in sorted(env_args):
            value = env_args[key]
            if value not in (None, ""):
                ordered_settings.append((f"env.{key}", value))

    state_columns = meta.get("state_columns")
    if isinstance(state_columns, list) and state_columns:
        ordered_settings.append(("state columns", state_columns))

    tools = meta.get("tools")
    if isinstance(tools, list):
        tool_names = sorted(
            name for name in (_tool_name(tool) for tool in tools) if name
        )
        if tool_names:
            ordered_settings.append(("tools", tool_names))

    for label, value in ordered_settings:
        rows.append((label, _format_setting_value(value)))

    return rows


def _build_settings_table(
    rows: List[Tuple[str, str]],
    heading: str,
    *,
    value_header: str = "Value",
) -> Group | Text:
    if not rows:
        return Text()

    title = Text()
    title.append(heading, style="bold dim")

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column("Setting", style="dim", width=20, no_wrap=True)
    table.add_column(value_header, ratio=1)

    for setting, value in rows:
        table.add_row(setting, value)

    return Group(title, table)


def _run_setting_variation_rows(
    runs: List[RunInfo], *, max_rows: int = 8
) -> Tuple[List[Tuple[str, str]], int]:
    if not runs:
        return [], 0

    setting_maps = [dict(_run_setting_rows(run.load_metadata())) for run in runs]

    ordered_keys: List[str] = []
    for setting_map in setting_maps:
        for key in setting_map:
            if key not in ordered_keys:
                ordered_keys.append(key)

    rows: List[Tuple[str, str]] = []
    for key in ordered_keys:
        counts: Dict[str, int] = defaultdict(int)
        for setting_map in setting_maps:
            counts[setting_map.get(key, "(unset)")] += 1
        if len(counts) <= 1:
            continue
        parts = [
            f"{value} ({count} run{'s' if count != 1 else ''})"
            for value, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        rows.append((key, ", ".join(parts)))

    hidden_rows = max(0, len(rows) - max_rows)
    return rows[:max_rows], hidden_rows


def _varying_run_setting_keys(
    runs: List[RunInfo],
) -> Tuple[List[str], List[Tuple[RunInfo, Dict[str, str]]]]:
    if not runs:
        return [], []

    run_settings = [(run, dict(_run_setting_rows(run.load_metadata()))) for run in runs]
    ordered_keys: List[str] = []
    for _, settings in run_settings:
        for key in settings:
            if key not in ordered_keys:
                ordered_keys.append(key)

    if len(run_settings) == 1:
        return ordered_keys, run_settings

    varying_keys = [
        key
        for key in ordered_keys
        if len({settings.get(key, "(unset)") for _, settings in run_settings}) > 1
    ]
    return varying_keys, run_settings


def _reward_bucket_counts(values: List[float]) -> List[Tuple[str, int, str]]:
    bucket_counts = [
        ("<0", 0, "bold red"),
        ("=0", 0, "bold red"),
        ("0-<0.25", 0, "red"),
        ("0.25-<0.5", 0, "yellow"),
        ("0.5-<0.75", 0, "yellow"),
        ("0.75-<1", 0, "green"),
        ("=1", 0, "bold green"),
        (">1", 0, "bold green"),
    ]

    for reward in values:
        if reward < 0:
            bucket_idx = 0
        elif reward == 0:
            bucket_idx = 1
        elif reward < 0.25:
            bucket_idx = 2
        elif reward < 0.5:
            bucket_idx = 3
        elif reward < 0.75:
            bucket_idx = 4
        elif reward < 1.0:
            bucket_idx = 5
        elif reward == 1.0:
            bucket_idx = 6
        else:
            bucket_idx = 7
        label, count, style = bucket_counts[bucket_idx]
        bucket_counts[bucket_idx] = (label, count + 1, style)

    # Only include <0 and >1 buckets if they have values.
    return [
        (label, count, style)
        for label, count, style in bucket_counts
        if not (label in ("<0", ">1") and count == 0)
    ]


_COMPARE_ALIAS_PALETTE: Tuple[str, ...] = (
    "#61afef",
    "#98c379",
    "#e5c07b",
    "#c678dd",
    "#56b6c2",
    "#e06c75",
)


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
    content = _stringify_message(message)
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
        reasoning = _stringify_message_reasoning(value)
        if reasoning:
            return _truncate_preview(reasoning, limit)
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

    bucket_counts = _reward_bucket_counts(values)
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
    prepared: List[Tuple[int, str, str, MetricSummary]] = []
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

        display_name = summary.name.replace("_", " ")
        prepared.append(
            (
                category_order.get(category, 99),
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
    for _, display_name, category, summary in sorted(prepared):
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


class TabbedScrollPane(VerticalScroll):
    """A VerticalScroll that switches sibling tabs with left/right arrows."""

    BINDINGS = [
        Binding("left", "prev_tab", "Prev tab", show=False),
        Binding("right", "next_tab", "Next tab", show=False),
    ]

    def _get_tabbed_content(self) -> TabbedContent | None:
        node = self.parent
        while node is not None:
            if isinstance(node, TabbedContent):
                return node
            node = node.parent
        return None

    def action_prev_tab(self) -> None:
        tc = self._get_tabbed_content()
        if tc is not None:
            tc.query_one(ContentTabs).action_previous_tab()

    def action_next_tab(self) -> None:
        tc = self._get_tabbed_content()
        if tc is not None:
            tc.query_one(ContentTabs).action_next_tab()


class LogScrollPane(VerticalScroll):
    """A VerticalScroll that switches log file tabs with left/right arrows."""

    BINDINGS = [
        Binding("left", "prev_log_tab", "Prev log", show=False),
        Binding("right", "next_log_tab", "Next log", show=False),
    ]

    def _get_view_run_screen(self) -> Optional["ViewRunScreen"]:
        node = self.parent
        while node is not None:
            if isinstance(node, Screen):
                return node if isinstance(node, ViewRunScreen) else None
            node = node.parent
        return None

    def action_prev_log_tab(self) -> None:
        screen = self._get_view_run_screen()
        if screen is not None:
            screen._cycle_log_tab(-1)

    def action_next_log_tab(self) -> None:
        screen = self._get_view_run_screen()
        if screen is not None:
            screen._cycle_log_tab(1)


# ----------------------------
# Search helpers
# ----------------------------
@dataclass(frozen=True)
class SearchHit:
    column: str
    line_index: int
    line_text: str
    section_index: int = 0
    nested_index: int = -1  # -1 = parent body, 0+ = nested section index


@dataclass(frozen=True)
class SearchResult:
    column: str
    pattern: str
    section_index: int = 0
    nested_index: int = -1


@dataclass(frozen=True)
class HistorySectionData:
    title: str
    body: str
    column: str
    collapsed: bool
    classes: str
    nested_sections: Tuple["HistorySectionData", ...] = ()
    body_first: bool = True


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
# Markdown rendering
# ----------------------------
_LATEX_BEGIN_END_RE = re.compile(r"\\(?:begin|end)\{[^}]+\}")
_LATEX_BRACED_SCRIPT_RE = re.compile(r"([_^])\{([^{}]+)\}")
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|operatorname|text)\{([^{}]+)\}"
)
_LATEX_FRACTION_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+|.)")
_LATEX_COMMAND_REPLACEMENTS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "sigma": "σ",
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "cdot": "·",
    "times": "×",
    "pm": "±",
    "neq": "!=",
    "leq": "<=",
    "geq": ">=",
    "approx": "~",
    "to": "->",
    "rightarrow": "->",
    "leftarrow": "<-",
    "infty": "∞",
    "ldots": "...",
    "cdots": "...",
    "sum": "sum",
    "prod": "prod",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "|": "||",
    ",": " ",
    ";": " ",
    "!": "",
}


def _replace_latex_groups(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Callable[[re.Match[str]], str],
) -> str:
    while True:
        updated = pattern.sub(replacement, text)
        if updated == text:
            return updated
        text = updated


def _replace_latex_fraction(match: re.Match[str]) -> str:
    numerator, denominator = (part.strip() for part in match.groups())
    if re.search(r"\s|[+\-*/]", numerator):
        numerator = f"({numerator})"
    if re.search(r"\s|[+\-*/]", denominator):
        denominator = f"({denominator})"
    return f"{numerator}/{denominator}"


def _replace_latex_command(match: re.Match[str]) -> str:
    command = match.group(1)
    if command in _LATEX_COMMAND_REPLACEMENTS:
        return _LATEX_COMMAND_REPLACEMENTS[command]
    if len(command) == 1 and not command.isalpha():
        return command
    return command


def _fallback_latex_to_text(latex: str, *, preserve_newlines: bool) -> str:
    text = _LATEX_BEGIN_END_RE.sub("", latex)
    text = text.replace("&", " ")
    text = text.replace("\\\\", "\n" if preserve_newlines else " ")
    text = _replace_latex_groups(text, _LATEX_WRAPPER_RE, r"\1")
    text = _replace_latex_groups(text, _LATEX_BRACED_SCRIPT_RE, r"\1\2")
    text = _replace_latex_groups(text, _LATEX_FRACTION_RE, _replace_latex_fraction)
    text = _replace_latex_groups(text, _LATEX_SQRT_RE, r"sqrt(\1)")
    text = _LATEX_COMMAND_RE.sub(_replace_latex_command, text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    if preserve_newlines:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    return " ".join(text.split())


def _latex_to_text(latex: str, *, preserve_newlines: bool) -> str:
    return _fallback_latex_to_text(latex, preserve_newlines=preserve_newlines)


def render_inline_math(latex: str) -> str:
    return " ".join(_latex_to_text(latex, preserve_newlines=False).split())


def render_block_math(latex: str) -> str:
    return _latex_to_text(latex, preserve_newlines=True).strip()


def make_math_parser() -> MarkdownIt:
    parser = MarkdownIt("gfm-like")
    parser.use(
        dollarmath_plugin,
        allow_space=False,
        allow_digits=False,
    )
    parser.use(amsmath_plugin)
    return parser


class MathInlineMixin:
    """Teach Textual's Markdown blocks how to render inline math tokens."""

    def _token_to_content(self, token: Any) -> Content:
        if token.children is None:
            return Content("")

        parts: List[str] = []
        spans: List[Span] = []
        style_stack: List[Tuple[Style | str, int]] = []
        position = 0

        def add_text(text: str) -> None:
            nonlocal position
            parts.append(text)
            position += len(text)

        def push_style(style: Style | str) -> None:
            style_stack.append((style, position))

        def pop_style() -> None:
            if not style_stack:
                return
            style, start = style_stack.pop()
            spans.append(Span(start, position, style))

        for child in token.children:
            child_type = child.type
            attrs = child.attrs or {}

            if child_type == "text":
                add_text(re.sub(r"\s+", " ", child.content))
            elif child_type == "hardbreak":
                add_text("\n")
            elif child_type == "softbreak":
                add_text(" ")
            elif child_type == "code_inline":
                push_style(".code_inline")
                add_text(child.content)
                pop_style()
            elif child_type in {"math_inline", "math_inline_double"}:
                push_style("italic")
                add_text(render_inline_math(child.content))
                pop_style()
            elif child_type == "em_open":
                push_style(".em")
            elif child_type == "strong_open":
                push_style(".strong")
            elif child_type == "s_open":
                push_style(".s")
            elif child_type == "link_open":
                href = attrs.get("href", "")
                action = f"link({href!r})"
                push_style(Style.from_meta({"@click": action}))
            elif child_type == "image":
                href = attrs.get("src", "")
                alt = attrs.get("alt", "")
                action = f"link({href!r})"
                push_style(Style.from_meta({"@click": action}))
                add_text(" ")
                if alt:
                    add_text(f"({alt})")
                if child.children is not None:
                    for grandchild in child.children:
                        add_text(grandchild.content)
                pop_style()
            elif child_type.endswith("_close"):
                pop_style()

        return Content("".join(parts), spans=spans)


class MathParagraph(MathInlineMixin, MarkdownParagraph):
    pass


class MathH1(MathInlineMixin, MarkdownH1):
    pass


class MathH2(MathInlineMixin, MarkdownH2):
    pass


class MathH3(MathInlineMixin, MarkdownH3):
    pass


class MathH4(MathInlineMixin, MarkdownH4):
    pass


class MathH5(MathInlineMixin, MarkdownH5):
    pass


class MathH6(MathInlineMixin, MarkdownH6):
    pass


class MathTH(MathInlineMixin, MarkdownTH):
    pass


class MathTD(MathInlineMixin, MarkdownTD):
    pass


class MathDisplayBlock(MarkdownBlock):
    DEFAULT_CSS = """
    MathDisplayBlock {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: $boost;
        border-left: outer $primary 60%;
    }
    """

    def __init__(self, markdown: "MathMarkdown", token: Any):
        super().__init__(markdown, token)
        text = render_block_math(token.content)
        if token.type == "math_block_label" and getattr(token, "info", ""):
            text = f"[{token.info}]\n{text}"
        self.set_content(Content(text))


class MathMarkdown(BaseMarkdown):
    BLOCKS = BaseMarkdown.BLOCKS | {
        "paragraph_open": MathParagraph,
        "h1": MathH1,
        "h2": MathH2,
        "h3": MathH3,
        "h4": MathH4,
        "h5": MathH5,
        "h6": MathH6,
        "th_open": MathTH,
        "td_open": MathTD,
    }

    def __init__(self, markdown: str | None = None, **kwargs: Any) -> None:
        super().__init__(markdown, parser_factory=make_math_parser, **kwargs)

    def unhandled_token(self, token: Any) -> MarkdownBlock | None:
        if token.type in {"math_block", "math_block_label", "amsmath"}:
            return MathDisplayBlock(self, token)
        return None


# ----------------------------
# Screens
# ----------------------------
class CompareRunsScreen(Screen):
    """Dedicated comparison view for runs, optionally across models."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b,backspace", "back", "Back"),
        Binding("g", "enter_group_mode", "Group by"),
        Binding("left", "group_cursor_left", show=False),
        Binding("right", "group_cursor_right", show=False),
        Binding("enter", "group_select", show=False),
        Binding("escape", "exit_group_mode", show=False),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, env_id: str, model: Optional[str], runs: List[RunInfo]):
        super().__init__()
        self.env_id = env_id
        self.model = model
        self.runs = list(runs)
        self._stats_by_path: Dict[Path, RunOverviewStats] = {}
        self._setting_keys: List[str] = []
        self._run_settings: List[Tuple[RunInfo, Dict[str, str]]] = []
        self._group_mode: bool = False
        self._group_cursor: int = 0
        self._grouped_by_key: str | None = None
        self._distinct_prompts_by_group: Dict[Tuple[str, ...], int] = {}
        self._prompt_count_cache: Dict[str, int] = {}  # run-ID-set hash → count

    def compose(self) -> ComposeResult:
        with Container():
            yield Panel(
                Label(Text("Run Comparison", style="bold"), classes="title"),
                Static("", id="compare-subtitle", classes="subtitle", markup=False),
                VerticalScroll(
                    Static("", id="compare-header", markup=False),
                    Static("", id="compare-outcomes", markup=False),
                    id="compare-scroll",
                    classes="surface-scroll",
                ),
                classes="compare-panel",
            )
        yield Footer()

    def on_mount(self) -> None:
        subtitle = Text()
        subtitle.append(self.model or "all models", style="bold")
        subtitle.append("\n")
        subtitle.append(f"{self.env_id}   {len(self.runs)} runs", style="dim")
        self.query_one("#compare-subtitle", Static).update(subtitle)
        self.query_one("#compare-header", Static).update(
            Text("Loading comparison…", style="dim")
        )
        self._load_comparison_stats()

    def action_back(self) -> None:
        if self._grouped_by_key is not None:
            self._grouped_by_key = None
            self._distinct_prompts_by_group = {}
            self._refresh_outcomes()
            self._load_distinct_prompt_counts()
            return
        if self._group_mode:
            self._group_mode = False
            self._refresh_outcomes()
            return
        self.app.pop_screen()

    @staticmethod
    def _renderable_to_text(renderable: Any, width: int = 220) -> str:
        buf = StringIO()
        Console(file=buf, force_terminal=False, color_system=None, width=width).print(
            renderable
        )
        return buf.getvalue().rstrip()

    def action_copy(self) -> None:
        if not self._stats_by_path:
            return
        outcomes_table, axis_legend, value_legend = self._build_grouped_outcomes_table(
            self._stats_by_path,
            self._setting_keys,
            self._run_settings,
            group_by_key=self._grouped_by_key,
        )
        parts: List[str] = [
            self._renderable_to_text(outcomes_table),
        ]
        legend = self._build_argument_legend(axis_legend, value_legend)
        if isinstance(legend, Group) or (isinstance(legend, Text) and legend.plain):
            parts.append(self._renderable_to_text(legend))
        all_body = "\n\n".join(parts)
        self.app.push_screen(
            CompactCopyScreen(
                [RolloutCopyItem(key="all", label="Comparison", body=all_body)],
                start_key="all",
                title="Copy Comparison",
            )
        )

    @work(
        thread=True,
        group="run-comparison",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_comparison_stats(self) -> None:
        stats_by_path = {
            run.path: _compute_run_overview_stats(run) for run in self.runs
        }
        self.app.call_from_thread(self._finish_loading_comparison_stats, stats_by_path)

    def _finish_loading_comparison_stats(
        self, stats_by_path: Dict[Path, RunOverviewStats]
    ) -> None:
        if not self.is_mounted:
            return
        self._stats_by_path = stats_by_path
        self._setting_keys, self._run_settings = _varying_run_setting_keys(self.runs)
        if self.model is None:
            # Cross-model comparison: inject "model" as a setting axis
            for run, settings in self._run_settings:
                settings["model"] = run.model
            if "model" not in self._setting_keys:
                self._setting_keys.insert(0, "model")
        self.query_one("#compare-header", Static).update(Text(""))
        self._refresh_outcomes()
        self._load_distinct_prompt_counts()

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_outcomes()

    def _refresh_outcomes(self) -> None:
        if not self._stats_by_path:
            return
        self.query_one("#compare-outcomes", Static).update(
            self._build_comparison_outcomes()
        )

    @work(
        thread=True,
        group="prompt-counting",
        exclusive=True,
        exit_on_error=False,
    )
    def _load_distinct_prompt_counts(self) -> None:
        """Compute distinct prompt counts per group in a background thread."""

        group_keys = (
            [self._grouped_by_key] if self._grouped_by_key else self._setting_keys
        )

        # Build group membership
        groups: Dict[Tuple[str, ...], List[Tuple[RunInfo, Dict[str, str]]]] = (
            defaultdict(list)
        )
        for run, settings in self._run_settings:
            gk = tuple(settings.get(key, "(unset)") for key in group_keys)
            groups[gk].append((run, settings))

        # Process one group at a time; update table after each group completes
        for group_key_val, group_runs in groups.items():
            if not self.is_mounted:
                return

            # Cache key: hash of sorted run IDs in this group
            run_ids = sorted(run.run_id for run, _ in group_runs)
            cache_key = hashlib.md5(str(run_ids).encode()).hexdigest()

            cached = self._prompt_count_cache.get(cache_key)
            if cached is not None:
                self.app.call_from_thread(
                    self._update_group_prompt_count, group_key_val, cached
                )
                continue

            # Not cached — compute by streaming results.jsonl
            hashes: set[str] = set()
            for run, _ in group_runs:
                if not self.is_mounted:
                    return
                try:
                    with (run.path / "results.jsonl").open("r", encoding="utf-8") as f:
                        for line in f:
                            if not self.is_mounted:
                                return
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(record, dict):
                                continue
                            prompt = record.get("prompt")
                            ph = _compute_prompt_hash(prompt)
                            if ph is not None:
                                hashes.add(ph)
                except OSError:
                    pass

            count = len(hashes)
            self._prompt_count_cache[cache_key] = count
            self.app.call_from_thread(
                self._update_group_prompt_count, group_key_val, count
            )

    def _update_group_prompt_count(
        self, group_key: Tuple[str, ...], count: int
    ) -> None:
        if not self.is_mounted:
            return
        self._distinct_prompts_by_group[group_key] = count
        self._refresh_outcomes()

    def action_enter_group_mode(self) -> None:
        if not self._setting_keys:
            return
        self._group_mode = True
        self._group_cursor = 0
        self._grouped_by_key = None
        self._refresh_outcomes()

    def action_group_cursor_left(self) -> None:
        if not self._group_mode or not self._setting_keys:
            return
        self._group_cursor = (self._group_cursor - 1) % len(self._setting_keys)
        self._refresh_outcomes()

    def action_group_cursor_right(self) -> None:
        if not self._group_mode or not self._setting_keys:
            return
        self._group_cursor = (self._group_cursor + 1) % len(self._setting_keys)
        self._refresh_outcomes()

    def action_group_select(self) -> None:
        if not self._group_mode or not self._setting_keys:
            return
        self._grouped_by_key = self._setting_keys[self._group_cursor]
        self._distinct_prompts_by_group = {}
        self._refresh_outcomes()
        self._load_distinct_prompt_counts()

    def action_exit_group_mode(self) -> None:
        if self._group_mode:
            self._group_mode = False
            self._grouped_by_key = None
            self._distinct_prompts_by_group = {}
            self._refresh_outcomes()
            self._load_distinct_prompt_counts()

    def _short_setting_key(self, key: str) -> str:
        replacements = {
            "rollouts/example": "r/ex",
            "sampling.": "",
            "env.": "",
        }
        short = key
        for source, target in replacements.items():
            short = short.replace(source, target)
        return short

    def _alias_style(self, label: str) -> str:
        match = re.fullmatch(r"v(\d+)(?:\.\d+)?", label)
        if match is None:
            return ""
        alias_idx = int(match.group(1)) - 1
        return f"bold {_COMPARE_ALIAS_PALETTE[alias_idx % len(_COMPARE_ALIAS_PALETTE)]}"

    def _share_style(self, share: float, positive: bool) -> str:
        if share <= 0:
            return "dim"
        if positive:
            return "bold green" if share >= 0.5 else "green"
        return "bold red" if share >= 0.5 else "red"

    def _build_reward_mix_bar(self, values: List[float], width: int = 18) -> Text:
        if not values:
            return Text("—", style="dim")

        counts = _reward_bucket_counts(values)
        total = len(values)
        raw_widths = [
            (count / total) * width if total else 0.0 for _, count, _ in counts
        ]
        segment_widths = [int(raw) for raw in raw_widths]
        used = sum(segment_widths)

        remainders = sorted(
            [
                (raw - int(raw), idx)
                for idx, ((_, count, _), raw) in enumerate(zip(counts, raw_widths))
                if count > 0
            ],
            reverse=True,
        )
        for _, idx in remainders:
            if used >= width:
                break
            segment_widths[idx] += 1
            used += 1

        out = Text()
        for (_, count, style), segment_width in zip(counts, segment_widths):
            if count <= 0 or segment_width <= 0:
                continue
            out.append("█" * segment_width, style=style)
        if used < width:
            out.append("░" * (width - used), style="dim")
        return out

    def _build_grouped_outcomes_table(
        self,
        stats_by_path: Dict[Path, RunOverviewStats],
        setting_keys: List[str],
        run_settings: List[Tuple[RunInfo, Dict[str, str]]],
        group_by_key: str | None = None,
        highlight_col: int | None = None,
    ) -> Tuple[Table, List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
        # Determine which keys to actually group by.
        group_keys = [group_by_key] if group_by_key else setting_keys

        grouped: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for run, settings in run_settings:
            group_key_val = tuple(settings.get(key, "(unset)") for key in group_keys)
            group = grouped.setdefault(
                group_key_val,
                {
                    "runs": [],
                    "rewards": [],
                    "avg_rewards": [],
                    "run_settings": [],
                },
            )
            cast(List[RunInfo], group["runs"]).append(run)
            group["run_settings"].append(settings)
            stats = stats_by_path.get(run.path, RunOverviewStats([], []))
            if stats.rewards:
                cast(List[float], group["rewards"]).extend(stats.rewards)
            avg_reward = _numeric_reward(run.load_metadata().get("avg_reward"))
            if avg_reward is not None:
                cast(List[float], group["avg_rewards"]).append(avg_reward)

        rows = list(grouped.items())
        rows.sort(
            key=lambda item: (
                -(
                    sum(cast(List[float], item[1]["rewards"]))
                    / len(cast(List[float], item[1]["rewards"]))
                    if cast(List[float], item[1]["rewards"])
                    else (
                        sum(cast(List[float], item[1]["avg_rewards"]))
                        / len(cast(List[float], item[1]["avg_rewards"]))
                        if cast(List[float], item[1]["avg_rewards"])
                        else float("-inf")
                    )
                ),
                -len(cast(List[RunInfo], item[1]["runs"])),
            )
        )

        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            show_edge=False,
            pad_edge=False,
            padding=(0, 1),
            collapse_padding=True,
            row_styles=["none", "dim"],
        )
        # --- Responsive aliasing and column hiding ---
        #
        # 1. Compute per-column budget assuming all optional columns are shown.
        #    Alias any header or value that exceeds that budget.
        # 2. Drop optional columns until setting columns (with min_width) fit.
        terminal_width = self.size.width if self.is_mounted else 120
        n = len(setting_keys)
        optional_cols = [
            ("=0", 7),
            ("=1", 7),
            ("mix", 22),
            ("rollouts", 11),
            ("unique prompts", 10),
            ("runs", 7),
        ]  # name, width (includes 2 for padding)
        all_opt_w = sum(w for _, w in optional_cols)
        budget = (terminal_width - 9 - all_opt_w) // n - 2 if n > 0 else 999

        # -- Alias headers and values that exceed the budget --
        def _alias_settings(
            budget: int,
        ) -> Tuple[
            List[str],
            List[Tuple[str, str, str]],
            List[Tuple[str, str, str, str]],
            Dict[str, Dict[str, str]],
            Dict[str, Dict[str, str]],
        ]:
            """Alias headers/values exceeding budget. budget<=0 aliases everything."""
            hdrs: List[str] = []
            ax_legend: List[Tuple[str, str, str]] = []
            val_legend: List[Tuple[str, str, str, str]] = []
            d_maps: Dict[str, Dict[str, str]] = {}
            s_maps: Dict[str, Dict[str, str]] = {}
            for kidx, key in enumerate(setting_keys):
                short_name = self._short_setting_key(key)
                axis_num = kidx + 1
                ordered: List[str] = []
                for _, settings in run_settings:
                    v = settings.get(key, "(unset)")
                    if v not in ordered:
                        ordered.append(v)
                # Alias header?
                if budget <= 0 or len(short_name) > budget:
                    hdrs.append(f"a{axis_num}")
                    ax_legend.append((f"a{axis_num}", short_name, "bold dim"))
                else:
                    hdrs.append(short_name)
                # Alias values?
                max_val_w = max(
                    (len(_truncate_preview(" ".join(v.split()), 20)) for v in ordered),
                    default=0,
                )
                if budget <= 0 or max_val_w > budget:
                    d_maps[key] = {}
                    s_maps[key] = {}
                    for vidx, value in enumerate(ordered):
                        alias = f"v{axis_num}.{vidx + 1}"
                        d_maps[key][value] = alias
                        s_maps[key][value] = self._alias_style(alias)
                        val_legend.append(
                            (
                                alias,
                                short_name,
                                _truncate_preview(" ".join(value.split()), 120),
                                s_maps[key][value],
                            )
                        )
                else:
                    d_maps[key] = {
                        v: _truncate_preview(" ".join(v.split()), 20) for v in ordered
                    }
                    s_maps[key] = {v: "" for v in ordered}
            return hdrs, ax_legend, val_legend, d_maps, s_maps

        def _compute_col_widths(
            hdrs: List[str], d_maps: Dict[str, Dict[str, str]]
        ) -> Tuple[List[int], int]:
            widths = [
                max(
                    len(hdrs[i]), max((len(d_maps[k][v]) for v in d_maps[k]), default=0)
                )
                for i, k in enumerate(setting_keys)
            ]
            return widths, sum(w + 2 for w in widths)

        # Try budget-based aliasing first
        col_headers, axis_legend_rows, value_legend_rows, display_maps, style_maps = (
            _alias_settings(budget)
        )
        col_widths, settings_need = _compute_col_widths(col_headers, display_maps)

        # If it doesn't fit, force-alias everything
        if terminal_width - 9 - all_opt_w < settings_need + n * 2:
            (
                col_headers,
                axis_legend_rows,
                value_legend_rows,
                display_maps,
                style_maps,
            ) = _alias_settings(0)
            col_widths, settings_need = _compute_col_widths(col_headers, display_maps)

        # Drop optional columns if even fully-aliased content doesn't fit
        visible: set[str] = {name for name, _ in optional_cols}
        for drop_group in [
            {"=0", "=1"},
            {"mix"},
            {"rollouts"},
            {"unique prompts"},
            {"runs"},
        ]:
            opt_w = sum(w for name, w in optional_cols if name in visible)
            if terminal_width - 9 - opt_w >= settings_need + n * 2:
                break
            visible -= drop_group
        show = visible.__contains__

        # Distribute leftover space evenly across setting columns.
        opt_w = sum(w for name, w in optional_cols if name in visible)
        leftover = max(terminal_width - 9 - opt_w - settings_need, 0)
        extra = leftover // n if n > 0 else 0

        for idx, (header, cw) in enumerate(zip(col_headers, col_widths)):
            header_style = "bold reverse" if highlight_col == idx else "bold dim"
            table.add_column(
                header,
                header_style=header_style,
                width=cw + extra,
                no_wrap=True,
            )
        if show("runs"):
            table.add_column("runs", justify="right", width=5, header_style="bold dim")
        if show("rollouts"):
            table.add_column(
                "rollouts", justify="right", width=9, header_style="bold dim"
            )
        if show("unique prompts"):
            table.add_column(
                "unique prompts", justify="right", width=8, header_style="bold dim"
            )
        table.add_column("avg", justify="right", width=7, header_style="bold #e5c07b")
        if show("=0"):
            table.add_column("=0", justify="right", width=5, header_style="bold red")
        if show("=1"):
            table.add_column("=1", justify="right", width=5, header_style="bold green")
        if show("mix"):
            table.add_column("mix", width=20, header_style="bold dim")

        for _group_key_val, group in rows:
            rewards = cast(List[float], group["rewards"])
            avg_rewards = cast(List[float], group["avg_rewards"])
            avg_reward = (
                (sum(rewards) / len(rewards))
                if rewards
                else ((sum(avg_rewards) / len(avg_rewards)) if avg_rewards else None)
            )
            total = len(rewards)
            zero_count = next(
                (
                    count
                    for label, count, _ in _reward_bucket_counts(rewards)
                    if label == "=0"
                ),
                0,
            )
            one_count = next(
                (
                    count
                    for label, count, _ in _reward_bucket_counts(rewards)
                    if label == "=1"
                ),
                0,
            )
            # Build setting cells.
            setting_cells: List[Text] = []
            for key in setting_keys:
                if group_by_key is None or key == group_by_key:
                    # Show actual value — find it from any run in the group.
                    value = group["run_settings"][0].get(key, "(unset)")
                    setting_cells.append(
                        Text(
                            display_maps[key][value],
                            style=style_maps[key][value] or "",
                        )
                    )
                else:
                    # Non-grouped column: show "N values" or the value if uniform.
                    distinct = {s.get(key, "(unset)") for s in group["run_settings"]}
                    if len(distinct) == 1:
                        value = next(iter(distinct))
                        setting_cells.append(
                            Text(
                                display_maps[key][value],
                                style=style_maps[key][value] or "dim",
                            )
                        )
                    else:
                        setting_cells.append(
                            Text(f"{len(distinct)} values", style="dim italic")
                        )

            prompt_count = self._distinct_prompts_by_group.get(_group_key_val)
            row_cells: list[Any] = list(setting_cells)
            if show("runs"):
                row_cells.append(str(len(cast(List[RunInfo], group["runs"]))))
            if show("rollouts"):
                row_cells.append(str(len(rewards)) if rewards else "—")
            if show("unique prompts"):
                row_cells.append(
                    Text(
                        str(prompt_count) if prompt_count is not None else "?",
                        style="dim",
                    )
                )
            row_cells.append(
                Text(
                    _format_reward_value(avg_reward) if avg_reward is not None else "—",
                    style=_reward_style(avg_reward)
                    if avg_reward is not None
                    else "dim",
                )
            )
            if show("=0"):
                row_cells.append(
                    Text(
                        f"{(zero_count / total):.0%}" if total else "—",
                        style=self._share_style(
                            (zero_count / total) if total else 0.0, False
                        ),
                    )
                )
            if show("=1"):
                row_cells.append(
                    Text(
                        f"{(one_count / total):.0%}" if total else "—",
                        style=self._share_style(
                            (one_count / total) if total else 0.0, True
                        ),
                    )
                )
            if show("mix"):
                row_cells.append(self._build_reward_mix_bar(rewards))
            table.add_row(*row_cells)

        return table, axis_legend_rows, value_legend_rows

    def _build_argument_legend(
        self,
        axis_rows: List[Tuple[str, str, str]],
        value_rows: List[Tuple[str, str, str, str]],
    ) -> Group | Text:
        """Build the argument legend.

        axis_rows: (alias, full_name, style)
        value_rows: (alias, full_name, preview, style)
        """
        if not axis_rows and not value_rows:
            return Text()

        def _make_table(*columns: Tuple[str, dict]) -> Table:
            t = Table(
                box=box.SIMPLE_HEAD,
                expand=True,
                show_edge=False,
                pad_edge=False,
                padding=(0, 1),
                collapse_padding=True,
            )
            for name, kwargs in columns:
                t.add_column(name, header_style="bold dim", **kwargs)
            return t

        items: List[Any] = []

        if axis_rows:
            t = _make_table(
                ("Alias", {"width": 8, "no_wrap": True}),
                ("Full Name", {"ratio": 1, "no_wrap": True}),
            )
            for alias, full_name, style in axis_rows:
                t.add_row(
                    Text(alias, style=style),
                    Text(full_name, style="dim"),
                )
            items.extend([Text("Arg name legend", style="bold dim"), t])

        if value_rows:
            t = _make_table(
                ("Alias", {"width": 8, "no_wrap": True}),
                ("Full Name", {"ratio": 1, "no_wrap": True}),
                ("Value", {"ratio": 2}),
            )
            for alias, full_name, preview, style in value_rows:
                t.add_row(
                    Text(alias, style=style),
                    Text(full_name, style="dim"),
                    Text(preview, style="dim"),
                )
            if items:
                items.append(Text(""))
            items.extend([Text("Arg value legend", style="bold dim"), t])

        return Group(*items)

    def _build_comparison_outcomes(self) -> Group:
        highlight_col = self._group_cursor if self._group_mode else None
        outcomes_table, axis_legend, value_legend = self._build_grouped_outcomes_table(
            self._stats_by_path,
            self._setting_keys,
            self._run_settings,
            group_by_key=self._grouped_by_key,
            highlight_col=highlight_col,
        )
        items: List[Any] = [
            Text(""),
            Group(
                Text("Outcome groups", style="bold dim"),
                outcomes_table,
            ),
        ]
        legend = self._build_argument_legend(axis_legend, value_legend)
        if isinstance(legend, Group) or (isinstance(legend, Text) and legend.plain):
            items.extend([Text(""), legend])
        return Group(*items)


class BrowseRunsScreen(Screen):
    """Single-screen browser for environments, models, and runs."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "enter_selected", "Open/toggle", priority=True),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("v", "compare_selected", "Compare"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index
        self._run_overview_cache: Dict[Path, RunOverviewStats] = {}
        self._click_selected_node: object | None = None

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
        buffer = StringIO()
        Console(
            file=buffer,
            force_terminal=False,
            color_system=None,
            width=180,
        ).print(self._details_for(payload))
        self.app.push_screen(
            CompactCopyScreen(
                [
                    RolloutCopyItem(
                        key="details", label="Details", body=buffer.getvalue().rstrip()
                    )
                ],
                start_key="details",
                title="Copy Details",
            )
        )

    def action_compare_selected(self) -> None:
        tree = self.query_one("#run-browser-tree", Tree)
        payload = getattr(getattr(tree, "cursor_node", None), "data", None)
        if not isinstance(payload, BrowserNodeData):
            return

        env_id = payload.env_id
        model = payload.model
        if payload.kind == "run" and payload.run is not None:
            env_id = payload.run.env_id
            model = payload.run.model
        elif payload.kind == "env":
            all_runs: List[RunInfo] = []
            for model_runs in self.index.get(env_id, {}).values():
                all_runs.extend(model_runs)
            runs = _sorted_runs(all_runs)
            if not runs:
                return
            self.app.push_screen(CompareRunsScreen(env_id, None, runs))
            return
        elif payload.kind != "model":
            return

        runs = _sorted_runs(self.index.get(env_id, {}).get(model, []))
        if not runs:
            return
        self.app.push_screen(CompareRunsScreen(env_id, model, runs))

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
                data=BrowserNodeData(
                    kind="env",
                    env_id=env_id,
                    tree_name=env_id,
                    tree_suffix=(
                        ("  ", ""),
                        (f"{len(models)} models", "dim"),
                        ("  ", ""),
                        (f"{total_runs} runs", "dim"),
                    ),
                ),
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
                    data=BrowserNodeData(
                        kind="model",
                        env_id=env_id,
                        model=model,
                        tree_name=model,
                        tree_suffix=(
                            ("  ", ""),
                            (f"{len(runs)} runs", "dim"),
                        ),
                    ),
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
                            tree_name=run.run_id,
                            tree_suffix=(
                                (
                                    f"  {_format_reward_value(avg_reward)}",
                                    _reward_style(avg_reward),
                                ),
                            )
                            if avg_reward is not None
                            else (),
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

    def action_enter_selected(self) -> None:
        """Enter key: immediately open the highlighted run or toggle folder."""
        tree = self.query_one("#run-browser-tree", Tree)
        node = tree.cursor_node
        if node is None:
            return
        payload = node.data
        if not isinstance(payload, BrowserNodeData):
            return
        if payload.kind == "run" and payload.run is not None:
            self.app.push_screen(ViewRunScreen(payload.run))
            return
        if node.allow_expand:
            node.toggle()

    @on(Tree.NodeSelected, "#run-browser-tree")
    def on_tree_selected(self, event: Tree.NodeSelected) -> None:
        """Click: first click selects, second click enters rollout view."""
        payload = event.node.data
        if not isinstance(payload, BrowserNodeData):
            return
        if payload.kind == "run" and payload.run is not None:
            if self._click_selected_node is event.node:
                self.app.push_screen(ViewRunScreen(payload.run))
            else:
                self._click_selected_node = event.node
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

            variation_rows, hidden_variations = _run_setting_variation_rows(runs)
            if variation_rows:
                items.extend(
                    [
                        Text(""),
                        _build_settings_table(
                            variation_rows,
                            "Setting variations",
                            value_header="Across runs",
                        ),
                    ]
                )
                if hidden_variations:
                    items.extend(
                        [
                            Text(""),
                            Text(
                                f"{hidden_variations} more varied settings not shown",
                                style="dim",
                            ),
                        ]
                    )

            items.extend(
                [
                    Text(""),
                    Text("Press v to open compare mode for these runs", style="dim"),
                ]
            )

        return Group(*items)

    def _build_run_details(
        self,
        run: RunInfo,
        stats: Optional[RunOverviewStats] = None,
    ) -> Group:
        meta = run.load_metadata()
        rewards = stats.rewards if stats is not None else []
        setting_rows = _run_setting_rows(meta)

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
                    values.items(), key=lambda item: _int_like_sort_key(item[0])
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
        if setting_rows:
            items.extend(
                [
                    _build_settings_table(setting_rows, "Run settings"),
                    Text(""),
                ]
            )
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
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("l", "show_logs", "Logs"),
        Binding("r", "show_rollouts", "Rollouts"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("e", "expand_all", "Expand all"),
        Binding("x", "collapse_all", "Collapse all"),
        Binding("s", "search", "Search"),
        Binding("m", "toggle_markdown_math", "Toggle markdown"),
        Binding("c", "copy", "Copy"),
        Binding("ctrl+c", "copy", show=False),
    ]

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Dynamically show/hide footer bindings based on view mode."""
        mode = getattr(self, "_view_mode", "rollouts")
        hide_in_logs = (
            "expand_all",
            "collapse_all",
            "toggle_markdown_math",
            "show_logs",
        )
        hide_in_rollouts = ("show_rollouts",)
        if mode == "logs":
            if action in hide_in_logs:
                return False
        else:
            if action in hide_in_rollouts:
                return False
        return True

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = LazyRunResults(run)
        self._record_count = self.records.count_hint()
        self.current_record_idx = 0
        self._prompt_text: str = ""
        self._completion_text: str = ""
        self._highlight_regex: Optional[re.Pattern] = None
        self._highlight_column: Optional[str] = None
        self._highlight_timer = None
        self._previous_animation_level: Optional[AnimationLevel] = None
        self._render_markdown_math = True
        # Log viewer state
        # Tab 0 = "all" (merged), tab 1+ = individual files
        self._log_files: List[Path] = _discover_log_files(run.path)
        self._log_loaders: Dict[int, LazyLogFile] = {}
        self._merged_log_lines: Optional[List[str]] = None
        self._active_log_tab: int = 0
        self._view_mode: Literal["rollouts", "logs"] = "rollouts"
        self._log_highlight_regex: Optional[re.Pattern] = None
        self._log_highlight_timer = None
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
                with Panel(id="logs-panel", classes="logs-panel"):
                    yield Label(
                        Text("Logs", style="bold"),
                        id="logs-header",
                        classes="column-header",
                    )
                    yield Static(
                        "", id="logs-tab-bar", classes="subtitle", markup=False
                    )
                    yield LogScrollPane(id="logs-scroll")
                with Panel(id="details-panel", classes="details-panel"):
                    yield Label(Text("Details", style="bold"), classes="column-header")
                    with TabbedContent(initial="details-task", id="details-tabs"):
                        with TabPane("Task", id="details-task"):
                            yield TabbedScrollPane(
                                Static("", id="task-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Score", id="details-score"):
                            yield TabbedScrollPane(
                                Static("", id="score-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Usage", id="details-usage"):
                            yield TabbedScrollPane(
                                Static("", id="usage-content", markup=False),
                                classes="details-scroll surface-scroll",
                            )
                        with TabPane("Info", id="details-info"):
                            yield TabbedScrollPane(
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

    def _build_history_summary_text(self, record: Dict[str, Any]) -> Text:
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
        return Text.assemble(*parts)

    def _build_header_metric_text(self) -> Text:
        meta = self.run.load_metadata()
        stats: List[Tuple[str, Any]] = []

        pass_at_k = meta.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            for key in sorted(pass_at_k.keys(), key=_int_like_sort_key):
                stats.append((f"pass@{key}", pass_at_k[key]))

        pass_all_k = meta.get("pass_all_k")
        if isinstance(pass_all_k, dict):
            for key in sorted(pass_all_k.keys(), key=_int_like_sort_key):
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

        breakdown = sorted(_extract_numeric_metric_values(record).items())
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
        self.update_display()
        self.call_after_refresh(self._focus_primary_content)
        self._update_responsive_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(event.size.width)

    def on_unmount(self) -> None:
        self.records.close()
        for loader in self._log_loaders.values():
            loader.close()
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

    def _hydrate_rollout_option(self, index: int) -> None:
        rollout_list = self.query_one("#rollout-list", OptionList)
        if not (0 <= index < rollout_list.option_count):
            return
        rollout_list.replace_option_prompt_at_index(
            index,
            self._build_rollout_prompt(index, self.records[index]),
        )

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
            reasoning = _stringify_message_reasoning(message)
            if role == "assistant":
                out.append("assistant: ", style="bold")
            elif role == "tool":
                out.append("tool result: ", style="bold dim")
            else:
                out.append(f"{role}: ", style="bold dim")
            if reasoning:
                out.append("\n")
                out.append("reasoning:\n", style="dim")
                out.append(reasoning, style="dim")
                out.append("\n")
                if content:
                    out.append("\n")
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
        self.query_one("#rollout-summary", Label).update(
            self._build_rollout_summary_text(record)
        )
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

    # ------ Log viewer ------

    def action_show_logs(self) -> None:
        if not self._log_files:
            self.notify("No log files available for this run", severity="warning")
            return
        if self._view_mode == "logs":
            return
        self._view_mode = "logs"
        self.query_one("#history-panel", Panel).display = False
        self.query_one("#logs-panel", Panel).display = True
        self._populate_logs_view()
        self.query_one("#logs-scroll", LogScrollPane).focus()
        self.refresh_bindings()

    def action_show_rollouts(self) -> None:
        if self._view_mode == "rollouts":
            return
        self._view_mode = "rollouts"
        self.query_one("#logs-panel", Panel).display = False
        self.query_one("#history-panel", Panel).display = True
        self._focus_primary_content()
        self.refresh_bindings()

    def _cycle_log_tab(self, delta: int) -> None:
        num_tabs = self._log_tab_count()
        if num_tabs < 2:
            return
        self._active_log_tab = (self._active_log_tab + delta) % num_tabs
        self._populate_logs_view()

    def _log_tab_count(self) -> int:
        """Number of log tabs: 'all' + individual files if 2+ files, else just 1."""
        if len(self._log_files) >= 2:
            return len(self._log_files) + 1  # "all" tab + individual files
        return len(self._log_files)  # 0 or 1

    def _build_log_tab_bar(self) -> Text:
        num_tabs = self._log_tab_count()
        if num_tabs <= 1:
            return Text()
        text = Text()
        # Tab 0 = "all", tabs 1+ = individual files
        labels = ["all"] + [_log_tab_label(p) for p in self._log_files]
        for i, label in enumerate(labels):
            if i > 0:
                text.append("  ")
            if i == self._active_log_tab:
                text.append(f"[{label}]", style="bold")
            else:
                text.append(f" {label} ", style="dim")
        text.append("  ")
        text.append("(←/→ to switch)", style="dim italic")
        return text

    def _get_active_log_lines(self) -> Tuple[List[str], str]:
        """Return (lines, tab_label) for the active log tab."""
        is_merged = len(self._log_files) >= 2 and self._active_log_tab == 0
        if is_merged:
            if self._merged_log_lines is None:
                self._merged_log_lines = _merge_log_files(self._log_files)
            return self._merged_log_lines, "all"
        # Individual file tab — index into _log_files
        file_idx = (
            self._active_log_tab - 1
            if len(self._log_files) >= 2
            else self._active_log_tab
        )
        if file_idx not in self._log_loaders:
            self._log_loaders[file_idx] = LazyLogFile(self._log_files[file_idx])
        loader = self._log_loaders[file_idx]
        line_count = len(loader)
        lines = [loader.get_line(i) for i in range(line_count)]
        return lines, _log_tab_label(self._log_files[file_idx])

    def _populate_logs_view(self) -> None:
        if not self._log_files:
            self.query_one("#logs-tab-bar", Static).update(
                Text("No log files available", style="dim")
            )
            return

        # Update tab bar
        self.query_one("#logs-tab-bar", Static).update(self._build_log_tab_bar())

        lines, log_name = self._get_active_log_lines()
        line_count = len(lines)

        # Update header
        self.query_one("#logs-header", Label).update(
            Text.assemble(
                ("Logs", "bold"),
                (f"  {log_name}", "dim"),
                (f"  ({line_count:,} lines)", "dim"),
            )
        )

        # Build log content
        container = self.query_one("#logs-scroll", LogScrollPane)
        container.remove_children()

        if line_count == 0:
            container.mount(Static(Text("(empty log file)", style="dim"), markup=False))
            return

        text = Text()
        # Cap display at last MAX_DISPLAY_LINES lines
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        if start > 0:
            text.append(
                f"... {start:,} earlier lines not shown ...\n\n", style="dim italic"
            )
        for i in range(start, line_count):
            if i > start:
                text.append("\n")
            line = lines[i]
            if self._log_highlight_regex:
                _append_styled_log_line(text, line)
                # Apply highlights on top
                offset = len(text.plain) - len(line)
                for match in self._log_highlight_regex.finditer(line):
                    text.stylize(
                        "reverse", offset + match.start(), offset + match.end()
                    )
            else:
                _append_styled_log_line(text, line)

        container.mount(Static(text, markup=False, classes="log-content"))

    def _build_search_lines(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """Build tagged (section_index, nested_index, line) lists for search."""
        sections = self._history_section_data(record)
        prompt_lines: List[Tuple[int, int, str]] = []
        completion_lines: List[Tuple[int, int, str]] = []
        for idx, section in enumerate(sections):
            target = prompt_lines if section.column == "prompt" else completion_lines

            def _append_body(tgt: List[Tuple[int, int, str]], body: str) -> None:
                for line in body.splitlines():
                    tgt.append((idx, -1, line))

            def _append_nested() -> None:
                for nested_idx, nested in enumerate(section.nested_sections):
                    nested_target = (
                        prompt_lines if nested.column == "prompt" else completion_lines
                    )
                    for line in nested.body.splitlines():
                        nested_target.append((idx, nested_idx, line))

            if section.body_first:
                _append_body(target, section.body)
                _append_nested()
            else:
                _append_nested()
                _append_body(target, section.body)
        return prompt_lines, completion_lines

    def action_search(self) -> None:
        if self._view_mode == "logs":
            self._search_logs()
            return
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        prompt_lines, completion_lines = self._build_search_lines(record)
        self.app.push_screen(
            SearchScreen(prompt_lines, completion_lines),
            self._handle_search_result,
        )

    def _search_logs(self) -> None:
        if not self._log_files:
            return
        lines, _ = self._get_active_log_lines()
        line_count = len(lines)
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        log_lines: List[Tuple[int, int, str]] = [
            (0, -1, lines[i]) for i in range(start, line_count)
        ]
        self.app.push_screen(
            SearchScreen([], log_lines),
            self._handle_log_search_result,
        )

    def _handle_log_search_result(self, result: Optional[SearchResult]) -> None:
        if self._log_highlight_timer is not None:
            self._log_highlight_timer.stop()
            self._log_highlight_timer = None
        self._log_highlight_regex = None
        if result is not None:
            try:
                self._log_highlight_regex = re.compile(result.pattern, re.IGNORECASE)
            except re.error:
                return
            self._log_highlight_timer = self.set_timer(3.0, self._clear_log_highlight)
        self._populate_logs_view()
        if result is not None and self._log_highlight_regex is not None:
            self._scroll_to_first_log_match()

    def _scroll_to_first_log_match(self) -> None:
        """Scroll the logs panel so the first matching line is visible."""
        if not self._log_highlight_regex or not self._log_files:
            return
        lines, _ = self._get_active_log_lines()
        line_count = len(lines)
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        # Find the first matching line index (relative to displayed range)
        first_match_display_idx: Optional[int] = None
        for i in range(start, line_count):
            if self._log_highlight_regex.search(lines[i]):
                first_match_display_idx = i - start
                break
        if first_match_display_idx is None:
            return
        # Account for the "... N earlier lines not shown ..." header (2 lines)
        offset_lines = 2 if start > 0 else 0
        target_line = first_match_display_idx + offset_lines
        container = self.query_one("#logs-scroll", LogScrollPane)

        def _do_scroll() -> None:
            # Estimate scroll position: each log line is roughly 1 row of height
            # in the Static widget. We scroll to a Y offset proportional to the
            # target line relative to total content height.
            content_height = container.virtual_size.height
            visible_height = container.size.height
            total_lines = (line_count - start) + offset_lines
            if total_lines <= 0 or content_height <= visible_height:
                return
            fraction = target_line / total_lines
            target_y = int(fraction * content_height)
            # Center the match in the viewport
            target_y = max(0, target_y - visible_height // 2)
            container.scroll_to(y=target_y, animate=False)

        self.call_after_refresh(_do_scroll)

    def _clear_log_highlight(self) -> None:
        self._log_highlight_regex = None
        self._log_highlight_timer = None
        if self._view_mode == "logs":
            self._populate_logs_view()

    def action_copy(self) -> None:
        if self._view_mode == "logs":
            self._copy_logs()
            return
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        self.app.push_screen(
            RolloutCopyScreen(
                self._build_rollout_copy_items(record),
                start_key="snapshot",
                title=f"Copy Rollout #{self.current_record_idx}",
            )
        )

    def _copy_logs(self) -> None:
        if not self._log_files:
            return
        items: List[RolloutCopyItem] = []
        has_merged = len(self._log_files) >= 2
        if has_merged:
            # "all" tab uses merged lines
            if self._merged_log_lines is None:
                self._merged_log_lines = _merge_log_files(self._log_files)
            items.append(
                RolloutCopyItem(
                    key="log-all",
                    label="Log: all (merged)",
                    body="\n".join(self._merged_log_lines),
                )
            )
        for idx, path in enumerate(self._log_files):
            items.append(
                RolloutCopyItem(
                    key=f"log-file-{idx}",
                    label=f"Log: {_log_tab_label(path)}",
                    body=path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        # Select the copy target matching the active tab
        if has_merged and self._active_log_tab == 0:
            current_key = "log-all"
        else:
            file_idx = self._active_log_tab - 1 if has_merged else self._active_log_tab
            current_key = f"log-file-{file_idx}"
        self.app.push_screen(
            RolloutCopyScreen(
                items,
                start_key=current_key,
                title="Copy Logs",
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

    @on(TabbedContent.TabActivated, "#details-tabs")
    def on_details_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Focus the scroll pane in the newly active details tab."""
        for child in event.pane.children:
            if isinstance(child, TabbedScrollPane):
                child.focus()
                break

    def _should_skip_focus(self, widget: Widget) -> bool:
        """Return True for widgets that should be skipped during tab cycling."""
        # Skip the scroll container itself — only its children should get focus.
        if widget.id == "completion-scroll":
            return True
        # Skip the details tab bar — the TabbedScrollPane handles tab switching.
        if isinstance(widget, ContentTabs):
            return True
        # Skip widgets inside hidden ancestors (compact-layout panels,
        # inactive tab panes, etc.).
        node: DOMNode | None = widget.parent
        while node is not None:
            if isinstance(node, Widget) and not node.display:
                return True
            node = node.parent
        return False

    def action_focus_next_pane(self) -> None:
        starting = self.focused
        self.focus_next()
        first_candidate = self.focused
        while self.focused is not None and self.focused is not starting:
            if not self._should_skip_focus(self.focused):
                break
            self.focus_next()
            if self.focused is first_candidate:
                break

    def action_focus_prev_pane(self) -> None:
        starting = self.focused
        self.focus_previous()
        first_candidate = self.focused
        while self.focused is not None and self.focused is not starting:
            if not self._should_skip_focus(self.focused):
                break
            self.focus_previous()
            if self.focused is first_candidate:
                break

    def _center_scroll_target(self) -> VerticalScroll:
        if self._view_mode == "logs":
            return self.query_one("#logs-scroll", LogScrollPane)
        return self.query_one("#completion-scroll", VerticalScroll)

    def action_history_page_up(self) -> None:
        self._center_scroll_target().scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        self._center_scroll_target().scroll_page_down(animate=False)

    def action_history_home(self) -> None:
        self._center_scroll_target().scroll_home(animate=False)

    def action_history_end(self) -> None:
        self._center_scroll_target().scroll_end(animate=False)

    def _make_body_widget(self, body: str, column: str) -> Widget:
        """Create the appropriate body widget based on render mode."""
        if self._render_markdown_math and not (
            self._highlight_regex and self._highlight_column == column
        ):
            return MathMarkdown(body, classes="section-body")
        text = Text(body)
        if self._highlight_regex and self._highlight_column == column:
            _stylize_matches(text, self._highlight_regex, "reverse")
        return Static(text, classes="section-body", markup=False)

    def _collect_section_bodies(
        self, sections: List[HistorySectionData]
    ) -> List[Tuple[str, str]]:
        """Flatten all section (body, column) pairs in DOM order."""
        result: List[Tuple[str, str]] = []
        for section in sections:
            parent = (
                [(section.body, section.column)]
                if section.body or not section.nested_sections
                else []
            )
            nested = [
                (n.body, n.column)
                for n in section.nested_sections
                if n.body or not n.nested_sections
            ]
            if section.body_first:
                result.extend(parent)
                result.extend(nested)
            else:
                result.extend(nested)
                result.extend(parent)
        return result

    def _swap_section_bodies(self) -> None:
        """Re-render all .section-body widgets in-place (preserves collapsed state)."""
        if not (self.records and self.is_mounted):
            return
        record = self.records[self.current_record_idx]
        section_data = self._history_section_data(record)
        body_entries = self._collect_section_bodies(section_data)
        container = self.query_one("#completion-scroll", VerticalScroll)
        body_widgets = list(container.query(".section-body"))
        for i, body_widget in enumerate(body_widgets):
            parent = body_widget.parent
            if not isinstance(parent, Widget) or i >= len(body_entries):
                continue
            body, column = body_entries[i]
            replacement = self._make_body_widget(body, column)
            parent.mount(replacement, after=body_widget)
            body_widget.remove()

    def action_toggle_markdown_math(self) -> None:
        self._render_markdown_math = not self._render_markdown_math
        self._swap_section_bodies()

    def _handle_search_result(self, result: Optional[SearchResult]) -> None:
        if result is not None:
            self._set_highlight(result)

    def _set_highlight(
        self, result: Optional[SearchResult], *, repaint: bool = True
    ) -> None:
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
            self._highlight_timer = None

        had_highlight = self._highlight_regex is not None
        self._highlight_regex = None
        self._highlight_column = None
        self._highlight_section_index: int = 0
        self._highlight_nested_index: int = -1

        if result is not None:
            try:
                self._highlight_regex = re.compile(result.pattern, re.IGNORECASE)
            except re.error:
                return
            self._highlight_column = result.column
            self._highlight_section_index = result.section_index
            self._highlight_nested_index = result.nested_index
            self._highlight_timer = self.set_timer(
                3.0, lambda: self._set_highlight(None)
            )

        if repaint and self.is_mounted and (had_highlight or result is not None):
            self._swap_section_bodies()

            # For new searches, expand the target section and scroll to it.
            if result is not None:
                container = self.query_one("#completion-scroll", VerticalScroll)
                self._expand_and_scroll_to_match(container)

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

    def _reasoning_section_data(
        self,
        message: Dict[str, Any],
        *,
        collapsed: bool = True,
    ) -> Tuple[HistorySectionData, ...]:
        reasoning = _stringify_message_reasoning(message)
        if not reasoning:
            return ()
        return (
            HistorySectionData(
                title="Reasoning",
                body=reasoning,
                column="completion",
                collapsed=collapsed,
                classes="history-section reasoning-section nested-section",
            ),
        )

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
                reasoning_sections = self._reasoning_section_data(message)

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
                        nested_sections=reasoning_sections,
                        body_first=not reasoning_sections,
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
                            _stringify_message(output)
                            if isinstance(output, dict)
                            else str(output)
                        )
                        if self._highlight_regex.search(output_text):
                            collapsed = False
                            break

            nested_sections: List[HistorySectionData] = list(
                self._reasoning_section_data(message)
            )
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
                    _stringify_message(matched_output)
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
                    _stringify_message(output_message)
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
                    body_first=False if nested_sections else True,
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

    def _expand_and_scroll_to_match(self, container: VerticalScroll) -> None:
        """Expand the target section (and nested subsection) and scroll to it."""
        # Get top-level sections only (direct children of the scroll container).
        sections = [
            child for child in container.children if isinstance(child, Collapsible)
        ]
        idx = self._highlight_section_index
        if not (0 <= idx < len(sections)):
            return
        parent = sections[idx]
        if parent.collapsed:
            parent.collapsed = False

        # If the hit is in a nested section, expand that too.
        scroll_target = parent
        nested_idx = self._highlight_nested_index
        if nested_idx >= 0:
            nested_collapsibles = [
                child for child in parent.query(Collapsible) if child is not parent
            ]
            if 0 <= nested_idx < len(nested_collapsibles):
                nested = nested_collapsibles[nested_idx]
                if nested.collapsed:
                    nested.collapsed = False
                scroll_target = nested

        self.call_after_refresh(lambda t=scroll_target: self._scroll_to_section(t))

    def _scroll_to_section(self, section: Collapsible) -> None:
        section.scroll_visible(animate=False)
        title_widget = next(iter(section.children), None)
        if title_widget is not None and getattr(title_widget, "can_focus", False):
            title_widget.focus()

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
        body = [_indent_block(section.body, f"{indent}  ")] if section.body else []
        nested = [
            self._render_history_section_copy_text(child, depth=depth + 1)
            for child in section.nested_sections
        ]
        if section.body_first:
            parts.extend(body)
            parts.extend(nested)
        else:
            parts.extend(nested)
            parts.extend(body)
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

        history_summary = _text_to_plain(self._build_history_summary_text(record))
        history_text = self._render_history_copy_text(history_sections)
        history_parts = ["Completion History"]
        if history_summary:
            history_parts.append(history_summary)
        if history_text:
            history_parts.append(history_text)
        blocks.append("\n\n".join(history_parts))

        if self.is_mounted:
            active_tab_id = (
                self.query_one("#details-tabs", TabbedContent).active or "details-task"
            )
        else:
            active_tab_id = "details-task"
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

    def _section_matches_highlight(self, section: HistorySectionData) -> bool:
        if not (self._highlight_regex and self._highlight_column == section.column):
            return False
        if self._highlight_regex.search(section.title) or self._highlight_regex.search(
            section.body
        ):
            return True
        return any(
            self._section_matches_highlight(nested_section)
            for nested_section in section.nested_sections
        )

    def _make_section(self, section: HistorySectionData) -> Collapsible:
        collapsed = section.collapsed
        if self._section_matches_highlight(section):
            collapsed = False
        body_children: List[Any] = []
        if section.body:
            if not self._render_markdown_math or (
                self._highlight_regex and self._highlight_column == section.column
            ):
                text = Text(section.body)
                if self._highlight_regex and self._highlight_column == section.column:
                    _stylize_matches(text, self._highlight_regex, "reverse")
                content = Static(
                    text,
                    classes="section-body",
                    markup=False,
                )
            else:
                content = MathMarkdown(section.body, classes="section-body")
            body_children.append(content)
        elif not section.nested_sections:
            text = Text(section.body)
            content = Static(
                text,
                classes="section-body",
                markup=False,
            )
            body_children.append(content)
        nested_children = [
            self._make_section(nested_section)
            for nested_section in section.nested_sections
        ]
        children = (
            [*body_children, *nested_children]
            if section.body_first
            else [*nested_children, *body_children]
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

    .logs-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
        display: none;
    }

    #logs-scroll {
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

    #logs-scroll:focus {
        background-tint: $foreground 4%;
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

    .details-scroll:focus {
        background-tint: $foreground 4%;
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

    #compare-scroll {
        padding: 0 1 0 2;
        scrollbar-size-vertical: 2;
        scrollbar-gutter: stable;
    }

    #compare-content {
        margin-right: 8;
    }

    .browser-columns {
        height: 1fr;
        layout: horizontal;
    }

    .browser-tree-panel {
        width: 56;
        height: 1fr;
        layout: vertical;
    }

    #run-browser-tree {
        height: 1fr;
        background: $surface;
        color: $text;
        overflow-x: hidden;
    }

    #run-browser-tree:focus {
        background-tint: $foreground 4%;
    }

    .browser-details-panel {
        height: 1fr;
        width: 1fr;
    }

    #run-browser-details-scroll:focus {
        background-tint: $foreground 4%;
    }

    .compare-panel {
        height: 1fr;
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

    .compact-copy-body {
        height: 1fr;
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

    def __init__(
        self,
        prompt_lines: List[Tuple[int, int, str]],
        completion_lines: List[Tuple[int, int, str]],
    ):
        super().__init__()
        self._tagged_lines: Dict[str, List[Tuple[int, int, str]]] = {
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
        self.dismiss(
            SearchResult(
                column=selection.column,
                pattern=pattern,
                section_index=selection.section_index,
                nested_index=selection.nested_index,
            )
        )

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
        for column, tagged_lines in self._tagged_lines.items():
            hits: List[SearchHit] = []
            for line_index, (section_index, nested_index, line) in enumerate(
                tagged_lines
            ):
                if not compiled.search(line):
                    continue
                hits.append(
                    SearchHit(
                        column=column,
                        line_index=line_index,
                        line_text=line,
                        section_index=section_index,
                        nested_index=nested_index,
                    )
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
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        self.app.exit()

    def __init__(
        self,
        items: List[RolloutCopyItem],
        *,
        start_key: Optional[str] = None,
        title: str = "Copy Rollout",
    ):
        super().__init__()
        self._items = items
        self._title = title
        self._current_idx = 0
        if start_key:
            for i, item in enumerate(items):
                if item.key == start_key:
                    self._current_idx = i
                    break
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text(self._title, style="bold"))
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
        option_list.highlighted = self._current_idx
        self._sync_preview()
        self.query_one("#rollout-copy-preview", TextArea).focus()

    @on(OptionList.OptionHighlighted, "#rollout-copy-targets")
    def _on_target_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None and event.option_index != self._current_idx:
            self._current_idx = event.option_index
            self._sync_preview()

    @on(OptionList.OptionSelected, "#rollout-copy-targets")
    def _on_target_selected(self, event: OptionList.OptionSelected) -> None:
        """Click on a target: update preview and return focus to TextArea."""
        if event.option_index is not None:
            self._current_idx = event.option_index
            self._sync_preview()
        self.query_one("#rollout-copy-preview", TextArea).focus()

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

    def on_key(self, event: events.Key) -> None:
        # Only intercept arrow keys when the OptionList has focus;
        # let all keys pass through to the TextArea normally.
        option_list = self.query_one("#rollout-copy-targets", OptionList)
        if self.focused is not option_list:
            return
        if event.key in ("left", "up"):
            self._move_section(-1)
            event.prevent_default()
            event.stop()
        elif event.key in ("right", "down"):
            self._move_section(1)
            event.prevent_default()
            event.stop()

    def _move_section(self, delta: int) -> None:
        if not self._items:
            return
        self._current_idx = (self._current_idx + delta) % len(self._items)
        self.query_one(
            "#rollout-copy-targets", OptionList
        ).highlighted = self._current_idx
        self._sync_preview()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        preview = self.query_one("#rollout-copy-preview", TextArea)
        selected = preview.selected_text or ""
        copied_text = selected or item.body
        if not copied_text:
            self.query_one("#rollout-copy-status", Label).update(
                Text("Nothing to copy.", style="dim")
            )
            return
        self.app.copy_to_clipboard(copied_text)
        self._last_copied_selection = copied_text
        label = "selection" if selected else item.label.lower()
        self.query_one("#rollout-copy-status", Label).update(
            Text(f"Copied {label} ({len(copied_text):,} chars).", style="dim")
        )

    def _sync_preview(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        self.query_one("#rollout-copy-preview-label", Label).update(
            Text(f"{item.label}  ({len(item.body):,} chars)", style="bold")
        )
        self.query_one("#rollout-copy-preview", TextArea).load_text(item.body)


class CompactCopyScreen(ModalScreen[None]):
    """Compact copy screen with section tabs above a preview area."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("tab", "next_section", "Next section"),
        Binding("shift+tab", "prev_section", "Prev section"),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        self.app.exit()

    def __init__(
        self,
        items: List[RolloutCopyItem],
        *,
        start_key: Optional[str] = None,
        title: str = "Copy",
    ):
        super().__init__()
        self._items = items
        self._title = title
        self._current_idx = 0
        if start_key:
            for i, item in enumerate(items):
                if item.key == start_key:
                    self._current_idx = i
                    break
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text(self._title, style="bold"))
                yield Label("", id="compact-copy-status", classes="subtitle")
            with Panel(classes="compact-copy-body"):
                preview = TextArea(
                    "", id="compact-copy-preview", classes="copy-textarea"
                )
                preview.read_only = True
                yield preview
        yield Footer()

    def on_mount(self) -> None:
        self._sync()
        self.query_one("#compact-copy-preview", TextArea).focus()

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area.id != "compact-copy-preview":
            return
        selected = event.text_area.selected_text or ""
        if selected and selected != self._last_copied_selection:
            self.app.copy_to_clipboard(selected)
            self._last_copied_selection = selected
            self.query_one("#compact-copy-status", Label).update(
                Text(f"Copied selection ({len(selected):,} chars).", style="dim")
            )

    def action_close(self) -> None:
        self.dismiss(None)

    def on_key(self, event: events.Key) -> None:
        if event.key in ("tab", "shift+tab", "backtab"):
            if event.key == "tab":
                self.action_next_section()
            else:
                self.action_prev_section()
            event.prevent_default()
            event.stop()

    def action_prev_section(self) -> None:
        if self._items:
            self._current_idx = (self._current_idx - 1) % len(self._items)
            self._sync()

    def action_next_section(self) -> None:
        if self._items:
            self._current_idx = (self._current_idx + 1) % len(self._items)
            self._sync()

    def action_copy(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        preview = self.query_one("#compact-copy-preview", TextArea)
        selected = preview.selected_text or ""
        copied_text = selected or item.body
        if not copied_text:
            self.query_one("#compact-copy-status", Label).update(
                Text("Nothing to copy.", style="dim")
            )
            return
        self.app.copy_to_clipboard(copied_text)
        self._last_copied_selection = copied_text
        label = "selection" if selected else item.label.lower()
        self.query_one("#compact-copy-status", Label).update(
            Text(f"Copied {label} ({len(copied_text):,} chars).", style="dim")
        )

    def _sync(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        preview = self.query_one("#compact-copy-preview", TextArea)
        preview.load_text(item.body)


class CopyScreen(ModalScreen[None]):
    """Modal screen for selecting and copying prompt/completion text."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close", "Back (esc/b)"),
        Binding("b,backspace", "close", show=False),
        Binding("tab", "cycle_column", "Next column"),
        Binding("shift+tab", "cycle_column", "Prev column"),
        Binding("c", "copy", "Copy"),
    ]

    async def action_quit(self) -> None:
        self.app.exit()

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
            copy_text = f"c: copy selection ({count} {unit})"
        else:
            active_label = (
                self._prompt_label
                if self._active_column == "prompt"
                else self._completion_label
            ).lower()
            copy_text = f"c: copy {active_label}"
        self.query_one("#copy-hint", Label).update(Text(copy_text, style="dim"))


def main() -> None:
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    VerifiersTUI(discover_results(env_dir, outputs_dir)).run()


if __name__ == "__main__":
    main()
