"""
Textual-based TUI for viewing verifiers eval results.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self._count: Optional[int] = None

        meta = run.load_metadata()
        num_examples = meta.get("num_examples")
        rollouts_per_example = meta.get("rollouts_per_example")
        if isinstance(num_examples, int) and num_examples >= 0:
            if isinstance(rollouts_per_example, int) and rollouts_per_example >= 0:
                self._count = num_examples * rollouts_per_example
            else:
                self._count = num_examples

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


def load_run_results(run: RunInfo) -> LazyRunResults:
    """Open results.jsonl lazily."""
    return LazyRunResults(run)


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
            content = _stringify_message_content(msg.get("content", ""))
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
                        except json.JSONDecodeError:
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
    parsed: List[Any] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, str):
            try:
                parsed.append(json.loads(tool_call))
            except json.JSONDecodeError:
                parsed.append(tool_call)
        else:
            parsed.append(tool_call)
    return parsed


def _truncate_preview(text: str, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _format_message_preview(message: Any, fallback: str) -> str:
    if not isinstance(message, dict):
        return _truncate_preview(str(message), 56) or fallback

    content = _stringify_message_content(message.get("content", ""))
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if content:
        return _truncate_preview(content, 56)
    if tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            function = first.get("function", {})
            name = function.get("name") or first.get("name") or "tool call"
            return f"calls {name}"
        return f"calls {_truncate_preview(str(first), 48)}"
    return fallback


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
    return str(value) if value not in (None, "") else "N/A"


def _format_compact_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _numeric_reward(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _pretty_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _format_tool_arguments(arguments: Any) -> str:
    parsed = _coerce_info_value(arguments)
    if isinstance(parsed, dict):
        if set(parsed.keys()) == {"code"}:
            return str(parsed["code"])
        return _pretty_json_or_str(parsed)
    if isinstance(parsed, list):
        return _pretty_json_or_str(parsed)
    return str(arguments) if arguments not in (None, "") else "No arguments"


def _tool_call_parts(tool_call: Any) -> Tuple[str, str, Optional[str]]:
    if not isinstance(tool_call, dict):
        return "tool call", str(tool_call), None

    function = tool_call.get("function")
    payload = function if isinstance(function, dict) else tool_call
    name = str(payload.get("name") or tool_call.get("name") or "tool call")
    arguments = _format_tool_arguments(
        payload.get("arguments", tool_call.get("arguments", ""))
    )
    call_id = tool_call.get("id")
    return name, arguments, str(call_id) if call_id not in (None, "") else None


def _tool_output_preview(message: Any) -> str:
    if not isinstance(message, dict):
        return _truncate_preview(str(message), 44)
    content = _stringify_message_content(message.get("content", ""))
    first_line = _first_non_empty_line(content)
    return _truncate_preview(first_line or content or "No output yet", 44)


def _tool_group_preview(message: Any, tool_outputs: List[Any]) -> str:
    base = _format_message_preview(message, "tool work")
    if not tool_outputs:
        return base
    return _truncate_preview(f"{base} -> {_tool_output_preview(tool_outputs[0])}", 68)


def _coerce_info_value(info: Any) -> Any:
    """Return parsed JSON when info is a JSON string, otherwise original value."""
    if not isinstance(info, str):
        return info
    stripped = info.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return info
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return info


def format_info_for_details(info: Any) -> str:
    """Format record info for the details panel in rollout view."""
    info_value = _coerce_info_value(info)
    if isinstance(info_value, (dict, list)):
        return _pretty_json_or_str(info_value)
    return str(info_value)


def _run_average_rewards(runs: List[RunInfo]) -> List[float]:
    rewards: List[float] = []
    for run in runs:
        reward = _numeric_reward(run.load_metadata().get("avg_reward"))
        if reward is not None:
            rewards.append(reward)
    return rewards


def _append_reward_histogram(out: Text, values: List[float], heading: str) -> None:
    if out.plain:
        out.append("\n\n")
    out.append(f"{heading}\n", style="bold dim")
    if not values:
        out.append("No reward data", style="dim")
        return

    avg_reward = sum(values) / len(values)
    out.append("count ", style="bold")
    out.append(str(len(values)))
    out.append("   avg ", style="bold")
    out.append(f"{avg_reward:.3f}", style=_reward_style(avg_reward))
    out.append("   min ", style="bold")
    out.append(f"{min(values):.3f}", style=_reward_style(min(values)))
    out.append("   max ", style="bold")
    out.append(f"{max(values):.3f}", style=_reward_style(max(values)))
    out.append("\n")

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
    for label, count, style in bucket_counts:
        bar_width = max(1, round((count / peak_count) * 18)) if count else 1
        bar = ("#" * bar_width) if count else "."
        out.append(label.rjust(8), style="dim")
        out.append("  ")
        out.append(bar.ljust(18), style=style)
        out.append(f" {count}\n", style="dim")


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
    line_index: int
    pattern: str


def _stylize_matches(text: Text, pattern: re.Pattern, style: str) -> Text:
    plain = text.plain
    for match in pattern.finditer(plain):
        text.stylize(style, match.start(), match.end())
    return text


def _sorted_runs(runs: List[RunInfo]) -> List[RunInfo]:
    return sorted(
        runs,
        key=lambda run: (
            run.load_metadata().get("date", ""),
            run.load_metadata().get("time", ""),
            run.run_id,
        ),
    )


def _format_run_datetime(meta: Dict[str, Any]) -> str:
    return f"{meta.get('date', '')} {meta.get('time', '')}".strip()


def _build_env_tree_label(env_id: str, models: Dict[str, List[RunInfo]]) -> Text:
    total_runs = sum(len(runs) for runs in models.values())
    label = Text()
    label.append(env_id, style="bold")
    label.append("  ")
    label.append(f"{len(models)} models", style="dim")
    label.append("  ")
    label.append(f"{total_runs} runs", style="dim")
    return label


def _build_model_tree_label(model: str, runs: List[RunInfo]) -> Text:
    label = Text()
    label.append(model, style="bold")
    label.append("  ")
    label.append(f"{len(runs)} runs", style="dim")
    return label


def _build_run_tree_label(run: RunInfo) -> Text:
    meta = run.load_metadata()
    label = Text()
    label.append(run.run_id, style="bold")
    datetime_str = _format_run_datetime(meta)
    if datetime_str:
        label.append("  ")
        label.append(datetime_str, style="dim")
    avg_reward = meta.get("avg_reward")
    if avg_reward is not None:
        label.append("  ")
        label.append("reward ", style="dim")
        label.append(_format_reward_value(avg_reward), style=_reward_style(avg_reward))
    return label


# ----------------------------
# Screens
# ----------------------------
class BrowseRunsScreen(Screen):
    """Single-screen browser for environments, models, and runs."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
    ]

    def __init__(self, index: Dict[str, Dict[str, List[RunInfo]]]):
        super().__init__()
        self.index = index
        self._run_reward_cache: Dict[Path, List[float]] = {}

    def compose(self) -> ComposeResult:
        with Container():
            with Horizontal(classes="browser-columns"):
                yield Panel(
                    Label(Text("Eval Browser", style="bold"), classes="title"),
                    Label(
                        Text("Enter opens runs  Space toggles folders"),
                        classes="subtitle",
                    ),
                    Tree("Completed evals", id="run-browser-tree"),
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

        details_widget = self.query_one("#run-browser-details", Static)
        first_run_node = self._populate_tree(tree)
        tree.focus()

        if first_run_node is not None:
            self.call_after_refresh(
                lambda: self._select_initial_run_node(first_run_node)
            )
        else:
            details_widget.update(Text("No completed evals found", style="dim"))

    def action_focus_next_pane(self) -> None:
        self.focus_next()

    def action_focus_prev_pane(self) -> None:
        self.focus_previous()

    def _select_initial_run_node(self, node: Any) -> None:
        tree = self.query_one("#run-browser-tree", Tree)
        tree.move_cursor(node)
        self._update_details_for_node(node)

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
            env_node = root.add(
                _build_env_tree_label(env_id, models),
                data=BrowserNodeData(kind="env", env_id=env_id),
                expand=env_idx == 0,
            )
            for model_idx, model in enumerate(sorted(models.keys())):
                runs = _sorted_runs(models[model])
                model_node = env_node.add(
                    _build_model_tree_label(model, runs),
                    data=BrowserNodeData(kind="model", env_id=env_id, model=model),
                    expand=env_idx == 0 and model_idx == 0,
                )
                for run in runs:
                    run_node = model_node.add(
                        _build_run_tree_label(run),
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
        self._update_details_for_node(event.node)

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

    def _update_details_for_node(self, node: Any) -> None:
        details_widget = self.query_one("#run-browser-details", Static)
        payload = getattr(node, "data", None)
        if not isinstance(payload, BrowserNodeData):
            details_widget.update(Text("Select a run to see details", style="dim"))
            return

        if payload.kind == "run" and payload.run is not None:
            details_widget.update(self._build_run_details(payload.run))
            return

        if payload.kind == "env":
            details_widget.update(self._build_env_details(payload.env_id))
            return

        if payload.kind == "model":
            details_widget.update(
                self._build_model_details(payload.env_id, payload.model)
            )
            return

        details_widget.update(Text("Select a run to see details", style="dim"))

    def _run_rewards(self, run: RunInfo) -> List[float]:
        cached = self._run_reward_cache.get(run.path)
        if cached is not None:
            return cached

        rewards: List[float] = []
        records = load_run_results(run)
        try:
            for idx in range(len(records)):
                reward = _numeric_reward(records[idx].get("reward"))
                if reward is not None:
                    rewards.append(reward)
        finally:
            records.close()

        self._run_reward_cache[run.path] = rewards
        return rewards

    def _build_env_details(self, env_id: str) -> Text:
        models = self.index.get(env_id, {})
        runs = [run for model_runs in models.values() for run in model_runs]
        rewards = _run_average_rewards(runs)

        out = Text()
        out.append("Environment\n", style="bold dim")
        out.append(env_id, style="bold")
        out.append("\n")
        out.append(f"{len(models)} models   {len(runs)} runs", style="dim")
        _append_reward_histogram(out, rewards, "Run avg rewards")

        if models:
            ranked_models = sorted(
                models.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:4]
            out.append("\n\nModel activity\n", style="bold dim")
            for model, model_runs in ranked_models:
                model_rewards = _run_average_rewards(model_runs)
                out.append(model, style="bold")
                out.append(f"  {len(model_runs)} runs", style="dim")
                if model_rewards:
                    avg_reward = sum(model_rewards) / len(model_rewards)
                    out.append("  avg ", style="dim")
                    out.append(
                        f"{avg_reward:.3f}",
                        style=_reward_style(avg_reward),
                    )
                out.append("\n")

        return out

    def _build_model_details(self, env_id: str, model: str) -> Text:
        runs = _sorted_runs(self.index.get(env_id, {}).get(model, []))
        rewards = _run_average_rewards(runs)

        out = Text()
        out.append("Model\n", style="bold dim")
        out.append(model, style="bold")
        out.append("\n")
        out.append(f"{env_id}   {len(runs)} runs", style="dim")
        _append_reward_histogram(out, rewards, "Run avg rewards")

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
            out.append("\n\nRecent runs\n", style="bold dim")
            self._append_run_row(out, "latest", latest)
            self._append_run_row(out, "best", best)

        return out

    def _build_run_details(self, run: RunInfo) -> Text:
        meta = run.load_metadata()
        rewards = self._run_rewards(run)

        out = Text()
        out.append("Run\n", style="bold dim")
        out.append(run.run_id, style="bold")
        out.append("\n")
        out.append(f"{run.env_id}   {run.model}", style="dim")

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
            out.append("\n\n")
            for idx, (label, value, style) in enumerate(summary_parts):
                if idx:
                    out.append("   ")
                out.append(f"{label} ", style="bold")
                out.append(value, style=style or "")

        _append_reward_histogram(out, rewards, "Rollout rewards")

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
        if pass_rates:
            out.append("\n\nPass rates\n", style="bold dim")
            for idx, (label, value) in enumerate(pass_rates[:6]):
                if idx and idx % 3 == 0:
                    out.append("\n")
                elif idx:
                    out.append("   ")
                out.append(f"{label} ", style="bold")
                out.append(f"{value:.3f}", style=_reward_style(value))

        return out

    def _append_run_row(self, out: Text, label: str, run: RunInfo) -> None:
        reward = _numeric_reward(run.load_metadata().get("avg_reward"))
        out.append(label, style="bold")
        out.append("  ")
        out.append(run.run_id)
        if reward is not None:
            out.append("  reward ", style="dim")
            out.append(f"{reward:.3f}", style=_reward_style(reward))
        out.append("\n")


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
    ]

    def __init__(self, run: RunInfo):
        super().__init__()
        self.run = run
        self.records = load_run_results(run)
        self.current_record_idx = 0
        self._prompt_lines: List[str] = []
        self._completion_lines: List[str] = []
        self._prompt_text: str = ""
        self._completion_text: str = ""
        self._highlight_regex: Optional[re.Pattern] = None
        self._highlight_column: Optional[str] = None
        self._highlight_timer = None
        self._completion_sections_version = 0
        self._focus_after_completion_rebuild = False
        if self.records:
            self._set_record_text_state(self.records[self.current_record_idx])

    def compose(self) -> ComposeResult:
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
                    yield VerticalScroll(
                        *self._completion_section_widgets_for_current_record(),
                        id="completion-scroll",
                    )
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
        progress.append(f"{self.current_record_idx + 1}/{len(self.records)}")
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
            return Text("No completion events", style="dim")

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
        return Text.assemble(
            (f"{len(groups)} events", "bold"),
            ("  ", ""),
            (f"{tool_groups} tool exchanges", "dim"),
            ("  ", ""),
            (f"{user_messages} user turns", "dim"),
            ("  ", ""),
            ("Enter toggles", "dim"),
            ("  ", ""),
            ("PgUp/PgDn scroll", "dim"),
        )

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
            return Text("Run metrics unavailable", style="dim")

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

    def _build_header_reward_text(self, record: Dict[str, Any]) -> Text:
        reward = record.get("reward")
        out = Text()
        out.append("Current Reward\n", style="bold dim")
        out.append(_format_reward_value(reward), style=_reward_style(reward))

        breakdown = self._extract_reward_metrics(record)
        if breakdown:
            out.append("\n")
            for idx, (name, value) in enumerate(breakdown[:3]):
                if idx:
                    out.append("   ")
                out.append(f"{name} ", style="bold")
                out.append(_format_reward_value(value), style=_reward_style(value))
        return out

    def on_mount(self) -> None:
        self._populate_rollout_list()
        self.query_one("#rollout-list", OptionList).focus()
        self.update_display(rebuild_sections=False)
        self._update_responsive_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(event.size.width)

    def on_unmount(self) -> None:
        self.records.close()

    def _populate_rollout_list(self) -> None:
        rollout_list = self.query_one("#rollout-list", OptionList)
        rollout_list.clear_options()

        if not self.records:
            rollout_list.add_option(
                Option("No rollouts in this run", id="__empty__", disabled=True)
            )
            return

        for idx in range(len(self.records)):
            rollout_list.add_option(self._build_rollout_option(idx))

        rollout_list.highlighted = self.current_record_idx
        rollout_list.scroll_to_highlight()

    def _build_rollout_option(self, idx: int) -> Option:
        record = self.records[idx]
        preview = self._record_preview(record)
        reward = record.get("reward")
        reward_text = _format_reward_value(reward)
        label = Text()
        label.append(f"#{idx}", style="bold")
        label.append("  ")
        label.append("reward ", style="dim")
        label.append(reward_text, style=_reward_style(reward))
        label.append("\n")
        label.append(
            _truncate_preview(preview or "No completion yet", 38),
            style="dim",
        )
        return Option(label, id=str(idx))

    def _record_preview(self, record: Dict[str, Any]) -> str:
        completion = record.get("completion")
        if isinstance(completion, list):
            idx = len(completion) - 1
            while idx >= 0:
                message = completion[idx]
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    idx -= 1
                    continue
                tool_calls = _parse_tool_calls(message.get("tool_calls"))
                if tool_calls:
                    tool_outputs: List[Any] = []
                    next_idx = idx + 1
                    while next_idx < len(completion):
                        next_message = completion[next_idx]
                        if (
                            not isinstance(next_message, dict)
                            or next_message.get("role") != "tool"
                        ):
                            break
                        tool_outputs.append(next_message)
                        next_idx += 1
                    preview = _tool_group_preview(message, tool_outputs)
                else:
                    preview = _format_message_preview(message, "")
                if preview:
                    return preview
                idx -= 1
            for message in reversed(completion):
                preview = _format_message_preview(message, "")
                if preview:
                    return preview
        if completion not in (None, ""):
            return _truncate_preview(str(completion), 56)

        prompt = record.get("prompt")
        if isinstance(prompt, list) and prompt:
            return _format_message_preview(prompt[-1], "Prompt only")
        if prompt not in (None, ""):
            return _truncate_preview(str(prompt), 56)
        return "Empty rollout"

    def _set_record_text_state(self, record: Dict[str, Any]) -> None:
        prompt_text = format_prompt_or_completion(record.get("prompt", ""))
        completion_text = format_prompt_or_completion(record.get("completion", ""))

        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")

        self._prompt_text = prompt_text.plain
        self._completion_text = completion_text.plain
        self._prompt_lines = prompt_text.plain.split("\n")
        self._completion_lines = completion_text.plain.split("\n")

    def update_display(self, *, rebuild_sections: bool = True) -> None:
        if not self.records:
            return

        record = self.records[self.current_record_idx]
        self._set_record_text_state(record)

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
        self.query_one("#task-content", Static).update(self._build_task_text(record))
        self.query_one("#score-content", Static).update(self._build_score_text(record))
        self.query_one("#usage-content", Static).update(self._build_usage_text(record))
        self.query_one("#info-content", Static).update(self._build_info_text(record))
        self._update_rollout_summary(record)
        if rebuild_sections:
            self._build_completion_sections(record)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_prev_record(self) -> None:
        self._move_record_cursor(-1)

    def action_next_record(self) -> None:
        self._move_record_cursor(1)

    def _move_record_cursor(self, delta: int) -> None:
        if self.records:
            new_index = (self.current_record_idx + delta) % len(self.records)
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
        self.app.push_screen(
            CopyScreen(self._prompt_text, self._completion_text, "completion")
        )

    def action_expand_all(self) -> None:
        for section in self._completion_sections():
            section.collapsed = False
        self._focus_primary_content()

    def action_collapse_all(self) -> None:
        for section in self._completion_sections():
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
        self._highlight_timer = self.set_timer(3.0, self._clear_highlight)

    def _clear_highlight(self) -> None:
        if not self.is_mounted:
            return
        self._highlight_regex = None
        self._highlight_column = None
        self.update_display()

    def _update_rollout_summary(self, record: Dict[str, Any]) -> None:
        summary = self.query_one("#rollout-summary", Label)
        summary.update(
            Text.assemble(
                (f"{self.current_record_idx + 1}/{len(self.records)}", "bold"),
                ("  ", ""),
                ("reward ", "dim"),
                (
                    _format_reward_value(record.get("reward")),
                    _reward_style(record.get("reward")),
                ),
            )
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
        if not (0 <= index < len(self.records)):
            return
        self.current_record_idx = index
        self._clear_search_state()
        self._focus_after_completion_rebuild = focus_history
        self.update_display()
        self.query_one("#completion-scroll", VerticalScroll).scroll_y = 0
        for scroll in self.query(".details-scroll"):
            scroll.scroll_y = 0

    def _clear_search_state(self) -> None:
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
            self._highlight_timer = None
        self._highlight_regex = None
        self._highlight_column = None

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

    def _completion_section_widgets_for_current_record(self) -> List[Collapsible]:
        if not self.records:
            return []
        return self._completion_section_widgets(self.records[self.current_record_idx])

    def _completion_section_widgets(self, record: Dict[str, Any]) -> List[Collapsible]:
        sections: List[Any] = []
        sections.append(
            self._build_simple_section(
                title="Initial Prompt",
                body=self._prompt_text or "No prompt context",
                column="prompt",
                collapsed=True,
                section_id="prompt-context",
                classes="history-section prompt-section",
            )
        )

        completion = record.get("completion")
        if isinstance(completion, list) and completion:
            groups = self._history_groups(completion)
            for idx, group in enumerate(groups, start=1):
                sections.append(
                    self._build_group_section(
                        idx,
                        collapsed=True,
                        group=group,
                    )
                )
        else:
            sections.append(
                self._build_simple_section(
                    title="Completion",
                    body=self._completion_text or "No completion",
                    column="completion",
                    collapsed=False,
                    section_id="completion-empty",
                    classes="history-section assistant-section",
                )
            )

        return sections

    def _build_completion_sections(self, record: Dict[str, Any]) -> None:
        self._completion_sections_version += 1
        version = self._completion_sections_version
        sections = self._completion_section_widgets(record)
        self._rebuild_completion_sections(version, sections)

    @work(group="completion-sections", exclusive=True)
    async def _rebuild_completion_sections(
        self, version: int, sections: List[Collapsible]
    ) -> None:
        if not self.is_mounted or version != self._completion_sections_version:
            return
        container = self.query_one("#completion-scroll", VerticalScroll)
        async with container.batch():
            await container.remove_children()
            if not self.is_mounted or version != self._completion_sections_version:
                return
            await container.mount(*sections)
        if self._focus_after_completion_rebuild:
            self._focus_after_completion_rebuild = False
            self.call_after_refresh(self._focus_primary_content)

    def _history_groups(self, completion: List[Any]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(completion):
            message = completion[idx]
            if isinstance(message, dict) and message.get("role") == "assistant":
                tool_calls = _parse_tool_calls(message.get("tool_calls"))
                if tool_calls:
                    tool_outputs: List[Any] = []
                    next_idx = idx + 1
                    while next_idx < len(completion):
                        next_message = completion[next_idx]
                        if (
                            not isinstance(next_message, dict)
                            or next_message.get("role") != "tool"
                        ):
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

    def _build_group_section(
        self,
        idx: int,
        *,
        group: Dict[str, Any],
        collapsed: bool,
    ) -> Collapsible:
        if group.get("kind") != "assistant-tools":
            return self._build_message_section(
                idx,
                group.get("message"),
                column="completion",
                collapsed=collapsed,
            )

        message = group.get("message")
        tool_calls = group.get("tool_calls", [])
        tool_outputs = group.get("tool_outputs", [])

        preview = _tool_group_preview(message, tool_outputs) or _format_message_preview(
            message,
            f"{len(tool_calls)} tool call(s)",
        )
        title = f"{idx}. assistant"
        if preview:
            title += f"  {preview}"

        content = ""
        if isinstance(message, dict):
            content = _stringify_message_content(message.get("content", "")).strip()

        body = content

        group_collapsed = self._assistant_tool_group_collapsed(
            collapsed=collapsed,
            body=body,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
        )

        nested_sections = self._build_tool_exchange_sections(
            turn_idx=idx,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            parent_collapsed=group_collapsed,
        )

        return self._build_section(
            title=title,
            body=body,
            column="completion",
            collapsed=group_collapsed,
            section_id=f"turn-{idx}",
            classes="history-section assistant-section",
            nested_sections=nested_sections,
        )

    def _build_message_section(
        self, idx: int, message: Any, *, column: str, collapsed: bool
    ) -> Collapsible:
        if not isinstance(message, dict):
            return self._build_simple_section(
                title=f"{idx}. item",
                body=str(message),
                column=column,
                collapsed=collapsed,
                section_id=f"turn-{idx}",
                classes="history-section assistant-section",
            )

        role = str(message.get("role", "message"))
        preview = _format_message_preview(message, "empty")
        title = f"{idx}. {role}"
        if preview:
            title += f"  {preview}"

        content = _stringify_message_content(message.get("content", "")).strip()
        body = content or "No text content"

        classes = "history-section assistant-section"
        if role == "tool":
            classes = "history-section tool-section"
        elif role not in ("assistant", "tool"):
            classes = "history-section prompt-section"

        return self._build_section(
            title=title,
            body=body,
            column=column,
            collapsed=self._section_collapsed(collapsed, body, column),
            section_id=f"turn-{idx}",
            classes=classes,
        )

    def _assistant_tool_group_collapsed(
        self,
        *,
        collapsed: bool,
        body: str,
        tool_calls: List[Any],
        tool_outputs: List[Any],
    ) -> bool:
        if not self._highlight_regex or self._highlight_column != "completion":
            return collapsed
        if body and self._highlight_regex.search(body):
            return False
        for tool_call in tool_calls:
            name, arguments, _ = _tool_call_parts(tool_call)
            if self._highlight_regex.search(name) or self._highlight_regex.search(
                arguments
            ):
                return False
        for output in tool_outputs:
            output_text = (
                _stringify_message_content(output.get("content", ""))
                if isinstance(output, dict)
                else str(output)
            )
            if self._highlight_regex.search(output_text):
                return False
        return collapsed

    def _build_tool_exchange_sections(
        self,
        *,
        turn_idx: int,
        tool_calls: List[Any],
        tool_outputs: List[Any],
        parent_collapsed: bool,
    ) -> List[Collapsible]:
        sections: List[Collapsible] = []
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

            sections.append(
                self._build_tool_exchange_section(
                    turn_idx=turn_idx,
                    tool_idx=tool_idx,
                    tool_name=name,
                    arguments=arguments,
                    output_message=matched_output,
                    collapsed=parent_collapsed or tool_idx > 1,
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
            sections.append(
                self._build_section(
                    title=f"tool output {len(sections) + 1}  {_tool_output_preview(output_message)}",
                    body=output_text or "No tool output recorded",
                    column="completion",
                    collapsed=self._section_collapsed(True, output_text, "completion"),
                    section_id=f"turn-{turn_idx}-tool-extra-{extra_idx}",
                    classes="history-section tool-section nested-section",
                )
            )

        return sections

    def _build_tool_exchange_section(
        self,
        turn_idx: int,
        tool_idx: int,
        tool_name: str,
        arguments: str,
        output_message: Any,
        *,
        collapsed: bool,
    ) -> Collapsible:
        output_text = ""
        if output_message is not None:
            if isinstance(output_message, dict):
                output_text = _stringify_message_content(
                    output_message.get("content", "")
                )
            else:
                output_text = str(output_message)

        output_preview = (
            _tool_output_preview(output_message) if output_message else "No output"
        )
        title = f"tool {tool_idx}  {tool_name}  -> {output_preview}"
        body = "\n".join(
            [
                "Call",
                arguments,
                "",
                "Output",
                output_text or "No tool output recorded",
            ]
        )

        return self._build_section(
            title=title,
            body=body,
            column="completion",
            collapsed=self._section_collapsed(collapsed, body, "completion"),
            section_id=f"turn-{turn_idx}-tool-{tool_idx}",
            classes="history-section tool-call-section nested-section",
        )

    def _build_simple_section(
        self,
        *,
        title: str,
        body: str,
        column: str,
        collapsed: bool,
        section_id: str,
        classes: str,
    ) -> Collapsible:
        return self._build_section(
            title=title,
            body=body,
            column=column,
            collapsed=self._section_collapsed(collapsed, body, column),
            section_id=section_id,
            classes=classes,
        )

    def _build_section(
        self,
        *,
        title: str,
        body: str,
        column: str,
        collapsed: bool,
        section_id: str,
        classes: str,
        nested_sections: Optional[List[Any]] = None,
    ) -> Collapsible:
        children: List[Any] = []
        if body or not nested_sections:
            content = Static(
                self._render_section_body(body, column),
                id=f"{section_id}-body",
                classes="section-body",
                markup=False,
            )
            children.append(content)
        children.extend(nested_sections or [])
        return Collapsible(
            *children,
            title=title,
            collapsed=collapsed,
            id=section_id,
            classes=classes,
        )

    def _render_section_body(self, body: str, column: str) -> Text:
        text = Text(body or "No content")
        if self._highlight_regex and self._highlight_column == column:
            _stylize_matches(text, self._highlight_regex, "reverse")
        return text

    def _section_collapsed(self, default: bool, body: str, column: str) -> bool:
        if self._highlight_regex and self._highlight_column == column:
            if self._highlight_regex.search(body):
                return False
        return default

    def _completion_sections(self) -> List[Collapsible]:
        container = self.query_one("#completion-scroll", VerticalScroll)
        return list(container.query(Collapsible))

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
            0.10,
            lambda: self._shift_expand_pulse(collapsible),
        )
        self.set_timer(
            0.22,
            lambda: self._clear_expand_pulse(collapsible),
        )
        collapsible.call_after_refresh(
            lambda: collapsible.scroll_visible(duration=0.18, easing="out_cubic")
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
        reward = record.get("reward")
        out = Text()
        out.append("Reward\n", style="bold dim")
        out.append(_format_reward_value(reward), style=_reward_style(reward))

        metrics = self._extract_reward_metrics(record)
        if metrics:
            out.append("\n\nBreakdown\n", style="bold dim")
            width = max(len(name) for name, _ in metrics)
            for name, value in metrics:
                out.append(name.ljust(width), style="bold")
                out.append("  ")
                out.append(_format_reward_value(value), style=_reward_style(value))
                out.append("\n")

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
        metric_values: Dict[str, Any] = {}
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values[key] = value

        info = _coerce_info_value(record.get("info"))
        if isinstance(info, dict):
            reward_signals = info.get("reward_signals")
            if isinstance(reward_signals, dict):
                for key, value in reward_signals.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_values.setdefault(key, value)

        standard_fields = {
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
        for key, value in record.items():
            if key in standard_fields:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values.setdefault(key, value)

        return sorted(metric_values.items())

    def _build_task_text(self, record: Dict[str, Any]) -> Text:
        out = Text()
        self._append_context_section(out, "Task", record.get("task"))
        self._append_context_section(out, "Answer", record.get("answer"))
        self._append_context_section(
            out, "Stop condition", record.get("stop_condition")
        )

        if not out.plain.strip():
            return Text("No task details for this rollout", style="dim")
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

        if not out.plain.strip():
            return Text("No usage metrics for this rollout", style="dim")
        return out

    def _build_info_text(self, record: Dict[str, Any]) -> Text:
        out = Text()
        error = record.get("error")
        if error not in (None, ""):
            self._append_context_section(out, "Error", error)

        info = record.get("info")
        if info not in (None, {}, ""):
            self._append_context_section(out, "Info", format_info_for_details(info))

        if not out.plain.strip():
            return Text("No info payload for this rollout", style="dim")
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

    .copy-textarea {
        height: 1fr;
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
        self._prompt_lines = prompt_lines
        self._completion_lines = completion_lines
        self._prompt_hits: List[SearchHit] = []
        self._completion_hits: List[SearchHit] = []
        self._active_column: Optional[str] = None
        self._prompt_cursor: Optional[int] = None
        self._completion_cursor: Optional[int] = None

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
            prompt_list.scroll_to_highlight()
        elif (
            self._active_column == "completion" and self._completion_cursor is not None
        ):
            completion_list.highlighted = self._completion_cursor
            prompt_list.highlighted = None
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


class CopyScreen(ModalScreen[None]):
    """Modal screen for selecting and copying prompt/completion text."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "quit", "Quit"),
        Binding("tab", "cycle_column", "Next column"),
        Binding("shift+tab", "cycle_column", show=False),
        Binding("c", "copy", "Copy"),
    ]

    def __init__(self, prompt_text: str, completion_text: str, start_column: str):
        super().__init__()
        self._prompt_text = prompt_text
        self._completion_text = completion_text
        self._active_column = (
            start_column if start_column in ("prompt", "completion") else "completion"
        )

    def compose(self) -> ComposeResult:
        with Container():
            with Panel(classes="modal-header"):
                yield Label(Text("Copy Mode", style="bold"))
                yield Label(
                    Text("q: quit", style="dim"),
                    id="copy-hint-q",
                    classes="copy-hint",
                )
                yield Label(
                    Text("Tab: switch columns", style="dim"),
                    id="copy-hint-1",
                    classes="copy-hint",
                )
                yield Label(
                    Text(
                        "Highlight text with mouse drag or Shift+Arrow",
                        style="dim",
                    ),
                    id="copy-hint-2",
                    classes="copy-hint",
                )
                yield Label("", id="copy-hint-3", classes="copy-hint")
                yield Label(
                    Text("Esc: close", style="dim"),
                    id="copy-hint-4",
                    classes="copy-hint",
                )
                yield Label("", id="copy-status", classes="subtitle")

            with Horizontal(classes="modal-columns"):
                with Panel(classes="modal-panel"):
                    yield Label(Text("Prompt", style="bold"), id="copy-prompt-label")
                    prompt_area = TextArea(
                        self._prompt_text,
                        id="copy-prompt",
                        classes="copy-textarea",
                    )
                    prompt_area.read_only = True
                    yield prompt_area
                with Panel(classes="modal-panel"):
                    yield Label(
                        Text("Completion", style="bold"), id="copy-completion-label"
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
        self._sync_focus()
        self._update_copy_hint()

    @property
    def active_bindings(self) -> dict[str, Any]:
        bindings = super().active_bindings
        ordered: dict[str, Any] = {}
        for key in ("q", "escape", "tab", "c"):
            if key in bindings:
                ordered[key] = bindings[key]
        for key, binding in bindings.items():
            if key not in ordered:
                ordered[key] = binding
        return ordered

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area is self._active_text_area():
            self._update_copy_hint()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_quit(self) -> None:
        self.app.exit()

    def on_key(self, event) -> None:
        if event.key in ("tab", "shift+tab", "backtab"):
            self.action_cycle_column()
            event.prevent_default()
            event.stop()

    def action_cycle_column(self) -> None:
        self._active_column = (
            "completion" if self._active_column == "prompt" else "prompt"
        )
        self._sync_focus()
        self._update_copy_hint()

    def action_copy(self) -> None:
        text_area = self._active_text_area()
        selected = _get_text_area_selection(text_area)
        full_text = _get_text_area_full_text(text_area)
        if not selected:
            selected = full_text
            copied_label = "full column"
        else:
            copied_label = "selection"
        if not isinstance(selected, str):
            selected = str(selected)
        if selected:
            _copy_to_clipboard(self.app, selected)
        status = self.query_one("#copy-status", Label)
        status.update(
            Text(f"Copied {copied_label} ({len(selected)} chars).", style="dim")
            if selected
            else Text("Nothing to copy.", style="dim")
        )
        self._update_copy_hint()

    def _active_text_area(self) -> TextArea:
        if self._active_column == "prompt":
            return self.query_one("#copy-prompt", TextArea)
        return self.query_one("#copy-completion", TextArea)

    def _sync_focus(self) -> None:
        prompt_label = self.query_one("#copy-prompt-label", Label)
        completion_label = self.query_one("#copy-completion-label", Label)
        if self._active_column == "prompt":
            prompt_label.update(Text("Prompt (active)", style="bold"))
            completion_label.update(Text("Completion", style="bold"))
        else:
            prompt_label.update(Text("Prompt", style="bold"))
            completion_label.update(Text("Completion (active)", style="bold"))
        self._active_text_area().focus()

    def _update_copy_hint(self) -> None:
        selected = _get_text_area_selection(self._active_text_area())
        if selected:
            count = len(selected)
            unit = "char" if count == 1 else "chars"
            copy_text = f"c: copy selection ({count} {unit})"
        elif self._active_column == "prompt":
            copy_text = "c: copy prompt"
        else:
            copy_text = "c: copy completion"
        hint = self.query_one("#copy-hint-3", Label)
        hint.update(Text(copy_text, style="dim"))


def _copy_to_clipboard(app: App, text: str) -> None:
    app.copy_to_clipboard(text)


def _get_text_area_selection(text_area: TextArea) -> str:
    return text_area.selected_text or ""


def _get_text_area_full_text(text_area: TextArea) -> str:
    return text_area.text


def main() -> None:
    # Optional args via env vars
    env_dir = os.environ.get("VF_ENV_DIR", "./environments")
    outputs_dir = os.environ.get("VF_OUTPUTS_DIR", "./outputs")
    app = VerifiersTUI(env_dir, outputs_dir)
    app.run()


if __name__ == "__main__":
    main()
