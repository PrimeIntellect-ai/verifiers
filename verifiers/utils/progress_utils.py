import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text




def _range_bar(
    value: float | None, min_v: float | None, max_v: float | None, width: int
) -> str:
    if width <= 0:
        return "||"
    if value is None or min_v is None or max_v is None:
        return "|" + (" " * width) + "|"
    if max_v == min_v:
        return "|" + (":" * width) + "|"
    ratio = (value - min_v) / (max_v - min_v)
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "|" + (":" * filled) + (" " * (width - filled)) + "|"


def _progress_bar(completed: int, total: int, width: int) -> str:
    if width <= 0:
        return ""
    if total <= 0:
        return "[" + ("?" * width) + "]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(ratio * width)
    if filled <= 0:
        bar = "-" * width
    elif filled >= width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + "-" * (width - filled)
    return "[" + bar + "]"


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds // 60
    seconds = seconds % 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    if hours < 24:
        return f"{hours:.0f}h {minutes:.0f}m"
    days = hours // 24
    hours = hours % 24
    return f"{days:.0f}d {hours:.0f}h"


@dataclass
class RollingMetric:
    history_size: int
    total: float = 0.0
    count: int = 0
    history: list[float] = field(default_factory=list)
    min_avg: float | None = None
    max_avg: float | None = None

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1
        avg = self.total / self.count
        self.history.append(avg)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]
        if self.min_avg is None or avg < self.min_avg:
            self.min_avg = avg
        if self.max_avg is None or avg > self.max_avg:
            self.max_avg = avg

    @property
    def avg(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


def _select_scale(name: str, metric: RollingMetric) -> tuple[float | None, float | None]:
    avg = metric.avg
    if avg is None:
        return None, None
    min_v = metric.min_avg
    max_v = metric.max_avg
    if min_v is None or max_v is None:
        return None, None
    if 0.0 <= min_v and max_v <= 1.0:
        return 0.0, 1.0
    lowered = name.lower()
    if any(token in lowered for token in ("num", "count", "calls", "steps", "tokens")):
        upper = max_v if max_v > 0 else 1.0
        return 0.0, upper
    return min_v, max_v


def _format_metric_value(value: float | None) -> str:
    if value is None:
        return "?"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    abs_val = abs(value)
    if abs_val >= 100:
        return f"{value:.0f}"
    if abs_val >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _metrics_table(
    pairs: list[tuple[str, str]],
    width: int,
    max_cols: int | None = None,
    min_leader: int = 4,
    min_col_width: int = 24,
) -> Table | None:
    if not pairs:
        return None

    max_metric = max(len(metric) for metric, _ in pairs)
    max_value = max(len(value) for _, value in pairs)
    base_width = max_metric + min_leader + 2 + max_value
    max_allowed = max_cols if max_cols is not None else width
    min_width_cols = max(1, width // max(min_col_width, 1))
    col_guess = max(1, min(max_allowed, max(1, width // max(base_width, 1))))
    col_guess = min(col_guess, min_width_cols)

    cols = col_guess
    rows = (len(pairs) + cols - 1) // cols
    by_col: list[list[tuple[str, str]]] = [[] for _ in range(cols)]
    for idx, pair in enumerate(pairs):
        by_col[idx % cols].append(pair)

    col_specs: list[tuple[int, int]] = []
    for col_pairs in by_col:
        if col_pairs:
            col_metric = max(len(metric) for metric, _ in col_pairs)
            col_value = max(len(value) for _, value in col_pairs)
        else:
            col_metric = max_metric
            col_value = max_value
        col_specs.append((col_metric, col_value))

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    for _ in range(cols):
        table.add_column(no_wrap=True)

    for row in range(rows):
        cells = []
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(pairs):
                cells.append("")
                continue
            metric, value = pairs[idx]
            col_metric, col_value = col_specs[col]
            value_start = col_metric + min_leader + 2
            dots_len = max(min_leader, value_start - len(metric) - 2)
            dots = "." * dots_len
            cell = f"{metric} {dots} {value}"
            cells.append(cell)
        table.add_row(*cells)
    return table


@dataclass
class EnvProgressState:
    name: str
    total: int
    completed: int = 0
    metrics: dict[str, RollingMetric] = field(default_factory=dict)
    metric_order: list[str] = field(default_factory=list)


class RolloutProgress:
    def __init__(
        self,
        total: int,
        desc: str,
        spark_width: int = 18,
        history_size: int = 60,
        bar_width: int = 24,
        metric_names: list[str] | None = None,
        console: Console | None = None,
    ) -> None:
        self.total = total
        self.desc = desc
        self.spark_width = spark_width
        self.history_size = max(history_size, total) if total > 0 else history_size
        self.bar_width = bar_width
        self.completed = 0
        self.start_time = time.time()
        self.metrics: dict[str, RollingMetric] = {}
        self.metric_order: list[str] = []
        self._init_metrics(metric_names)
        self.console = console or Console(stderr=True)
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )

    def _init_metrics(self, metric_names: list[str] | None) -> None:
        self.metrics = {"reward": RollingMetric(history_size=self.history_size)}
        self.metric_order = ["reward"]
        if metric_names:
            for name in metric_names:
                if name == "reward" or name in self.metrics:
                    continue
                self.metrics[name] = RollingMetric(history_size=self.history_size)
                self.metric_order.append(name)

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    def update(self, states: Iterable[dict]) -> None:
        self.completed += 1
        for state in states:
            reward = state.get("reward")
            if reward is not None:
                try:
                    self.metrics["reward"].add(float(reward))
                except (TypeError, ValueError):
                    pass
            metrics = state.get("metrics") or {}
            for name, value in metrics.items():
                if value is None:
                    continue
                metric = self.metrics.get(name)
                if metric is None:
                    metric = RollingMetric(history_size=self.history_size)
                    self.metrics[name] = metric
                    self.metric_order.append(name)
                try:
                    metric.add(float(value))
                except (TypeError, ValueError):
                    continue
        self._live.update(self._render(), refresh=True)

    def _render(self) -> Group:
        elapsed = time.time() - self.start_time
        reward_avg = self.metrics["reward"].avg
        postfix = None
        if reward_avg is not None:
            postfix = f"reward={reward_avg:.3f}"

        try:
            from tqdm import tqdm

            ncols = max(self.console.size.width - 1, 20)
            progress_text = tqdm.format_meter(
                self.completed,
                self.total,
                elapsed,
                ncols=ncols,
                prefix=self.desc,
                unit="it",
                postfix=postfix,
            )
        except Exception:
            progress_text = (
                f"{self.desc}: {self.completed}/{self.total} "
                f"{_progress_bar(self.completed, self.total, self.bar_width)}"
            )
            if elapsed >= 0:
                progress_text += f" elapsed {_format_seconds(elapsed)}"
            if self.completed > 0 and self.total > 0:
                eta = max((elapsed / self.completed) * (self.total - self.completed), 0.0)
                progress_text += f" eta {_format_seconds(eta)}"

        progress = Text(progress_text, style="bold")
        ncols = max(self.console.size.width - 1, 20)
        pairs: list[tuple[str, str]] = []
        for metric_name in self.metric_order:
            metric = self.metrics[metric_name]
            pairs.append((metric_name, _format_metric_value(metric.avg)))
        table = _metrics_table(pairs, width=ncols)

        return Group(progress, table) if table is not None else Group(progress)


class MultiEnvProgress:
    def __init__(
        self,
        env_totals: dict[str, int],
        metric_names_by_env: dict[str, list[str]] | None = None,
        console: Console | None = None,
    ) -> None:
        self.env_order = list(env_totals.keys())
        self.envs: dict[str, EnvProgressState] = {}
        self.start_time = time.time()
        self.console = console or Console(stderr=True)
        for name in self.env_order:
            total = env_totals[name]
            env_state = EnvProgressState(name=name, total=total)
            metric_names = (metric_names_by_env or {}).get(name)
            env_state.metrics = {"reward": RollingMetric(history_size=max(60, total))}
            env_state.metric_order = ["reward"]
            if metric_names:
                for metric_name in metric_names:
                    if metric_name == "reward" or metric_name in env_state.metrics:
                        continue
                    env_state.metrics[metric_name] = RollingMetric(
                        history_size=max(60, total)
                    )
                    env_state.metric_order.append(metric_name)
            self.envs[name] = env_state

        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    def update(self, states: Iterable[dict]) -> None:
        states_list = list(states)
        if not states_list:
            return
        task = states_list[0].get("task", "default")
        env_state = self.envs.get(task)
        if env_state is None:
            return
        env_state.completed += 1
        for state in states_list:
            reward = state.get("reward")
            if reward is not None:
                try:
                    env_state.metrics["reward"].add(float(reward))
                except (TypeError, ValueError):
                    pass
            metrics = state.get("metrics") or {}
            for name, value in metrics.items():
                if value is None:
                    continue
                metric = env_state.metrics.get(name)
                if metric is None:
                    metric = RollingMetric(history_size=max(60, env_state.total))
                    env_state.metrics[name] = metric
                    env_state.metric_order.append(name)
                try:
                    metric.add(float(value))
                except (TypeError, ValueError):
                    continue
        self._live.update(self._render(), refresh=True)

    def _render(self) -> Group:
        elapsed = time.time() - self.start_time
        ncols = max(self.console.size.width - 1, 20)
        renderables: list[Text] = []

        for idx, name in enumerate(self.env_order):
            env_state = self.envs[name]
            reward_avg = env_state.metrics.get("reward").avg if env_state.metrics else None
            postfix = None
            if reward_avg is not None:
                postfix = f"reward={reward_avg:.3f}"
            try:
                from tqdm import tqdm

                progress_text = tqdm.format_meter(
                    env_state.completed,
                    env_state.total,
                    elapsed,
                    ncols=ncols,
                    prefix=name,
                    unit="it",
                    postfix=postfix,
                )
            except Exception:
                progress_text = (
                    f"{name}: {env_state.completed}/{env_state.total} "
                    f"{_progress_bar(env_state.completed, env_state.total, 24)}"
                )
            renderables.append(Text(progress_text, style="bold"))

            pairs: list[tuple[str, str]] = []
            for metric_name in env_state.metric_order:
                metric = env_state.metrics[metric_name]
                pairs.append((metric_name, _format_metric_value(metric.avg)))
            table = _metrics_table(pairs, width=ncols)
            if table is not None:
                renderables.append(table)
            if idx < len(self.env_order) - 1:
                renderables.append(Text(""))

        return Group(*renderables)


def can_render_rich_progress() -> bool:
    return sys.stderr.isatty()
