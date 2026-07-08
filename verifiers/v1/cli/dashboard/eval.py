"""The eval `--rich` dashboard: a config overview, a progress bar, and one line per rollout.

Reads each `Rollout.trace`/`phase` every tick — no extra plumbing. Each row carries a bracketed
phase marker that reads at a glance — `[pending]` (dim), `[setup]` (yellow), `[rollout]` (cyan),
`[finalize]` (magenta), `[scoring]` (blue), `[success]` (green), `[error]` (red) — padded so the
brackets line up in a column down the left edge. The reward shows only once a rollout is fully
scored (phase DONE), so it never flips as scoring lands. A task's rollouts are grouped adjacently
and joined by a left brace (╭│╰), so an episode (a task's n rollouts) reads as a unit. Every
rollout is on screen from the start: one still queued behind the concurrency cap reads `[pending]`
(its task is all that's known yet) until it begins, and finished ones keep their result. The
overview + progress sit on top, above a rule.
"""

import contextlib
import time

from pydantic import BaseModel
from rich.console import Console, Group
from rich.markup import escape
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.cli.output import output_path
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.trace import Trace
from verifiers.v1.types import Usage
from verifiers.v1.utils.format import (
    format_count,
    format_mean,
    format_override,
    format_time,
)
from verifiers.utils.pricing_utils import format_cost_usd

# For sizing pages to the terminal: detects the real terminal height/width each access (the live
# view writes to the same terminal). Reused so we don't rebuild it every refresh tick.
_CONSOLE = Console()
_PAGE_SECONDS = 5.0  # rotate to the next page of rollouts this often when they overflow
# The under-bar breakdown pads its label column to the Overview's widest label, so the two
# `label  value` grids (above and below the progress bar) line their values up.
_LABEL_WIDTH = len("timeouts")

_STYLE = {
    "pending": "dim",
    "setup": "yellow",
    "running": "cyan",
    "finalize": "magenta",
    "scoring": "blue",
    "success": "green",
    "error": "red",
}
_MARK_LABEL = {
    "pending": "pending",
    "setup": "setup",
    "running": "rollout",
    "finalize": "finalize",
    "scoring": "scoring",
    "success": "success",
    "error": "error",
}
_MARK_WIDTH = max(len(label) for label in _MARK_LABEL.values())
# Each label padded to a common width and bracketed, so the `[ ]` line up in a column down the
# left edge (label left-aligned inside) — the current phase reads at a glance. `escape` keeps the
# brackets literal: Rich parses `[label]` in a cell as markup and would otherwise drop it.
_MARK = {
    state: escape(f"[{label:<{_MARK_WIDTH}}]") for state, label in _MARK_LABEL.items()
}


def _limits(config: EvalConfig) -> list[str]:
    """Per-rollout caps for the overview (concurrency first, then turns, tokens). An unset cap
    reads as 'no ...' rather than being hidden."""
    toks = []
    if config.max_input_tokens:
        toks.append(f"in≤{config.max_input_tokens}")
    if config.max_output_tokens:
        toks.append(f"out≤{config.max_output_tokens}")
    if config.max_total_tokens:
        toks.append(f"total≤{config.max_total_tokens}")
    return [
        f"≤{config.max_concurrent} concurrent"
        if config.max_concurrent
        else "no concurrency cap",
        f"{config.max_turns} turns" if config.max_turns else "no turn cap",
        f"{', '.join(toks)} tokens" if toks else "no token cap",
    ]


def _timeouts(config: EvalConfig) -> list[str]:
    """Per-stage rollout timeouts for the overview, each stage enumerated (unset → 'no <stage>
    timeout')."""
    return [
        f"{stage} {v:g}s"
        if (v := getattr(config.timeout, stage))
        else f"no {stage} timeout"
        for stage in ("setup", "rollout", "finalize", "scoring")
    ]


def _aligned(rows: list[list[str]]) -> list[str]:
    """Join each row's `·`-separated segments, padding shared columns to a common width so the
    separators line up across rows (each row's last segment is left ragged)."""
    widths: dict[int, int] = {}
    for row in rows:
        for i, seg in enumerate(row):
            widths[i] = max(widths.get(i, 0), len(seg))
    return [
        "  ·  ".join(
            seg.ljust(widths[i]) if i < len(row) - 1 else seg
            for i, seg in enumerate(row)
        )
        for row in rows
    ]


def _warning(config: EvalConfig) -> Text | None:
    """A local-runtime caution for a code-running harness (none for the tool-less `null`),
    shown above the overview rather than as a row in it."""
    if config.harness.id != "null" and config.harness.runtime.type == "subprocess":
        return Text(
            "warning  Runs on the local system; local files and settings may affect this "
            "evaluation. Use subprocess only for debugging, or use docker or prime for an "
            "isolated run.",
            style="yellow",
        )
    return None


def overrides(
    config: BaseModel,
    default: BaseModel | None = None,
    skip: frozenset[str] = frozenset(),
) -> list[str]:
    """`field=value` segments for the fields the *user* customized, sorted, diffed against each
    field's declared default. Not `model_fields_set`: a `--resume` run reloads its config via
    `model_validate(config.toml)`, and that toml is dumped with `exclude_none` (every field), so
    `model_fields_set` would flag them all. `default` is the reference instance, threaded through
    recursion so a pinned nested default (a taskset's `user=UserConfig(colocated=True)`) reads as
    unchanged. `skip` holds dotted paths (`harness.runtime.type`)."""
    segments: list[str] = []
    fields = type(config).model_fields
    for field in sorted(fields):
        if field in skip:
            continue
        value = getattr(config, field)
        # `get_default` returns `PydanticUndefined` for a required field, so it always reads as set.
        field_def = (
            fields[field].get_default(call_default_factory=True)
            if default is None
            else getattr(default, field, None)
        )
        if isinstance(value, BaseModel):  # nested config: flatten as `field.<sub>`
            child_skip = frozenset(
                s[len(field) + 1 :] for s in skip if s.startswith(f"{field}.")
            )
            # A switched discriminator (e.g. subprocess→docker) is an override the per-field diff
            # misses — within the new class `type` equals its own default — so surface it.
            class_changed = type(value) is not type(field_def)
            if (
                class_changed
                and "type" not in child_skip
                and (discriminator := getattr(value, "type", None)) is not None
            ):
                segments.append(f"{field}.type={format_override(discriminator)}")
            # A switched class is a new shape, so diff against its own defaults, not the instance.
            segments.extend(
                f"{field}.{seg}"
                for seg in overrides(
                    value, default=None if class_changed else field_def, skip=child_skip
                )
            )
        elif value != field_def:
            segments.append(f"{field}={format_override(value)}")
    return segments


def Overview(config: EvalConfig) -> Table:
    sampling = ", ".join(
        f"{k}={v}" for k, v in config.sampling.model_dump(exclude_none=True).items()
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row(
        "env",
        f"{config.taskset.name}  ·  {config.harness.name} harness  ·  {config.harness.runtime.type} runtime",
    )
    model = f"{config.model}  ({sampling})" if sampling else config.model
    grid.add_row("model", f"{model}  via {config.client.base_url}")
    # Non-default knobs the user set, one row each when non-empty. `escape` the cell: an override
    # value (or our `[...]`/`{...}` delimiters) can carry Rich markup that would otherwise be
    # parsed as styling and dropped. `id` is in the `env` row; harness `runtime.type` too (hidden
    # here), but only for the harness — a taskset's `user.runtime.type` has no other display.
    if taskset_over := overrides(config.taskset, skip=frozenset({"id"})):
        grid.add_row("taskset", escape("  ·  ".join(taskset_over)))
    if harness_over := overrides(
        config.harness, skip=frozenset({"id", "runtime.type"})
    ):
        grid.add_row("harness", escape("  ·  ".join(harness_over)))
    limits, timeouts = _aligned([_limits(config), _timeouts(config)])
    grid.add_row("limits", limits)
    grid.add_row("timeouts", timeouts)
    grid.add_row("output", Text(str(output_path(config)), overflow="fold"))
    return grid


def Progress(
    rollouts: list[Rollout],
    start: float,
    page: tuple[int, int] | None = None,
    finished: list[Trace] | None = None,
) -> Group:
    # On resume, `finished` holds the kept on-disk rollouts (reloaded as finished traces); count
    # them alongside this session's so progress, reward, err, and the breakdown cover the whole
    # run. `rollouts` is only this session's (owed) work, so the total adds the kept ones back.
    done = (finished or []) + [
        r.trace for r in rollouts if r.phase == Phase.DONE
    ]  # fully scored
    total = len(finished or []) + len(rollouts)
    # Headline reward = mean over non-errored; when any errored, `format_mean` appends the
    # global avg (errored count as 0) in parens. `err` is the share that errored.
    reward = format_mean(done, lambda t: t.reward)
    err = f"{sum(t.has_error for t in done) / len(done):.2f}" if done else "—"
    stats = (
        f"{len(done)}/{total} · {format_time(time.time() - start)} · "
        f"reward {reward} · err {err}"
    )
    if page is not None:  # overflowing — show which page, and that the arrows page
        stats += f"  (page {page[0]}/{page[1]} ◄ ►)"
    row = Table.grid(expand=True, padding=(0, 1))
    row.add_column(ratio=1)  # bar stretches to fill the width left of the stats
    row.add_column(justify="right", no_wrap=True)
    row.add_row(
        ProgressBar(total=total or 1, completed=len(done)),
        Text(stats),
    )
    breakdown = _breakdown(done)
    return Group(row, breakdown) if breakdown is not None else Group(row)


def _breakdown(done: list[Trace]) -> Table | None:
    """The per-component view under the headline (summed) reward: a `rewards` row of each named
    `@reward` contribution and a `metrics` row of each `@metric`, then a `usage` row (tokens and
    cost summed over completed rollouts) and a `time` row (each phase averaged over the rollouts
    that have it timed), laid out like the Overview (dim label column). Reward/metric components
    are error-corrected means, with the global mean (an errored trace's value counting as 0) in
    parens when some errored (see `format_mean`); they're skipped when every rollout errored (no
    clean mean to show), while usage/time still appear — those resources were spent regardless.
    `None` when no rollout has completed."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=_LABEL_WIDTH)
    grid.add_column()
    # rewards/metrics are error-corrected means — skip when every rollout errored (no clean mean to
    # show); usage/time below still cover errored rollouts (their resources were spent regardless).
    has_clean = any(not t.has_error for t in done)
    score_rows = (("rewards", "rewards"), ("metrics", "metrics")) if has_clean else ()
    for label, source in score_rows:
        # every key seen across traces, first-seen order (a trace records only the functions
        # that ran for it, so keys can vary)
        names: list[str] = []
        for trace in done:
            names.extend(n for n in getattr(trace, source) if n not in names)
        if not names:
            continue
        segments = []
        for name in names:
            mean = format_mean(
                done, lambda t, n=name, s=source: getattr(t, s).get(n, 0.0)
            )
            segments.append(f"{name} {mean}")
        grid.add_row(label, "  ·  ".join(segments))

    # Resource use over every completed rollout (errored ones still spent tokens/time): tokens and
    # cost are summed; each timing phase is averaged over the rollouts that have it timed (averaged
    # per phase rather than summed, since phases run concurrently across rollouts).
    total_in = total_out = total_cached = total_reasoning = 0
    total_judge_in = total_judge_out = 0
    total_cost = total_judge_cost = 0.0
    have_cost = have_cached = have_reasoning = have_judge = False
    phase_secs: dict[str, float] = {}
    phase_count: dict[str, int] = {}
    for trace in done:
        prompt, completion, cached, reasoning, _ = _tokens(trace)
        total_in += prompt
        total_out += completion
        if cached is not None:
            total_cached += cached
            have_cached = True
        if reasoning is not None:
            total_reasoning += reasoning
            have_reasoning = True
        if trace.usage is not None and trace.usage.cost is not None:
            total_cost += trace.usage.cost
            have_cost = True
        # Judge / auxiliary scoring calls (off the message graph) shown separately from the agent's.
        judge = Usage.aggregate(trace.extra_usage)
        if judge is not None:
            total_judge_in += judge.input_tokens
            total_judge_out += judge.completion_tokens
            if judge.cost is not None:
                total_judge_cost += judge.cost
            have_judge = True
        for phase in ("setup", "generation", "finalize", "scoring"):
            span = getattr(trace.timing, phase)
            if span.end:  # phase was timed for this rollout
                phase_secs[phase] = phase_secs.get(phase, 0.0) + span.duration
                phase_count[phase] = phase_count.get(phase, 0) + 1
    if (
        total_in
        or total_out
        or have_cost
        or have_cached
        or have_reasoning
        or have_judge
    ):
        tokens = f"{format_count(total_in)}/{format_count(total_out)} tokens"
        details = []
        if have_cached:
            details.append(f"{format_count(total_cached)} cached")
        if have_reasoning:
            details.append(f"{format_count(total_reasoning)} reasoning")
        if details:
            tokens += f" ({', '.join(details)})"
        usage = [tokens]
        if have_judge:
            usage.append(
                f"+ {format_count(total_judge_in)}/{format_count(total_judge_out)} judge"
            )
        if have_cost:
            cost = format_cost_usd(total_cost)
            if total_judge_cost:
                cost += f" + {format_cost_usd(total_judge_cost)} judge"
            usage.append(cost)
        grid.add_row("usage", "  ·  ".join(usage))
    time_segments = [
        f"{phase} {format_time(phase_secs[phase] / phase_count[phase])}"
        for phase in ("setup", "generation", "finalize", "scoring")
        if phase_count.get(phase)
    ]
    if time_segments:
        grid.add_row("time", "  ·  ".join(time_segments))
    return grid if grid.row_count else None


def _tokens(trace: Trace) -> tuple[int, int, int | None, int | None, int]:
    """Input/output tokens summed across all branches: per branch, output is every assistant
    (completion) token generated across its turns and input is its last turn's prompt — the full
    final context the model saw. A rollout yields one training sample per branch (a linear trace
    is a single branch; compaction and subagents add more), so the totals sum them — matching
    `Trace.num_input_tokens` / `Trace.num_output_tokens`. (Output can exceed the final context —
    reasoning tokens count toward completions but aren't re-fed — so it's not derived by
    subtraction.)

    Both counts read token ids when present and fall back to provider-reported usage when the
    endpoint returns no token ids (e.g. plain OpenAI completions), so the counts aren't shown as
    0/0. Returns the branch count from the same derived view so each dashboard tick materializes
    it once."""
    usage = trace.usage
    cached = usage.cached_input_tokens if usage else None
    reasoning = usage.reasoning_tokens if usage else None
    branches = trace.branches
    nbranches = len(branches)
    prompt = sum(b.num_input_tokens for b in branches)
    completion = sum(b.num_output_tokens for b in branches)
    return prompt, completion, cached, reasoning, nbranches


def _started(rollout: Rollout) -> float:
    # Sort key: when a rollout began (its setup start). A still-pending rollout has no trace
    # yet, so it sorts last (+inf) — behind everything already in flight, in task order.
    return (
        rollout.trace.timing.setup.start if rollout.trace is not None else float("inf")
    )


def _groups(rollouts: list[Rollout]) -> list[list[Rollout]]:
    # The n rollouts of each task, grouped together (so they sit adjacent); groups ordered by
    # earliest start, rollouts within a group by start. Every rollout carries its `task` from
    # construction, so ones still queued behind the concurrency cap (no trace yet) are grouped
    # and shown too — as `[pending]`. Finished ones stay (never removed).
    by_task: dict[int, list[Rollout]] = {}
    for rollout in rollouts:
        by_task.setdefault(rollout.task.idx, []).append(rollout)
    groups = list(by_task.values())
    for group in groups:
        group.sort(key=_started)
    groups.sort(key=lambda g: _started(g[0]))
    return groups


def _brace(i: int, size: int) -> str:
    """The left brace piece joining a task's rollouts — ╭ top, │ middle, ╰ bottom; a space for
    a lone rollout (n=1, nothing to group)."""
    if size == 1:
        return " "
    return "╭" if i == 0 else "╰" if i == size - 1 else "│"


def Rows(groups: list[list[Rollout]], now: float, runtime_type: str) -> Table:
    # (brace, state, left sections, result, time)
    rows: list[tuple[str, str, list[str], str, str]] = []
    for group in groups:
        for i, rollout in enumerate(group):
            t = rollout.trace
            task = rollout.task
            label = f"name={task.name[:32]}" if task.name else f"idx={task.idx}"
            if (
                t is None
            ):  # queued behind the concurrency cap — only its task is known yet
                rows.append(
                    (
                        _brace(i, len(group)),
                        "pending",
                        [f"task {label}", *[""] * 7],
                        "",
                        "",
                    )
                )
                continue
            if rollout.phase == Phase.DONE:  # fully scored — reward is final
                state = "error" if t.has_error else "success"
                result = t.error.type if t.has_error else f"reward={t.reward:.2f}"
                if t.has_error:
                    stop = ""  # error shown instead
                else:
                    stop = t.stop_condition or ""
                    if (
                        t.is_truncated
                    ):  # flag a clipped rollout next to its stop condition
                        stop = f"{stop} (truncated)".strip()
            else:
                state, result, stop = rollout.phase, "", ""
            descriptor = (
                rollout.runtime.descriptor if rollout.runtime is not None else None
            )
            runtime = f"{runtime_type}({descriptor})" if descriptor else runtime_type
            turns = t.num_turns
            start = t.timing.setup.start
            end = (
                t.timing.scoring.end
                or t.timing.finalize.end
                or t.timing.generation.end
                # a rollout that errored in setup has only setup.end — freeze there once done,
                # else (still running) the timer would grow off `now` forever
                or (t.timing.setup.end if t.is_completed else 0)
                or now
            )
            prompt, completion, cached, reasoning, nbranches = _tokens(t)
            cost = t.usage.cost if t.usage else None
            tokens = ""
            if prompt or completion:
                tokens = f"{format_count(prompt)}/{format_count(completion)} tokens"
                details = []
                if cached is not None:
                    details.append(f"{format_count(cached)} cached")
                if reasoning is not None:
                    details.append(f"{format_count(reasoning)} reasoning")
                if details:
                    tokens += f" ({', '.join(details)})"
            left = [
                f"task {label}",
                t.id[:8],
                runtime,
                f"{turns} turn{'s' * (turns != 1)}",
                f"{nbranches} branch{'es' * (nbranches != 1)}",
                tokens,
                f"{format_cost_usd(cost)}" if cost is not None else "",
                stop,  # stop condition (agent_completed / max_turns / harness_timeout), once done
            ]
            # No start time yet (queued, not generating) → blank, not `now - 0` (~56 years).
            elapsed = format_time(end - start) if start else ""
            rows.append((_brace(i, len(group)), state, left, result, elapsed))
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1, no_wrap=True)  # brace + mark + dot-separated sections
    grid.add_column(justify="right", no_wrap=True)  # result (reward / error)
    grid.add_column(justify="right", no_wrap=True)  # time
    if not rows:
        return grid
    # Pad each left section to its max width across rows (drop all-empty ones) so they align,
    # then join with " · ". Text sections left-justified, numeric right.
    pad = (
        str.ljust,
        str.ljust,
        str.ljust,
        str.rjust,
        str.rjust,
        str.ljust,
        str.rjust,
        str.ljust,
    )
    widths = [max(len(left[i]) for _, _, left, _, _ in rows) for i in range(8)]
    for brace, state, left, result, elapsed in rows:
        sections = [pad[i](left[i], widths[i]) for i in range(8) if widths[i]]
        grid.add_row(
            f"{brace} {_MARK[state]} " + " · ".join(sections),
            f"{result} ·" if result else "",  # trailing dot only when there's a result
            elapsed,
            style=_STYLE[state],
        )
    return grid


class Pager:
    """Which page of overflowing rollout rows is on screen. Auto-advances on a timer until the
    user takes over with the left/right arrows, after which it stays where they leave it. Paging
    opens on the first page: the timer is anchored to `origin` (the clock at the first paged frame),
    so `int((now - origin) / _PAGE_SECONDS)` is 0 when paging begins and rotates from there, rather
    than off the raw wall clock (which would open on an arbitrary page). `_paginate` clears `origin`
    whenever everything fits on one page, so paging re-anchors and re-opens on page 1 if rollouts
    overflow again after a resize. `count` (the page count, set each render by `_paginate`) gates the
    arrows: they're inert while a single page fits, so a stray press before rollouts overflow can't
    switch off auto-advance or offset the starting page once paging begins. The arrows wrap around
    (left on the first page lands on the last, right on the last lands on the first), so the pages
    form a circle rather than dead-ending. The chosen page is still clamped to `count` between
    presses (it can shrink on a resize; it otherwise only grows, as rollouts are never removed)."""

    def __init__(self) -> None:
        self.page = 0
        self.manual = False
        self.count = 1
        self.origin: float | None = None

    def on_key(self, key: str) -> None:
        if key in ("left", "right") and self.count > 1:
            self.manual = True
            self.page = (self.page + (1 if key == "right" else -1)) % self.count

    def index(self, now: float) -> int:
        # Track the auto page while it drives, so the first arrow continues from what's on screen
        # rather than jumping back to page 1. Clamp in manual mode (count can shrink on resize).
        if not self.manual:
            # Anchor the timer to the first paged frame, so paging opens on page 1 then rotates.
            if self.origin is None:
                self.origin = now
            self.page = int((now - self.origin) / _PAGE_SECONDS) % self.count
        else:
            self.page = max(0, min(self.page, self.count - 1))
        return self.page


def _paginate(
    groups: list[list[Rollout]], rows_per_page: int, pager: Pager, now: float
) -> tuple[list[list[Rollout]], int, int]:
    """Pack groups (a task's rollouts kept together) into pages of at most `rows_per_page` rows,
    selecting the one `pager` points at. Returns (this page's groups, 0-based index, page count) —
    a single page when everything already fits."""
    if sum(len(g) for g in groups) <= rows_per_page:
        # Everything fits: arrows stay inert, and clear the anchor so paging re-opens on page 1 if
        # rollouts overflow again later (e.g. a terminal resize) rather than off the stale origin.
        pager.count = 1
        pager.origin = None
        return groups, 0, 1
    pages: list[list[list[Rollout]]] = []
    current: list[list[Rollout]] = []
    used = 0
    for group in groups:
        if current and used + len(group) > rows_per_page:
            pages.append(current)
            current, used = [], 0
        current.append(group)
        used += len(group)
    if current:
        pages.append(current)
    pager.count = len(pages)
    index = pager.index(now)
    return pages[index], index, len(pages)


def _render(
    rollouts: list[Rollout],
    config: EvalConfig,
    start: float,
    pager: Pager,
    finished: list[Trace] | None = None,
) -> Group:
    now = time.time()
    warning = _warning(config)
    # `{warning}\n\n{overview}` — the caution sits at the very top, blank line, then the overview.
    header = Group(warning, Text(""), Overview(config)) if warning else Overview(config)
    # Measure the fixed top (header + progress + rule) so the rollout rows fill the rest of the
    # screen; page through them (timer, or the left/right arrows) when they'd overflow (rich would
    # otherwise truncate).
    top = Group(header, Progress(rollouts, start, finished=finished), Rule(style="dim"))
    rows_per_page = max(1, _CONSOLE.size.height - len(_CONSOLE.render_lines(top)) - 1)
    page_groups, index, count = _paginate(_groups(rollouts), rows_per_page, pager, now)
    progress = Progress(
        rollouts,
        start,
        page=(index + 1, count) if count > 1 else None,
        finished=finished,
    )
    return Group(
        header,
        progress,
        Rule(style="dim"),
        Rows(page_groups, now, config.harness.runtime.type),
    )


@contextlib.asynccontextmanager
async def dashboard(
    rollouts: list[Rollout],
    config: EvalConfig,
    start: float,
    finished: list[Trace] | None = None,
):
    """Refresh the live eval view until the `with` block exits, then a final frame. Left/right
    arrows page through rollout rows when they overflow the screen. On resume, `finished` carries
    the kept on-disk rollouts (reloaded as finished traces) so the counts and scores cover the
    whole run, not just this session's re-run rollouts."""
    pager = Pager()
    async with live_view(
        lambda: _render(rollouts, config, start, pager, finished), on_key=pager.on_key
    ):
        yield
