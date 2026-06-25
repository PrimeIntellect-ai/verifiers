"""The eval `--rich` dashboard: a config overview, a progress bar, and one line per rollout.

Reads each `Rollout.trace`/`phase` every tick — no extra plumbing. Each row carries a bracketed
phase marker that reads at a glance — `[setup]` (yellow), `[rollout]` (cyan), `[finalize]`
(magenta), `[scoring]` (blue), `[success]` (green), `[error]` (red) — padded so the brackets
line up in a column down the left edge. The reward shows only once a rollout is fully scored
(phase DONE), so it never flips as scoring lands. A task's rollouts are grouped adjacently and joined
by a left brace (╭│╰), so an episode (a task's n rollouts) reads as a unit. Every started
rollout stays on screen (finished ones keep their result); the overview + progress sit on top,
above a rule.
"""

import contextlib
import time

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
from verifiers.v1.utils.format import format_count, format_mean, format_time
from verifiers.utils.pricing_utils import format_cost_usd

# For sizing pages to the terminal: detects the real terminal height/width each access (the live
# view writes to the same terminal). Reused so we don't rebuild it every refresh tick.
_CONSOLE = Console()
_PAGE_SECONDS = 5.0  # rotate to the next page of rollouts this often when they overflow
# The under-bar breakdown pads its label column to the Overview's widest label, so the two
# `label  value` grids (above and below the progress bar) line their values up.
_LABEL_WIDTH = len("timeouts")

_STYLE = {
    "setup": "yellow",
    "running": "cyan",
    "finalize": "magenta",
    "scoring": "blue",
    "success": "green",
    "error": "red",
}
_MARK_LABEL = {
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
    """A local-runtime caution for a code-running harness (none for the tool-less `default`),
    shown above the overview rather than as a row in it."""
    if config.harness.id != "default" and config.harness.runtime.type == "subprocess":
        return Text(
            "warning  Runs on the local system; local files and settings may affect this "
            "evaluation. Use subprocess only for debugging, or use docker or prime for an "
            "isolated run.",
            style="yellow",
        )
    return None


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
    limits, timeouts = _aligned([_limits(config), _timeouts(config)])
    grid.add_row("limits", limits)
    grid.add_row("timeouts", timeouts)
    grid.add_row("output", str(output_path(config)))
    return grid


def Progress(
    rollouts: list[Rollout], start: float, page: tuple[int, int] | None = None
) -> Group:
    done = [r.trace for r in rollouts if r.phase == Phase.DONE]  # fully scored
    # Headline reward = mean over non-errored; when any errored, `format_mean` appends the
    # global avg (errored count as 0) in parens. `err` is the share that errored.
    reward = format_mean(done, lambda t: t.reward)
    err = f"{sum(t.has_error for t in done) / len(done):.2f}" if done else "—"
    stats = (
        f"{len(done)}/{len(rollouts)} · {format_time(time.time() - start)} · "
        f"reward {reward} · err {err}"
    )
    if page is not None:  # rollouts overflow the screen — show which page is on screen
        stats += f"  (page {page[0]}/{page[1]})"
    row = Table.grid(expand=True, padding=(0, 1))
    row.add_column(ratio=1)  # bar stretches to fill the width left of the stats
    row.add_column(justify="right", no_wrap=True)
    row.add_row(
        ProgressBar(total=len(rollouts) or 1, completed=len(done)),
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
        segments = [
            f"{name} {format_mean(done, lambda t, n=name, s=source: getattr(t, s).get(n, 0.0))}"
            for name in names
        ]
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
    `Trace.prompt_len` / `Trace.completion_len`. (Output can exceed the final context — reasoning
    tokens count toward completions but aren't re-fed — so it's not derived by subtraction.)

    Prefers the token-id counts; falls back to provider-reported usage when the endpoint returns
    no token ids (e.g. plain OpenAI completions), so the counts aren't shown as 0/0. Returns
    the branch count from the same derived view so each dashboard tick materializes it once."""
    usage = trace.usage
    cached = usage.cached_input_tokens if usage else None
    reasoning = usage.reasoning_tokens if usage else None
    branches = trace.branches
    nbranches = len(branches)
    prompt = sum(b.input_len for b in branches)
    completion = sum(b.output_len for b in branches)
    return prompt, completion, cached, reasoning, nbranches


def _groups(rollouts: list[Rollout]) -> list[list[Rollout]]:
    # The n rollouts of each task, grouped together (so they sit adjacent); groups ordered by
    # earliest start, rollouts within a group by start. Finished ones stay (never removed).
    by_task: dict[int, list[Rollout]] = {}
    for rollout in rollouts:
        if rollout.trace is not None:
            by_task.setdefault(rollout.trace.task.idx, []).append(rollout)
    groups = list(by_task.values())
    for group in groups:
        group.sort(key=lambda r: r.trace.timing.setup.start)
    groups.sort(key=lambda g: g[0].trace.timing.setup.start)
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
            label = f"name={t.task.name[:32]}" if t.task.name else f"idx={t.task.idx}"
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


def _paginate(
    groups: list[list[Rollout]], rows_per_page: int, now: float
) -> tuple[list[list[Rollout]], int, int]:
    """Pack groups (a task's rollouts kept together) into pages of at most `rows_per_page` rows,
    cycling to the next page every `_PAGE_SECONDS`. Returns (this page's groups, 0-based index,
    page count) — a single page when everything already fits."""
    if sum(len(g) for g in groups) <= rows_per_page:
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
    index = int(now / _PAGE_SECONDS) % len(pages)
    return pages[index], index, len(pages)


def _render(rollouts: list[Rollout], config: EvalConfig, start: float) -> Group:
    now = time.time()
    warning = _warning(config)
    # `{warning}\n\n{overview}` — the caution sits at the very top, blank line, then the overview.
    header = Group(warning, Text(""), Overview(config)) if warning else Overview(config)
    # Measure the fixed top (header + progress + rule) so the rollout rows fill the rest of the
    # screen; page through them on a timer when they'd overflow (rich would otherwise truncate).
    top = Group(header, Progress(rollouts, start), Rule(style="dim"))
    rows_per_page = max(1, _CONSOLE.size.height - len(_CONSOLE.render_lines(top)) - 1)
    page_groups, index, count = _paginate(_groups(rollouts), rows_per_page, now)
    progress = Progress(rollouts, start, page=(index + 1, count) if count > 1 else None)
    return Group(
        header,
        progress,
        Rule(style="dim"),
        Rows(page_groups, now, config.harness.runtime.type),
    )


@contextlib.asynccontextmanager
async def dashboard(rollouts: list[Rollout], config: EvalConfig, start: float):
    """Refresh the live eval view until the `with` block exits, then a final frame."""
    async with live_view(lambda: _render(rollouts, config, start)):
        yield
