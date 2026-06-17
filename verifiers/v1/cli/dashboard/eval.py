"""The eval `--rich` dashboard: a config overview, a progress bar, and one line per rollout.

Reads each `Rollout.trace`/`phase` every tick — no extra plumbing. Rows are colored by
phase/outcome: setup (yellow ○), running (cyan ●), finalize (magenta ◑), scoring (blue ◐),
success (green ✓), error (red ✗) — the reward shows only once a rollout is fully scored (phase
DONE), so it never flips as scoring lands. A task's rollouts are grouped adjacently and joined
by a left brace (╭│╰), so an episode (a task's n rollouts) reads as a unit. Every started
rollout stays on screen (finished ones keep their result); the overview + progress sit on top,
above a rule.
"""

import contextlib
import time

from rich.console import Group
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.cli.output import output_path
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.trace import Trace
from verifiers.v1.utils import format_count, format_reward, format_time

_STYLE = {
    "setup": "yellow",
    "running": "cyan",
    "finalize": "magenta",
    "scoring": "blue",
    "success": "green",
    "error": "red",
}
_MARK = {
    "setup": "○",
    "running": "●",
    "finalize": "◑",
    "scoring": "◐",
    "success": "✓",
    "error": "✗",
}


def _limits(config: EvalConfig) -> str:
    """Per-rollout caps for the overview: turns, tokens, concurrency. An unset cap reads as
    'no ...' rather than being hidden."""
    parts = [f"{config.max_turns} turns" if config.max_turns else "no turn cap"]
    toks = []
    if config.max_input_tokens:
        toks.append(f"in≤{config.max_input_tokens}")
    if config.max_output_tokens:
        toks.append(f"out≤{config.max_output_tokens}")
    if config.max_total_tokens:
        toks.append(f"total≤{config.max_total_tokens}")
    parts.append(f"{', '.join(toks)} tokens" if toks else "no token cap")
    parts.append(
        f"≤{config.max_concurrent} concurrent"
        if config.max_concurrent
        else "no concurrency cap"
    )
    return "  ·  ".join(parts)


def _timeouts(config: EvalConfig) -> str:
    """Per-stage rollout timeouts for the overview, each stage enumerated (unset → 'no <stage>
    timeout')."""
    parts = []
    for stage in ("setup", "rollout", "finalize", "scoring"):
        value = getattr(config.timeout, stage)
        parts.append(f"{stage} {value:g}s" if value else f"no {stage} timeout")
    return "  ·  ".join(parts)


def _retries(config: EvalConfig) -> str:
    """Per-call (model, runtime) and whole-rollout retry budgets for the overview."""
    r = config.retries
    return (
        f"model ×{r.model.max_retries}  ·  runtime ×{r.runtime.max_retries}  ·  "
        f"rollout ×{r.rollout.max_retries}"
    )


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
    if config.harness.id != "default" and config.harness.runtime.type == "subprocess":
        grid.add_row(
            "warning",
            Text(
                "Runs on the local system; local files and settings may affect this "
                "evaluation. Use subprocess only for debugging, or use docker or prime "
                "for an isolated run.",
                style="yellow",
            ),
        )
    grid.add_row("model", f"{config.model}  ({sampling})" if sampling else config.model)
    grid.add_row("limits", _limits(config))
    grid.add_row("timeouts", _timeouts(config))
    grid.add_row("retries", _retries(config))
    grid.add_row("output", str(output_path(config)))
    return grid


def Progress(rollouts: list[Rollout], start: float) -> Table:
    done = [r.trace for r in rollouts if r.phase == Phase.DONE]  # fully scored
    # Headline reward = mean over non-errored; when any errored, `format_reward` appends the
    # global avg (errored count as 0) in parens. `err` is the share that errored.
    reward = format_reward(done)
    err = f"{sum(t.has_error for t in done) / len(done):.2f}" if done else "—"
    stats = (
        f"{len(done)}/{len(rollouts)} · {format_time(time.time() - start)} · "
        f"reward {reward} · err {err}"
    )
    row = Table.grid(expand=True, padding=(0, 1))
    row.add_column(ratio=1)  # bar stretches to fill the width left of the stats
    row.add_column(justify="right", no_wrap=True)
    row.add_row(
        ProgressBar(total=len(rollouts) or 1, completed=len(done)),
        Text(stats),
    )
    return row


def _tokens(trace: Trace) -> str:
    """Input/output tokens for the main branch: output is every assistant (completion) token
    generated across the branch's turns; input is the last turn's prompt — the full final
    context the model saw. (Output can exceed the final context — reasoning tokens count toward
    completions but aren't re-fed — so it's not derived by subtraction.)

    Prefers the token-id counts; falls back to provider-reported usage when the endpoint returns
    no token ids (e.g. plain OpenAI completions), so the counts aren't shown as 0/0."""
    branches = trace.branches
    if not branches or not branches[0].nodes:
        return ""
    b = branches[0]
    prompt = b.prompt_len or b.num_prompt_tokens
    completion = b.completion_len or b.num_completion_tokens
    if not prompt and not completion:
        return ""
    return f"{format_count(prompt)}/{format_count(completion)} tokens"


def _groups(rollouts: list[Rollout]) -> list[list[Rollout]]:
    # The n rollouts of each task, grouped together (so they sit adjacent); groups ordered by
    # earliest start, rollouts within a group by start. Finished ones stay (never removed).
    by_task: dict[int, list[Rollout]] = {}
    for rollout in rollouts:
        if rollout.trace is not None:
            by_task.setdefault(rollout.trace.task.idx, []).append(rollout)
    groups = list(by_task.values())
    for group in groups:
        group.sort(key=lambda r: r.trace.timing.generation.start)
    groups.sort(key=lambda g: g[0].trace.timing.generation.start)
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
            nbranches = len(t.branches)
            start = t.timing.generation.start
            end = (
                t.timing.scoring.end
                or t.timing.finalize.end
                or t.timing.generation.end
                or now
            )
            left = [
                f"task {label}",
                t.id[:8],
                runtime,
                f"{turns} turn{'s' * (turns != 1)}",
                f"{nbranches} branch{'es' * (nbranches != 1)}",
                _tokens(t),
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
    pad = (str.ljust, str.ljust, str.ljust, str.rjust, str.rjust, str.rjust, str.ljust)
    widths = [max(len(left[i]) for _, _, left, _, _ in rows) for i in range(7)]
    for brace, state, left, result, elapsed in rows:
        sections = [pad[i](left[i], widths[i]) for i in range(7) if widths[i]]
        grid.add_row(
            f"{brace} {_MARK[state]} " + " · ".join(sections),
            f"{result} ·" if result else "",  # trailing dot only when there's a result
            elapsed,
            style=_STYLE[state],
        )
    return grid


def _render(rollouts: list[Rollout], config: EvalConfig, start: float) -> Group:
    return Group(
        Overview(config),
        Rule(style="dim"),
        Progress(rollouts, start),
        Rule(style="dim"),
        Rows(_groups(rollouts), time.time(), config.harness.runtime.type),
    )


@contextlib.asynccontextmanager
async def dashboard(rollouts: list[Rollout], config: EvalConfig, start: float):
    """Refresh the live eval view until the `with` block exits, then a final frame."""
    async with live_view(lambda: _render(rollouts, config, start)):
        yield
