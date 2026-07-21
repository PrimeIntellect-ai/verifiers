"""Live eval dashboard."""

import contextlib
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console, Group
from rich.markup import escape
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from verifiers.v1.cli.dashboard.base import live_view
from verifiers.v1.cli.output import output_path
from verifiers.v1.utils.install import env_name
from verifiers.v1.utils.interrupt import cleaning_up
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.cli.eval.slots import RunSlot
from verifiers.v1.env import agent_harnesses
from verifiers.v1.trace import Trace
from verifiers.v1.types import Usage
from verifiers.v1.utils.format import (
    format_count,
    format_mean,
    format_override,
    format_time,
)
from verifiers.utils.pricing_utils import format_cost_usd

if TYPE_CHECKING:
    from verifiers.v1.push import PushState

# For sizing pages to the terminal: detects the real terminal height/width each access (the live
# view writes to the same terminal). Reused so we don't rebuild it every refresh tick.
_CONSOLE = Console()
_PAGE_SECONDS = 5.0  # rotate to the next page of rollouts this often when they overflow
# The under-bar breakdown pads its label column to the Overview's widest label, so the two
# `label  value` grids (above and below the progress bar) line their values up.
_LABEL_WIDTH = len("timeouts")

_STYLE = {
    "pending": "dim",
    "boot": "orange3",
    "build": "dark_orange",
    "setup": "yellow",
    "running": "cyan",
    "finalize": "magenta",
    "scoring": "blue",
    "success": "green",
    "error": "red",
}
_MARK_LABEL = {
    "pending": "pending",
    "boot": "boot",
    "build": "build",
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


def _agent_value(config: EvalConfig, read):
    """A cap as the overview shows it: the declared agents' shared value, the
    string 'per-agent' when they disagree (caps live on the agents)."""
    from verifiers.v1.agents import agent_config_fields

    values = {read(spec) for spec in agent_config_fields(config.env).values()}
    return values.pop() if len(values) == 1 else "per-agent"


def _limits(config: EvalConfig) -> list[str]:
    """Per-run caps for the overview (concurrency first, then turns, tokens), read
    off the declared agents. An unset cap reads as 'no ...' rather than being hidden."""
    toks = []
    for label, field in (
        ("in", "max_input_tokens"),
        ("out", "max_output_tokens"),
        ("total", "max_total_tokens"),
    ):
        v = _agent_value(config, lambda spec, field=field: getattr(spec, field))
        if v == "per-agent":
            toks.append(f"{label} per-agent")
        elif v:
            toks.append(f"{label}≤{v}")
    turns = _agent_value(config, lambda spec: spec.max_turns)
    return [
        f"≤{config.max_concurrent} concurrent"
        if config.max_concurrent
        else "no concurrency cap",
        "per-agent turn caps"
        if turns == "per-agent"
        else (f"{turns} turns" if turns else "no turn cap"),
        f"{', '.join(toks)} tokens" if toks else "no token cap",
    ]


def _timeouts(config: EvalConfig) -> list[str]:
    """Per-stage run timeouts for the overview (each agent's own), plus the env's
    score() bound (unset → 'no <stage> timeout')."""
    rows = []
    for stage in ("setup", "rollout", "finalize", "scoring"):
        v = _agent_value(config, lambda spec, stage=stage: getattr(spec.timeout, stage))
        if v == "per-agent":
            rows.append(f"{stage} per-agent")
        elif v:
            rows.append(f"{stage} {v:g}s")
        else:
            rows.append(f"no {stage} timeout")
    env_finalize = config.env.timeout.finalize
    rows.append(
        f"env finalize {env_finalize:g}s" if env_finalize else "no env finalize timeout"
    )
    return rows


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


def _interrupt_footer() -> Group | None:
    """The graceful-shutdown notice under the rollouts once Ctrl-C has begun teardown — the
    on-screen echo of the warning (console logging is silenced in rich mode); a further Ctrl-C
    is ignored while it shows. Sits beside the `--push` line, mirroring its placement."""
    if not cleaning_up():
        return None
    return Group(
        Rule(style="dim"),
        Text(
            "interrupted — cleaning up, tearing down containers/sandboxes. "
            "please wait; a further ctrl-c is ignored.",
            style="yellow",
        ),
    )


def _warning(config: EvalConfig) -> Text | None:
    """A local-runtime caution when any code-running agent resolves to the subprocess
    runtime (the tool-less chat loops are exempt), shown above the overview rather
    than as a row in it."""
    from verifiers.v1.loaders import harness_class

    if any(
        h.runtime.type == "subprocess" and harness_class(h.id).EXECUTES_CODE
        for h in agent_harnesses(config.env).values()
    ):
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
    recursion so a pinned nested default (`taskset.task.tools`) reads as
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
    agents = agent_harnesses(config.env)
    taskset = config.env.taskset
    env_label = taskset.name if taskset is not None else "no taskset"
    if config.env.id:
        env_label = f"{env_name(config.env.id)}+{env_label}"
    # One story when every agent resolves the same way (the common case); one
    # row per agent when they diverge (a grader on its own harness/runtime).
    stories = list(
        dict.fromkeys(
            f"{h.name} harness  ·  {h.runtime.type} runtime" for h in agents.values()
        )
    )
    if len(stories) == 1:
        grid.add_row("env", f"{env_label}  ·  {stories[0]}")
    else:
        grid.add_row("env", env_label)
        for agent, h in agents.items():
            grid.add_row(f"  {agent}", f"{h.name} harness  ·  {h.runtime.type} runtime")
    model = f"{config.model}  ({sampling})" if sampling else config.model
    grid.add_row("model", f"{model}  via {config.client.base_url}")
    # Non-default knobs the user set, one row each when non-empty. `escape` the cell: an override
    # value (or our `[...]`/`{...}` delimiters) can carry Rich markup that would otherwise be
    # parsed as styling and dropped. `id` is in the `env` row; harness `runtime.type` too (hidden
    # here), but only for the harness — `taskset.task.tools.runtime.type` has no other display.
    if taskset is not None and (
        taskset_over := overrides(taskset, skip=frozenset({"id"}))
    ):
        grid.add_row("taskset", escape("  ·  ".join(taskset_over)))
    for agent, h in agents.items():
        if harness_over := overrides(h, skip=frozenset({"id", "runtime.type"})):
            label = f"{agent}.harness" if len(agents) > 1 else "harness"
            grid.add_row(label, escape("  ·  ".join(harness_over)))
    limits, timeouts = _aligned([_limits(config), _timeouts(config)])
    grid.add_row("limits", limits)
    grid.add_row("timeouts", timeouts)
    grid.add_row("output", Text(str(output_path(config)), overflow="fold"))
    return grid


def _push_footer(push: "PushState | None") -> Group | None:
    """The `--push` status line under the rollouts, shown once the run finishes and the upload
    begins: dim `Pushing traces...` while it runs, then white `Traces pushed (<url>)` or red
    `Trace push failed (<err>)`. `None` (no line) until the upload starts and when `--push` is off."""
    if push is None or not push.started:
        return None
    if not push.done:
        line = Text("Pushing traces...", style="dim")
    elif push.url:
        line = Text(f"Traces pushed ({push.url})", style="white", overflow="fold")
    else:
        line = Text(f"Trace push failed ({push.error})", style="red", overflow="fold")
    return Group(Rule(style="dim"), line)


def Progress(
    slots: list[RunSlot],
    start: float,
    page: tuple[int, int] | None = None,
) -> Group:
    # On resume, `slots` includes the previous session's kept rollouts (as finished slots), so
    # progress, reward, err, and the breakdown cover the whole run, not just this session's.
    done = [s for s in slots if s.done]  # fully scored env-rollouts
    done_traces = [t for s in done for t in s.traces]
    # Score aggregates read the policy's traces: auxiliary agents (a grader's verdict
    # run, a modeled user) are `trainable=False` and carry no rewards, so counting
    # them dilutes every mean with structural zeros. An all-untrainable run (every
    # agent frozen) falls back to all traces rather than showing nothing.
    scored = [t for t in done_traces if t.trainable] or done_traces
    total = len(slots)
    # Headline reward = mean over non-errored traces; when any errored, `format_mean` appends
    # the global avg (errored count as 0) in parens. `err` is the share of env-rollouts that
    # ended not-ok (a trace errored, or the env's rollout()/score() hook itself failed).
    reward = format_mean(scored, lambda t: t.reward)
    err = f"{sum(not s.ok for s in done) / len(done):.2f}" if done else "—"
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
    breakdown = _breakdown(scored, done_traces)
    return Group(row, breakdown) if breakdown is not None else Group(row)


def _score_segments(traces: list[Trace], source: str) -> str | None:
    """`name mean` segments for every reward/metric key seen across `traces`,
    first-seen order (a trace records only the functions that ran for it, so keys
    can vary); None when no trace recorded anything."""
    names: list[str] = []
    for trace in traces:
        names.extend(n for n in getattr(trace, source) if n not in names)
    if not names:
        return None
    segments = []
    for name in names:
        mean = format_mean(
            traces, lambda t, n=name, s=source: getattr(t, s).get(n, 0.0)
        )
        segments.append(f"{name} {mean}")
    return "  ·  ".join(segments)


def _breakdown(scored: list[Trace], done: list[Trace]) -> Table | None:
    """Score rows read the policy view (`scored` — trainable traces); with several
    agents in play they split per agent, each averaging over its OWN traces (no
    dilution, so the split covers every agent — an untrainable agent's received
    rewards stay visible). Resource rows read every completed trace: a grader's
    tokens were spent regardless of trainability."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=_LABEL_WIDTH)
    grid.add_column()
    # rewards/metrics are error-corrected means — skip when every rollout errored (no clean mean to
    # show); usage/time below still cover errored rollouts (their resources were spent regardless).
    has_clean = any(not t.has_error for t in done)
    score_rows = (("rewards", "rewards"), ("metrics", "metrics")) if has_clean else ()
    by_agent: dict[str | None, list[Trace]] = {}
    for trace in done:
        name = trace.agent.name if trace.agent is not None else "agent"
        by_agent.setdefault(name, []).append(trace)
    for label, source in score_rows:
        if len(by_agent) > 1:
            segments = [
                f"[dim]{name or '—'}:[/dim] {means}"
                for name, traces in by_agent.items()
                if (means := _score_segments(traces, source)) is not None
            ]
            if segments:
                grid.add_row(label, "    ".join(segments))
        elif (means := _score_segments(scored, source)) is not None:
            grid.add_row(label, means)

    # Resource use over every completed rollout (errored ones still spent tokens/time): tokens and
    # cost are summed; each timing phase is averaged over the rollouts that have it timed (averaged
    # per phase rather than summed, since phases run concurrently across rollouts).
    total_in = total_out = total_cached = total_reasoning = 0
    total_judge_in = total_judge_out = 0
    total_cost = total_judge_cost = 0.0
    have_cost = have_cached = have_reasoning = have_judge = False
    phase_secs: dict[str, float] = {}
    phase_count: dict[str, int] = {}
    model_secs = harness_secs = 0.0
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
        for phase in ("boot", "setup", "generation", "finalize", "scoring"):
            span = getattr(trace.timing, phase)
            if span.end:  # phase was timed for this rollout
                phase_secs[phase] = phase_secs.get(phase, 0.0) + span.duration
                phase_count[phase] = phase_count.get(phase, 0) + 1
        model_secs += trace.timing.generation.model.duration
        harness_secs += trace.timing.generation.harness.duration
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
    time_segments = []
    for phase in ("boot", "setup", "generation", "finalize", "scoring"):
        count = phase_count.get(phase)
        if not count:
            continue
        segment = f"{phase} {format_time(phase_secs[phase] / count)}"
        if phase == "generation":
            segment += (
                f" (model {format_time(model_secs / count)}"
                f" + harness {format_time(harness_secs / count)})"
            )
        time_segments.append(segment)
    if time_segments:
        grid.add_row("time", "  ·  ".join(time_segments))
    return grid if grid.row_count else None


def _tokens(trace: Trace) -> tuple[int, int, int | None, int | None, int]:
    """Input/output tokens summed across all branches: per branch, output is every assistant
    (completion) token generated across its turns and input is the fed-in tokens counted once
    (system + user + tool) — the final sequence minus everything the model generated. A rollout
    yields one training sample per branch (a linear trace is a single branch; compaction and
    subagents add more), so the totals sum them — matching `Trace.num_input_tokens` /
    `Trace.num_output_tokens`, whose sum is `num_total_tokens`.

    Both counts come from provider-reported usage. Returns the branch count from the same derived
    view so each dashboard tick materializes it once."""
    usage = trace.usage
    cached = usage.cached_input_tokens if usage else None
    reasoning = usage.reasoning_tokens if usage else None
    branches = trace.branches
    nbranches = len(branches)
    prompt = sum(b.num_input_tokens for b in branches)
    completion = sum(b.num_output_tokens for b in branches)
    return prompt, completion, cached, reasoning, nbranches


def _stage(trace: Trace) -> str:
    """The stage a live (not-yet-done) rollout is in, derived from its trace's timing
    spans — the engine opens and closes each span exactly at the stage transitions, so
    the current stage is the latest span started but not yet ended. A completed trace
    whose slot isn't done is waiting on its env-rollout's other traces (and the env's
    `score()`) — that's scoring."""
    if trace.is_completed:
        return "scoring"
    for stage, span in (
        ("scoring", trace.timing.scoring),
        ("finalize", trace.timing.finalize),
        ("running", trace.timing.generation),
        ("setup", trace.timing.setup),
        ("boot", trace.timing.boot),
    ):
        if span.start and not span.end:
            break
    else:
        stage = "boot"  # trace minted, first span not yet opened (an instant)
    # A boot stuck on a first-use platform image build reads differently from a
    # normal boot — it can sit there for ~10 minutes (prime runtime only).
    if stage == "boot" and getattr(trace.runtime, "image_cached", None) is False:
        return "build"
    return stage


def _started(slot: RunSlot) -> float:
    # Sort key: when a rollout began (its first trace's boot start; setup for
    # pre-boot-span traces on resume). A still-pending rollout has no trace yet, so it
    # sorts last (+inf) — behind everything already in flight, in task order.
    if not slot.traces:
        return float("inf")
    return min(t.timing.boot.start or t.timing.setup.start for t in slot.traces)


def _groups(slots: list[RunSlot]) -> list[list[RunSlot]]:
    # The n rollouts of each task, grouped together (so they sit adjacent); groups ordered by
    # earliest start, rollouts within a group by start. Every slot carries its `task` from
    # construction, so ones still queued behind the concurrency cap (no trace yet) are grouped
    # and shown too — as `[pending]`. Finished ones stay (never removed).
    by_task: dict[int, list[RunSlot]] = {}
    for slot in slots:
        by_task.setdefault(slot.task.data.idx, []).append(slot)
    groups = list(by_task.values())
    for group in groups:
        group.sort(key=_started)
    groups.sort(key=lambda g: _started(g[0]))
    return groups


def _brace(i: int, size: int) -> str:
    if size == 1:
        return " "
    return "╭" if i == 0 else "╰" if i == size - 1 else "│"


def Rows(groups: list[list[RunSlot]], now: float, runtime_type: str) -> Table:
    # (brace, state, left sections, result, time); a slot contributes one row per live
    # trace (a multi-agent env-rollout shows each agent's trace), braced per task.
    rows: list[tuple[str, str, list[str], str, str]] = []
    for group in groups:
        group_rows: list[tuple[str, list[str], str, str]] = []
        for slot in group:
            task = slot.task.data
            base = f"name={task.name[:32]}" if task.name else f"idx={task.idx}"
            if not slot.traces:
                if slot.done:  # the env's run() itself failed before any trace
                    error = slot.errors[-1] if slot.errors else None
                    group_rows.append(
                        (
                            "error",
                            [f"task {base}", *[""] * 7],
                            error.type if error is not None else "error",
                            "",
                        )
                    )
                else:  # queued behind the concurrency cap — only its task is known yet
                    group_rows.append(("pending", [f"task {base}", *[""] * 7], "", ""))
                continue
            for t in slot.traces:
                # Label the agent only when the episode has several traces — a
                # single-agent row's `agent=agent` would be noise.
                label = (
                    f"{base} agent={t.agent.name}"
                    if t.agent is not None and len(slot.traces) > 1
                    else base
                )
                if slot.done:  # fully scored — reward is final
                    state = "error" if t.has_error else "success"
                    # A trace that recorded nothing shows no reward: a judge or
                    # modeled-user agent's `reward=0.00` would read as a score.
                    result = (
                        t.error.type
                        if t.has_error
                        else (f"reward={t.reward:.2f}" if t.rewards else "")
                    )
                    if t.has_error:
                        stop = ""  # error shown instead
                    else:
                        stop = t.stop_condition or ""
                        if (
                            t.is_truncated
                        ):  # flag a clipped rollout next to its stop condition
                            stop = f"{stop} (truncated)".strip()
                elif t.is_completed and (err := t.error) is not None:
                    # An errored trace whose env-rollout is still running its other
                    # traces (or `score()`) is already a failure — show it, don't
                    # let it sit as "scoring" until the whole episode lands.
                    state, result, stop = "error", err.type, ""
                else:
                    state, result, stop = _stage(t), "", ""
                # The trace's own stamp, not the run-level runtime: an agent's harness
                # may resolve elsewhere (the solver-grader env's sandboxed grader on
                # a subprocess run).
                if t.runtime is not None:
                    runtime = (
                        f"{t.runtime.type}({t.runtime.id})"
                        if t.runtime.id
                        else t.runtime.type
                    )
                else:
                    runtime = runtime_type
                turns = t.num_turns
                start = t.timing.boot.start or t.timing.setup.start
                end = (
                    t.timing.scoring.end
                    or t.timing.finalize.end
                    or t.timing.generation.end
                    # a rollout that errored in boot/setup has only that span's end — freeze there
                    # once done, else (still running) the timer would grow off `now` forever
                    or (
                        (t.timing.setup.end or t.timing.boot.end)
                        if t.is_completed
                        else 0
                    )
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
                group_rows.append((state, left, result, elapsed))
        for i, (state, left, result, elapsed) in enumerate(group_rows):
            rows.append((_brace(i, len(group_rows)), state, left, result, elapsed))
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


def _rows_of(group: list[RunSlot]) -> int:
    """How many display rows a task's slots take: one per trace, one for a slot with
    none yet (pending) or none at all (the env's hook failed before any trace)."""
    return sum(max(1, len(slot.traces)) for slot in group)


def _paginate(
    groups: list[list[RunSlot]], rows_per_page: int, pager: Pager, now: float
) -> tuple[list[list[RunSlot]], int, int]:
    """Pack groups (a task's rollouts kept together) into pages of at most `rows_per_page` rows,
    selecting the one `pager` points at. Returns (this page's groups, 0-based index, page count) —
    a single page when everything already fits."""
    if sum(_rows_of(g) for g in groups) <= rows_per_page:
        # Everything fits: arrows stay inert, and clear the anchor so paging re-opens on page 1 if
        # rollouts overflow again later (e.g. a terminal resize) rather than off the stale origin.
        pager.count = 1
        pager.origin = None
        return groups, 0, 1
    pages: list[list[list[RunSlot]]] = []
    current: list[list[RunSlot]] = []
    used = 0
    for group in groups:
        if current and used + _rows_of(group) > rows_per_page:
            pages.append(current)
            current, used = [], 0
        current.append(group)
        used += _rows_of(group)
    if current:
        pages.append(current)
    pager.count = len(pages)
    index = pager.index(now)
    return pages[index], index, len(pages)


def _render(
    slots: list[RunSlot],
    config: EvalConfig,
    start: float,
    pager: Pager,
    push: "PushState | None" = None,
) -> Group:
    now = time.time()
    warning = _warning(config)
    header = Group(warning, Text(""), Overview(config)) if warning else Overview(config)
    # The --push status line (and, on Ctrl-C, the cleanup notice) appear under the rollouts. Measure
    # the fixed top (header + progress + rule) and the footer so the rollout rows fill what's left;
    # page through them (timer / arrows) when they'd overflow (else rich truncates).
    footers = [f for f in (_push_footer(push), _interrupt_footer()) if f is not None]
    footer = Group(*footers) if footers else None
    top = Group(header, Progress(slots, start), Rule(style="dim"))
    reserved = len(_CONSOLE.render_lines(top))
    if footer is not None:
        reserved += len(_CONSOLE.render_lines(footer))
    rows_per_page = max(1, _CONSOLE.size.height - reserved - 1)
    page_groups, index, count = _paginate(_groups(slots), rows_per_page, pager, now)
    progress = Progress(
        slots,
        start,
        page=(index + 1, count) if count > 1 else None,
    )
    parts = [
        header,
        progress,
        Rule(style="dim"),
        Rows(
            page_groups,
            now,
            "/".join(
                dict.fromkeys(
                    h.runtime.type for h in agent_harnesses(config.env).values()
                )
            )
            or "subprocess",
        ),
    ]
    if footer is not None:
        parts.append(footer)
    return Group(*parts)


@contextlib.asynccontextmanager
async def dashboard(
    slots: list[RunSlot],
    config: EvalConfig,
    start: float,
    push: "PushState | None" = None,
):
    pager = Pager()
    async with live_view(
        lambda: _render(slots, config, start, pager, push),
        on_key=pager.on_key,
    ):
        yield
