"""Compact human-readable formatting for display (durations, counts, headline reward)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.trace import Trace


def format_time(seconds: float) -> str:
    """A compact human-readable duration (mirrors prime-rl): s / m s / h m / d h."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    if seconds < 86400:
        h, rem = divmod(int(seconds), 3600)
        return f"{h}h {rem // 60}m"
    d, rem = divmod(int(seconds), 86400)
    return f"{d}d {rem // 3600}h"


def format_count(n: int) -> str:
    """A compact count (mirrors prime-rl): 936 / 4.8K / 1.5M."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1e3:.1f}K"
    return f"{n / 1e6:.1f}M"


def format_mean(
    traces: list[Trace],
    value: Callable[[Trace], float],
    digits: int = 2,
    *,
    base_sum: float = 0.0,
    base_n: int = 0,
) -> str:
    """Mean of `value` over completed `traces`: the mean over the non-errored ones
    (error-corrected). When some errored, also append the global mean — over *all* completed,
    an errored trace's value counting as 0 — in parens, so both the error-corrected and the raw
    mean are visible. "—" when nothing has completed (or all errored).

    `base_sum`/`base_n` seed the mean with already-counted, non-errored rows that aren't in
    `traces` — a resume's kept on-disk rollouts — so the displayed mean covers the whole run.
    They count toward both the error-corrected and the global mean (kept rows never errored)."""
    clean = [t for t in traces if not t.has_error]
    clean_n = base_n + len(clean)
    if clean_n == 0:
        return "—"
    mean = f"{(base_sum + sum(value(t) for t in clean)) / clean_n:.{digits}f}"
    if (
        base_n + len(traces) > clean_n
    ):  # some errored → show the raw global avg alongside
        all_sum = base_sum + sum(value(t) for t in traces)
        mean += f" ({all_sum / (base_n + len(traces)):.{digits}f})"
    return mean
