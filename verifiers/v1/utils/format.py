"""Compact human-readable formatting for display (durations, counts, headline reward)."""

from __future__ import annotations

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


def format_reward(traces: list[Trace], digits: int = 2) -> str:
    """Headline reward over completed `traces`: the mean over the non-errored ones
    (error-corrected). When some errored, also append the global mean — over *all*
    completed, errored counting as 0 — in parens, so both the error-corrected and the
    raw reward are visible. "—" when nothing has completed (or all errored)."""
    if not traces:
        return "—"
    clean = [t for t in traces if not t.has_error]
    if not clean:
        return "—"
    reward = f"{sum(t.reward for t in clean) / len(clean):.{digits}f}"
    if len(clean) < len(traces):  # some errored → show the raw global avg alongside
        reward += f" ({sum(t.reward for t in traces) / len(traces):.{digits}f})"
    return reward
