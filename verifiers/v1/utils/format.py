"""Compact human-readable duration formatting for display."""


def format_time(seconds: float) -> str:
    """A compact human-readable duration (mirrors prime-rl): ms / s / m s / h m / d h."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    if seconds < 86400:
        h, rem = divmod(seconds, 3600)
        return f"{int(h)}h {int(rem // 60)}m"
    d, rem = divmod(seconds, 86400)
    return f"{int(d)}d {int(rem // 3600)}h"
