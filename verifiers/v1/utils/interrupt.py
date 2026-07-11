"""Graceful-shutdown signal handling for the eval CLI.

The first Ctrl-C (or SIGTERM) flips `cleaning_up()` on and raises KeyboardInterrupt so
asyncio unwinds each rollout's `finally` — the path that tears down containers/sandboxes
and any worker pool the run spawned. Any further shutdown signal during that window is
swallowed, so an impatient second Ctrl-C can't cut cleanup short and orphan those
resources. A genuinely stuck run is still killable with SIGKILL.

In rich mode the dashboard renders the notice from `cleaning_up()` (console logging is
silenced there); otherwise the handler echoes it to stderr, where teardown logs stream
alongside it. The flag is a process-level shutdown latch — signals are process-global, so
it sits with the atexit runtime backstop rather than any per-run object."""

import signal
import sys

_cleaning_up = False


def cleaning_up() -> bool:
    """True once a shutdown signal has started graceful cleanup."""
    return _cleaning_up


def install(rich: bool) -> None:
    """Route SIGINT/SIGTERM through the graceful-shutdown handler."""

    def handle(*_) -> None:
        global _cleaning_up
        first, _cleaning_up = not _cleaning_up, True
        if not rich:
            sys.stderr.write(
                "\ninterrupted — cleaning up, please wait...\n"
                if first
                else "cleanup in progress — please wait (ctrl-c ignored)\n"
            )
            sys.stderr.flush()
        if first:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
