"""Graceful-shutdown signal handling for the eval CLI: the first Ctrl-C/SIGTERM warns and
raises KeyboardInterrupt so asyncio unwinds each rollout's teardown `finally`; further signals
are swallowed so a second Ctrl-C can't orphan containers/sandboxes mid-cleanup."""

import logging
import signal

logger = logging.getLogger(__name__)

_cleaning_up = False


def cleaning_up() -> bool:
    """Whether a shutdown signal has begun graceful cleanup (read by the dashboard)."""
    return _cleaning_up


def install() -> None:
    """Route SIGINT/SIGTERM through the graceful-shutdown handler."""

    def handle(*_) -> None:
        global _cleaning_up
        first, _cleaning_up = not _cleaning_up, True
        logger.warning(
            "interrupted — cleaning up, please wait..."
            if first
            else "cleanup in progress — please wait (ctrl-c ignored)"
        )
        if first:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
