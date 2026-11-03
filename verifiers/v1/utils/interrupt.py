"""Graceful-shutdown signal handling shared across the v1 CLI entrypoints (run, gepa,
validate, debug): the first Ctrl-C/SIGTERM warns and raises KeyboardInterrupt so
asyncio unwinds each run's teardown `finally`; further signals are swallowed so a second
Ctrl-C can't orphan containers/sandboxes/tool-server runtimes mid-cleanup."""

import signal

from verifiers.v1.utils.log_scope import get_logger

logger = get_logger(__name__)

_cleaning_up = False


def install_interrupt() -> None:
    """Route SIGINT/SIGTERM through the graceful-shutdown handler. Call once from a CLI
    entrypoint before running its async main."""

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
