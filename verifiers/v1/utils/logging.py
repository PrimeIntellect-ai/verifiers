"""Route the library's stdlib logging through loguru for the CLIs.

The library (`verifiers.v1`) logs via stdlib logging and is silent by default
(a NullHandler on the package root). A CLI opts in: it points loguru at
stderr — so logs never mix with the results printed to stdout — and installs an
`InterceptHandler` on the `verifiers.v1` logger so those records render through
loguru. Mirrors prime-rl's `intercept_vf_logging`. The `prime_tunnel.frpc` logger
(frpc output from each prime tunnel) is routed the same way at WARNING and above, so
tunnel disconnects and reconnects show up in the run's logs.
"""

import logging
import sys
from pathlib import Path

from loguru import logger

LIBRARY_LOGGER = "verifiers.v1"
TUNNEL_LOGGER = "prime_tunnel.frpc"
FORMAT = (
    "<dim>{time:HH:mm:ss}</dim> <level>{level: >7}</level> <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Forward stdlib log records into loguru, preserving level and call site."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "INFO", log_file: str | None = None, console: bool = True
) -> None:
    """Route the library's stdlib logs through loguru, to stderr (when `console`) and/or a
    `log_file`. `console=False` (used under the `--rich` dashboard, which owns the screen)
    keeps logs off the terminal while still writing them to `log_file`."""
    lvl = level.upper()
    logger.remove()
    if console:
        logger.add(sys.stderr, level=lvl, format=FORMAT)
    else:
        # The terminal is off-limits: stray stdlib records that bypass the intercept
        # below must not fall through to stderr either (they'd print over the UI).
        logging.lastResort = None
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=lvl, format=FORMAT)
    library = logging.getLogger(LIBRARY_LOGGER)
    library.handlers.clear()  # drop the opt-in NullHandler
    library.addHandler(InterceptHandler())
    library.setLevel(lvl)
    library.propagate = False
    # frpc logs routine events at INFO for every tunnel; only its warnings and errors
    # (dropped control connections, failed reconnects) are worth a line per run.
    tunnel = logging.getLogger(TUNNEL_LOGGER)
    tunnel.handlers.clear()
    tunnel.addHandler(InterceptHandler())
    tunnel.setLevel(logging.WARNING)
    tunnel.propagate = False
