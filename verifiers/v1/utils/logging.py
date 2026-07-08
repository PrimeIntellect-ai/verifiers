"""Route the library's stdlib logging through loguru for the CLIs.

The library (`verifiers.v1`) logs via stdlib logging and is silent by default
(a NullHandler on the package root). A CLI opts in: it points loguru at
stderr — so logs never mix with the results printed to stdout — and installs an
`InterceptHandler` on the `verifiers.v1` logger so those records render through
loguru. Mirrors prime-rl's `intercept_vf_logging`.
"""

import logging
import sys
from pathlib import Path

from loguru import logger
from rich import get_console

LIBRARY_LOGGER = "verifiers.v1"
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
    keeps routine logs off the terminal while still writing them to `log_file` — but
    warnings still surface, printed through the dashboard's own rich Console so an active
    `Live` renders them above the live region instead of tearing the frame (a Ctrl-C
    teardown drain must not look like a hang)."""
    lvl = level.upper()
    logger.remove()
    if console:
        logger.add(sys.stderr, level=lvl, format=FORMAT)
    else:
        rich_console = get_console()
        logger.add(
            lambda m: rich_console.print(m, end="", markup=False, highlight=False),
            level="WARNING",
            format=FORMAT,
        )
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=lvl, format=FORMAT)
    library = logging.getLogger(LIBRARY_LOGGER)
    library.handlers.clear()  # drop the opt-in NullHandler
    library.addHandler(InterceptHandler())
    library.setLevel(lvl)
    library.propagate = False
