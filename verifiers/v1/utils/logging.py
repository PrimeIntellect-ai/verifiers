"""Logging for the library and its CLIs.

Library modules log through `get_logger(__name__)`. It prefixes each line with its
scope: `[<episode>]` inside an episode, `[<episode>/<trace> <agent>]` inside an agent
run inside an episode, and `[<trace>]` for an agent run outside an episode. The
prefix comes from contextvars, so call sites log plain messages and asyncio tasks
inherit the scope they start in. Ids are shortened to `ID_CHARS`: a prefix of the
full id, so it still greps against stored traces.

The library (`verifiers.v1`) logs via stdlib logging and is silent by default
(a NullHandler on the package root). A CLI opts in: it points loguru at
stderr — so logs never mix with the results printed to stdout — and installs an
`InterceptHandler` on the `verifiers.v1` logger so those records render through
loguru. Mirrors prime-rl's `intercept_vf_logging`.
"""

import inspect
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

LIBRARY_LOGGER = "verifiers.v1"
FORMAT = (
    "<dim>{time:HH:mm:ss}</dim> <level>{level: >7}</level> <level>{message}</level>"
)

ID_CHARS = 8

_episode: ContextVar[str | None] = ContextVar("vf_log_episode", default=None)
_trace: ContextVar[str | None] = ContextVar("vf_log_trace", default=None)
_agent: ContextVar[str | None] = ContextVar("vf_log_agent", default=None)
_VARS = (_episode, _trace, _agent)

LogScope = tuple[str | None, str | None, str | None]


class _ScopeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        episode, trace, agent = current_scope()
        path = "/".join(part for part in (episode, trace) if part)
        prefix = " ".join(part for part in (path, agent) if part)
        if prefix:
            record.msg = f"[{prefix}] {record.getMessage()}"
            record.args = ()
        return True


_FILTER = _ScopeFilter()


def get_logger(name: str) -> logging.Logger:
    """The module logger for `name`, prefixing records with the current scope."""
    module_logger = logging.getLogger(name)
    if _FILTER not in module_logger.filters:
        module_logger.addFilter(_FILTER)
    return module_logger


def short_id(id: str) -> str:
    return id[:ID_CHARS]


def clip(message: str, limit: int = 200) -> str:
    """The first line of an error message, short enough for a log line."""
    line = message.strip().splitlines()[0] if message.strip() else ""
    return line if len(line) <= limit else f"{line[: limit - 3]}..."


def current_scope() -> LogScope:
    return _episode.get(), _trace.get(), _agent.get()


@contextmanager
def scope(
    *, episode: str | None = None, trace: str | None = None, agent: str | None = None
) -> Iterator[None]:
    """Tag lines logged inside with an episode, or with an agent run's trace and
    agent name. A trace scope always sets the agent name, so a nested run without
    one doesn't show its parent's name."""
    tokens = []
    if episode is not None:
        tokens.append((_episode, _episode.set(episode)))
    if trace is not None:
        tokens.append((_trace, _trace.set(trace)))
        tokens.append((_agent, _agent.set(agent)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def enter_scope(saved: LogScope) -> None:
    """Enter a scope captured with `current_scope()` for the rest of the current task
    (e.g. a server handler serving that rollout). Wrap the task's work in `isolated()`
    when the task serves more than one rollout."""
    for var, value in zip(_VARS, saved):
        var.set(value)


@contextmanager
def isolated() -> Iterator[None]:
    """Undo any scope entered inside on exit."""
    tokens = [(var, var.set(None)) for var in _VARS]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class InterceptHandler(logging.Handler):
    """Forward stdlib log records into loguru, preserving level and call site."""

    def emit(self, record: logging.LogRecord) -> None:
        from loguru import logger

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Step out of this handler, then past stdlib logging's frames, to the call site.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Route the library's stdlib logs through loguru, to stderr and, when given, a
    `log_file`."""
    # Library modules import this file, and only the CLIs need loguru.
    from loguru import logger

    lvl = level.upper()
    logger.remove()
    logger.add(sys.stderr, level=lvl, format=FORMAT)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=lvl, format=FORMAT)
    library = logging.getLogger(LIBRARY_LOGGER)
    library.handlers.clear()  # drop the opt-in NullHandler
    library.addHandler(InterceptHandler())
    library.setLevel(lvl)
    library.propagate = False
