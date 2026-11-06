"""Scope prefixes for the library's log lines.

An episode tags every line logged inside it `[<episode>]`. An agent run inside an
episode extends that to `[<episode>/<trace> <agent>]`, and an agent run outside an
episode logs `[<trace>]`. The prefix comes from contextvars, so call sites log
plain messages and asyncio tasks inherit the scope they start in. Ids are
shortened to `ID_CHARS`: a prefix of the full id, so it still greps against
stored traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

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
    logger = logging.getLogger(name)
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
    return logger


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
